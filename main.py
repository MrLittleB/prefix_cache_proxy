#!/usr/bin/env python3
"""prefix-hash-router 命令行入口。

用法示例：
  python main.py --dp-size 8 --mode prefix_hash --port 38294
  python main.py --mode first_rank --port 38294   # 对照：固定 rank 0
  python main.py --mode round_robin               # 对照：纯轮转
  # 默认自动从 vLLM /metrics 拉取每 DP 的 running/waiting，做过载保护；--no-overload 关闭

配置来源与优先级（由高到低）：
  1. 启动命令行参数（--upstream/--metrics-token/host/port 等）
  2. .env 文件（同一进程目录下，UPSTREAM/METRICS_TOKEN/HOST/PORT/... 等键）
  3. 代码内置默认值

  即：命令行显式传了就用命令行的；没传则看 .env；再没有才用代码默认。
"""
from __future__ import annotations

import sys
import os
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.prefix_hash_router.app import build_router, set_radix_debug
from src.prefix_hash_router.ingress.server import run_server
from src.prefix_hash_router.router.policies.overload import set_overload_debug
from src.prefix_hash_router.dispatcher.forward import set_forward_debug
from src.prefix_hash_router.metrics_collector import MetricsCollector
from env_loader import (
    load_env_values,
    str_or_none,
    int_or_default,
    float_or_default,
)


def _metrics_url_from_upstream(upstream: str) -> str:
    """上游 base(到 /v1) → 同 host 的 /metrics。"""
    up = urllib.parse.urlsplit(upstream)
    hostport = up.netloc
    return f"{up.scheme}://{hostport}/metrics"


def _bool_from_env(val: str | None, default: bool) -> bool:
    """把 .env 字符串转 bool；非法/缺失回退 default。"""
    if val is None:
        return default
    v = val.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def main():
    import argparse
    # 读取 .env（优先级：命令行参数 > .env > 代码默认值）
    _env = load_env_values()

    ap = argparse.ArgumentParser(description="prefix-hash-router")
    ap.add_argument("--host", default=str_or_none(_env.get("HOST")) or "0.0.0.0")
    ap.add_argument("--port", type=int, default=int_or_default(_env.get("PORT"), 38294))
    # 注意：upstream 的默认值不再硬编码真实地址，改由 .env 提供（若配置了）
    ap.add_argument("--upstream",
                    default=str_or_none(_env.get("UPSTREAM")) or "http://localhost:9100/v1")
    ap.add_argument("--metrics-url", default=str_or_none(_env.get("METRICS_URL")),
                    help="vLLM /metrics 地址(默认由 --upstream 推导)；用于每 DP running/waiting 过载保护")
    ap.add_argument("--metrics-token", default=str_or_none(_env.get("METRICS_TOKEN")),
                    help="拉取/拉 /metrics 时用的 Bearer token(若 vLLM /metrics 需要鉴权)")
    ap.add_argument("--dp-size", type=int, default=int_or_default(_env.get("DP_SIZE"), 8))
    ap.add_argument("--mode", choices=["prefix_hash", "consistent_hash", "radix", "first_rank", "round_robin"],
                    default=str_or_none(_env.get("MODE")) or "prefix_hash")
    ap.add_argument("--waiting-weight", type=float, default=float_or_default(_env.get("WAITING_WEIGHT"), 4.0),
                    help="waiting 的渐进权重上限(默认4：第1个waiting权重2、第2个3、第3个起4，避免小waiting被夸大)")
    ap.add_argument("--load-skew", type=float, default=float_or_default(_env.get("LOAD_SKEW"), 1.5))
    ap.add_argument("--min-load-for-overload", type=float, default=float_or_default(_env.get("MIN_LOAD_FOR_OVERLOAD"), 5.0),
                    help="过载绝对阈值(running+waiting)：只有首选 rank 负载>=该值才 spill；低于该值不干预, 保持 radix 前缀一致(默认5)")
    ap.add_argument("--no-overload", action="store_true", help="关闭基于真实 metrics 的过载保护")
    ap.add_argument("--max-body-size", type=int, default=int_or_default(_env.get("MAX_BODY_SIZE"), 0),
                    help="请求体大小上限(字节)，超过返回413；0=不限制(默认)")
    ap.add_argument("--max-workers", type=int, default=int_or_default(_env.get("MAX_WORKERS"), 256),
                    help="并发处理上限(线程池大小)，防线程爆炸")
    ap.add_argument("--radix-max-nodes", type=int, default=int_or_default(_env.get("RADIX_MAX_NODES"), 100000),
                    help="radix 树消息段上限(默认100000，LRU淘汰后保持<=此值)；设None可完全关闭淘汰")
    # --debug-rank 默认值来自 .env 的 DEBUG_RANK（true/false）；CLI 显式传则覆盖。
    # argparse 的 store_true 只有在显式给出时才置 True，因此 CLI 天然优先于该默认值。
    ap.add_argument("--debug-rank", action="store_true", default=_bool_from_env(_env.get("DEBUG_RANK"), False),
                    help="打印精简路由诊断([route] 行: rank/radix命中/会话键)")
    # --deep-debug: 独立的深度调试开关（radix-debug/forward-debug/overload SPILL），
    # 与 --debug-rank 解耦，默认关闭以减少日志量。
    ap.add_argument("--deep-debug", action="store_true", default=_bool_from_env(_env.get("DEEP_DEBUG"), False),
                    help="开启深度调试([radix-debug]/[forward-debug]/[overload] SPILL)；默认关, 与 --debug-rank 独立")
    args = ap.parse_args()

    metrics_url = args.metrics_url or _metrics_url_from_upstream(args.upstream)

    load_provider = None
    mc = None
    if not args.no_overload:
        mc = MetricsCollector(
            metrics_url, dp_size=args.dp_size, waiting_weight=args.waiting_weight,
            auth_token=args.metrics_token,
        )
        # 启动前先同步取一次，确认 vLLM 就绪；失败不放弃，仍启动后台轮询线程，
        # 待 vLLM 就绪后自动恢复过载保护（_has_data=False 时 mc.load() 返回 None，
        # OverloadGuard 退化为纯路由，安全）。
        try:
            mc.fetch_once()
            print(f"[info] 初始 metrics 拉取成功，过载保护已启用")
        except Exception as e:
            print(f"[warn] 初始 metrics 拉取失败(vLLM 可能尚未就绪)，后台轮询线程将继续重试: {e}")
        mc.start()
        load_provider = mc.load
        # 诊断：实际 vLLM engine 数 vs 配置 dp_size（发现 dp 配置错误导致漏服务）
        try:
            d = mc.diagnosis()
            if d.get("mismatch"):
                print(f"[warn] 检测到实际 engine 数({d['actual_dp']}) > 配置 dp_size({d['configured_dp']})！"
                      f" 这些 DP({d['dropped']}) 不会被路由到。请用 --dp-size 修正为 {d['actual_dp']}。")
            else:
                print(f"[info] vLLM engine 数 = {d.get('actual_dp')}，与 --dp-size {d['configured_dp']} 匹配")
        except Exception:
            pass

    # 深度调试由独立的 --deep-debug 控制（与精简的 --debug-rank 解耦）
    if args.deep_debug:
        set_radix_debug(True)
        set_overload_debug(True)
        set_forward_debug(True)

    router = build_router(dp_size=args.dp_size, mode=args.mode,
                          load_provider=load_provider, load_skew=args.load_skew,
                          min_load=args.min_load_for_overload,
                          radix_max_nodes=args.radix_max_nodes)
    # radix 模式：提示当前树上限（可观测内存有界）
    if args.mode == "radix":
        print(f"[info] radix 模式启用，树节点上限 = {router.radix_tree.max_nodes}"
              f" (--radix-max-nodes 可调；LRU 淘汰保持有界)")
    try:
        run_server(router, host=args.host, port=args.port, upstream=args.upstream,
                   max_body_size=args.max_body_size, max_workers=args.max_workers,
                   debug_rank=args.debug_rank)
    finally:
        if mc:
            mc.stop()


if __name__ == "__main__":
    main()
