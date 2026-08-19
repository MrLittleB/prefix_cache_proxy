"""组装应用：构建 Router（ChainedPolicy + 策略），供 Ingress 使用。"""
from __future__ import annotations

import random
import time

from .router.context import RequestContext
from .router.backend import Backend
from .router import ChainedPolicy
from .router.policies import (
    SessionAffinityPolicy, PrefixHashPolicy,
    ConsistentHashPolicy, RoundRobinPolicy, OverloadGuard,
    PrefixRadixTree, RadixPrefixPolicy,
)

from .router.policies._keys import extract_session_key, extract_prefix_key

# 深度 radix 诊断开关（由 main.py 在 --debug-rank 时置 True）。
# 用于定位"为什么命中深度比上次学到的少"等疑难问题。
_RADIX_DEBUG = False


def set_radix_debug(on: bool = True) -> None:
    """开启/关闭 radix 深度诊断输出。"""
    global _RADIX_DEBUG
    _RADIX_DEBUG = on


def _radix_debug_trace(ctx: RequestContext, backend: Backend,
                       pre: dict, post: dict) -> None:
    """打印 radix 决策前/后的深度诊断，精确定位命中深度异常。

    pre : 决策前 lookup_debug 结果（能看到断点在哪、断的是哪段）
    post: 学习后 lookup_debug 结果（能看到刚学入的路径是否完整）
    """
    import sys
    msgs = ctx.messages or []
    sk = extract_session_key(ctx)
    sk_str = f" session={sk!r}" if sk else ""
    segs = [s for s in (_msg_segment_safe(m) for m in msgs) if s[1]]
    print(
        f"[radix-debug] rank={backend.dp_rank}{sk_str} "
        f"msgs条数={len(msgs)} 非空段={len(segs)} "
        f"PRE walk={pre.get('walk')} found={pre.get('found_rank')}@{pre.get('found_depth')} "
        f"break_idx={pre.get('break_index')}/{pre.get('segs')} break_seg={pre.get('break_seg')} "
        f"POST walk={post.get('walk')} found={post.get('found_rank')}@{post.get('found_depth')} "
        f"post_break={post.get('break_index')}",
        file=sys.stderr,
    )


def _msg_segment_safe(m) -> tuple:
    """供诊断打印用的段提取（与树内一致但绝不抛异常）。"""
    from .router.policies.radix import _msg_segment
    try:
        return _msg_segment(m)
    except Exception:
        return ("", "")


class RadixFallbackPolicy:
    """radix 未命中时的冷启动分配：最闲 rank 优先。

    到达此策略的条件：无 session key（SessionAffinity 未处理）+ radix 未命中
    （完全未匹配 / 共享前缀占比 ≤ 50%）。这些请求没有 KV cache 依赖，
    直接分配到最闲 rank 即可，避免空闲 DP 浪费。

    有负载数据时：argmin(load)，多个最闲随机选。
    无负载数据时：随机分配（启动初期的短暂窗口）。
    """

    def __init__(self, dp_size: int, load=None, load_provider=None):
        self._dp = dp_size
        self._static_load = load
        self._provider = load_provider

    def route(self, ctx: RequestContext) -> Backend | None:
        """radix 未命中(完全未匹配 / 共享前缀占比 ≤ 50%)时的冷启动分配。

        到达此策略的请求一定无 session key（有的话第一层 SessionAffinity 已处理），
        也一定 radix 未命中。策略：选最闲 rank（argmin load），多个最闲随机选；
        无负载数据时随机分配。

        与"radix 命中"的区别：radix 命中走 OverloadGuard（保护 KV 一致性，
        过载才 spill）；未命中无 KV 依赖，直接填最闲节点即可。
        """
        load = []
        if self._provider is not None:
            load = self._provider() or []
        else:
            load = self._static_load or []
        if load:
            min_load = min(load)
            idle = [i for i in range(len(load)) if load[i] == min_load]
            return Backend(dp_rank=random.choice(idle))
        return Backend(dp_rank=random.randrange(self._dp))


# radix 树默认消息段上限（LRU 淘汰后树保持 <= 此值）。
# 默认 100000：单个对话可能有几百~一千条消息，需容得下几十个长对话 + 大量短会话，
# 同时内存仍可控（10万节点约几十 MB 量级），且避免无限增长。
# 可通过 build_router(radix_max_nodes=...) 或 --radix-max-nodes 调整。
DEFAULT_RADIX_MAX_NODES = 100_000


def build_router(
    dp_size: int = 8,
    mode: str = "prefix_hash",
    load: list[int] | None = None,
    load_provider=None,
    load_skew: float = 1.5,
    min_load: float = 5.0,
    radix_max_nodes: int | None = DEFAULT_RADIX_MAX_NODES,
):
    """组装一个可调用 ctx->Backend 的 Router。

    mode: prefix_hash | consistent_hash | radix | first_rank | round_robin
      - radix: 用 PrefixRadixTree 做最长连续前缀精确匹配（有状态、需学习回填）。
               冷启动(树无该前缀)时回退到 round-robin，并在每次决策后把完整
               messages 归属本次最终 rank（学习回填）。
    load:          静态各 rank 负载（测试用；load 与 load_provider 二选一）
    load_provider: 可调用 -> 当前各 rank 负载（生产用，从 MetricsCollector 实时取，
                   running/waiting 加权后喂进来；见 metrics_collector.MetricsCollector.load）
    load_skew:     过载阈值系数（用于 OverloadGuard 的 spill 判定）
    min_load:      过载绝对阈值（running+waiting）。只有首选 rank 负载 >= min_load
                   才可能 spill；低于该值一律不干预，保持 radix 前缀一致性（默认 5）。
    radix_max_nodes: radix 树消息段上限（默认 DEFAULT_RADIX_MAX_NODES=100000）；LRU 淘汰后保持 <= 此值。
    """
    if dp_size <= 0:
        raise ValueError(f"dp_size must be >= 1, got {dp_size}")
    if load_skew <= 0:
        raise ValueError(f"load_skew must be > 0, got {load_skew}")
    if min_load <= 0:
        raise ValueError(f"min_load must be > 0, got {min_load}")
    # 静态 load 长度必须与 dp_size 一致，否则过载保护里 % len(load) 会静默归一化导致分布错乱
    if load is not None and len(load) != dp_size:
        raise ValueError(
            f"load 长度({len(load)}) 必须等于 dp_size({dp_size})；"
            f"load 是每 DP 的负载，必须一一对应，否则过载保护会用错 rank。"
        )

    use_radix = mode == "radix"
    radix_tree = PrefixRadixTree(max_nodes=radix_max_nodes)
    radix_policy = RadixPrefixPolicy(dp_size, tree=radix_tree)

    if mode == "consistent_hash":
        core = ConsistentHashPolicy(dp_size)
    elif mode == "round_robin":
        core = RoundRobinPolicy(dp_size)
    elif mode == "first_rank":
        core = FirstRankPolicy(dp_size)
    elif use_radix:
        # radix 命中时返回 rank；未命中(冷启动)返回 None 交给 fallback(round-robin)
        core = radix_policy
    else:
        core = PrefixHashPolicy(dp_size)

    # radix 模式在核心未命中时用 RadixFallbackPolicy 兜底冷启动
    # (有会话键 HRW 稳定分散 / 无会话键随机打散 + 过载感知, 避免顺序轮转/共享system塌缩/落过载)
    fallback = (RadixFallbackPolicy(dp_size, load=load, load_provider=load_provider) if use_radix else None)

    def _atom(ctx: RequestContext):
        # 过载 spill 用的原子键：session key 优先，否则 prefix key
        return extract_session_key(ctx) or extract_prefix_key(ctx)

    # 包装 load_provider：运行时校验返回长度 == dp_size，避免过载保护 rank 静默错配
    if load_provider is not None:
        orig_provider = load_provider
        _warn_once = {"flag": False}
        def _checked_provider():
            l = orig_provider()
            if l is not None and len(l) != dp_size and not _warn_once["flag"]:
                import sys
                sys.stderr.write(
                    f"{time.strftime('[%Y-%m-%d %H:%M:%S]')} [warn] load_provider 返回长度"
                    f"({len(l)}) != dp_size({dp_size}); 本轮忽略该次负载，过载保护可能不生效\n")
                _warn_once["flag"] = True
            return l
        load_provider = _checked_provider

    # 组装策略链：SessionAffinity(若有会话键) → core(radix) → round-robin兜底
    session_policy = SessionAffinityPolicy(dp_size)
    if use_radix:
        if has_load_for(load, load_provider):
            policies = [
                OverloadGuard(session_policy, load=load, load_provider=load_provider,
                              load_skew=load_skew, min_load=min_load, atom_from_ctx=_atom),
                OverloadGuard(core, load=load, load_provider=load_provider,
                              load_skew=load_skew, min_load=min_load, atom_from_ctx=_atom),
                fallback,
            ]
        else:
            policies = [session_policy, core, fallback]
    else:
        if has_load_for(load, load_provider):
            policies = [
                OverloadGuard(session_policy, load=load, load_provider=load_provider,
                              load_skew=load_skew, min_load=min_load, atom_from_ctx=_atom),
                OverloadGuard(core, load=load, load_provider=load_provider,
                              load_skew=load_skew, min_load=min_load, atom_from_ctx=_atom),
            ]
        else:
            policies = [session_policy, core]

    chain = ChainedPolicy(policies)

    # radix 模式命中率/分布统计（可观测：判断 radix 是否有效、有无热点）
    radix_stats = {"total": 0, "hits": 0, "misses": 0, "ranks": {}}

    def router(ctx: RequestContext) -> Backend:
        if use_radix:
            radix_stats["total"] += 1
            # 决策前快照：一次 lookup 同时服务 ①命中统计 ②(debug)断点诊断 ③决策依据。
            # 让统计与决策基于同一个"决策前时刻"的 radix 状态，避免中间 learn 导致不一致。
            if _RADIX_DEBUG:
                pre_lookup = radix_tree.lookup_debug(ctx.messages or [])
                # lookup_debug 恒返回 dict，found_rank 是否为 None 即"radix 是否命中"
                # found_rank 为 None 但 subtree_rank 非 None 时也算命中(情况2子树回退)
                hit = pre_lookup.get("found_rank") is not None or pre_lookup.get("subtree_rank") is not None
            else:
                # 非 debug：仅需"是否命中"做统计，不额外做断点诊断
                hit = radix_policy.route(ctx) is not None
            radix_stats["hits" if hit else "misses"] += 1
            b = chain.route(ctx)
            # 学习回填：把完整 messages 归属最终实际 rank
            msgs = ctx.messages
            if msgs and isinstance(msgs, list):
                radix_tree.learn(msgs, b.dp_rank)
            radix_stats["ranks"][b.dp_rank] = radix_stats["ranks"].get(b.dp_rank, 0) + 1
            if _RADIX_DEBUG:
                post_lookup = radix_tree.lookup_debug(msgs or [])
                _radix_debug_trace(ctx, b, pre_lookup, post_lookup)
            return b
        return chain.route(ctx)

    # 仅 radix 模式暴露 radix 树与统计给调用方（便于诊断/测试/持久化）；
    # 非 radix 模式无这些属性。
    if use_radix:
        router.radix_tree = radix_tree
        router.radix_stats = radix_stats
    return router


def has_load_for(load, load_provider) -> bool:
    return load is not None or load_provider is not None


class FirstRankPolicy:
    """固定到 rank 0（用于对照/调试）。"""
    def __init__(self, dp_size: int):
        self._dp = dp_size

    def route(self, ctx: RequestContext) -> Backend | None:
        return Backend(dp_rank=0)
