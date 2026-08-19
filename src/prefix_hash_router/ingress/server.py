"""Ingress · HTTP Server：接收请求 → 构造 RequestContext → Router → Dispatcher → 回传。

三层串联（依赖注入，便于测试）：
  - router: 可调用 route(ctx) -> Backend
  - dispatcher: 可调用 forward_response(...) -> (status, reason, headers, body_iter)
"""
from __future__ import annotations

import http.server
import json
import socketserver
import sys
import threading
import time
from typing import Callable

from ..router.context import RequestContext
from ..router.policies._keys import extract_session_key
from ..dispatcher.forward import ForwardOptions, forward_response


def _ts() -> str:
    """返回日志时间戳前缀，如 '[2026-08-19 10:30:00]'。"""
    return time.strftime("[%Y-%m-%d %H:%M:%S]")

# 是否打印每个请求的详细路由诊断(方法/路径→rank/radix命中/会话键)。由 run_server 设置。
DEBUG_RANK: bool = False

# 需要前缀路由(加 X-data-parallel-rank 头)的真实消息生成接口(POST)。
# 基于 vLLM 实际暴露的路由(见启动日志)精确白名单，兼容带/不带 /v1 前缀。
# 只对"真实带消息的生成"做前缀缓存路由；render/derender/count_tokens/查询/cancel/
# models/health/批量等要么非生成、要么结构特殊/异步，均透传不加 rank。
def _is_chat_request(command: str, path: str) -> bool:
    """判断是否是需要前缀路由(加 X-data-parallel-rank 头)的消息类请求。

    加 rank 头（做前缀缓存路由）：
      - /v1/chat/completions   (OpenAI chat)
      - /v1/messages           (Anthropic)
      - /v1/responses          (OpenAI Responses)
    这些是真实、带 messages 的在线生成，适合前缀缓存路由。
    其它(/models、/health、/embeddings、/completions、render/derender/count_tokens、
    batch、查询/cancel、审计、GET/OPTIONS/HEAD 等)透传不加 rank，vLLM 自己负载均衡。
    """
    if command != "POST":
        return False
    p = path.split("?", 1)[0].rstrip("/")
    segs = [s for s in p.split("/") if s]
    # 去掉可选 /v1 版本前缀：/v1/chat/completions → chat/completions
    if segs and segs[0] == "v1":
        rel = segs[1:]
        # 精确匹配消息接口（只匹配"恰好两层/一层"的生成端点，不误匹配 batch/render 等）
        if len(rel) == 1 and rel[0] in ("messages", "responses"):
            return True
        if len(rel) == 2 and rel[0] == "chat" and rel[1] == "completions":
            return True
        return False
    # 无 /v1 前缀
    if len(segs) == 1 and segs[0] in ("messages", "responses"):
        return True
    if len(segs) == 2 and segs[0] == "chat" and segs[1] == "completions":
        return True
    return False

# 运行期注入（由 run_server 设置）
ROUTER: Callable | None = None
DISPATCHER: Callable | None = None
MAX_BODY_SIZE: int = 0   # 0=不限制；>0 时超过即返回 413

# 请求体超限的哨兵，用于 _handle 识别并返回 413
_BODY_TOO_LARGE = object()


def _read_body(h: http.server.BaseHTTPRequestHandler):
    """读取请求体；超过 MAX_BODY_SIZE 时返回 _BODY_TOO_LARGE。"""
    limit = MAX_BODY_SIZE
    if "transfer-encoding" in h.headers and h.headers["transfer-encoding"].lower() in ("chunked",):
        data = b""
        while True:
            line = h.rfile.readline().strip()
            # 允许 chunk 扩展（RFC7230 §4.1.1：size[;ext]），如 "5;foo=bar"
            size_token = line.split(b";", 1)[0].strip()
            try:
                size = int(size_token, 16)
            except ValueError:
                break
            if size == 0:
                h.rfile.readline()
                break
            data += h.rfile.read(size)
            if limit and len(data) > limit:
                return _BODY_TOO_LARGE
            h.rfile.readline()
        return data
    length = int(h.headers.get("content-length", 0) or 0)
    if limit and length > limit:
        return _BODY_TOO_LARGE
    if length > 0:
        data = h.rfile.read(length)
        if limit and len(data) > limit:
            return _BODY_TOO_LARGE
        return data
    return None


def _pre_check_radix_hit(ctx: RequestContext) -> bool | dict | None:
    """决策前探测 radix 是否命中(用于准确诊断). 非 radix 模式或无法判断返回 None.

    返回：
      None   : 非 radix 模式 / 无法判断
      False  : radix 未命中(将走冷启动)
      dict   : radix 命中明细（lookup_debug 结果），含 walk/found_depth/break_index/
               break_seg/segs/full_leaf，用于区分"完整命中/正常末尾追加/异常中间断档"。
    """
    router = ROUTER
    if router is None or not hasattr(router, "radix_tree"):
        return None
    msgs = ctx.messages
    if not msgs or not isinstance(msgs, list):
        return False
    # 用 lookup_debug 拿命中/断点明细（含 break_index/break_seg），从而区分末尾追加与中间断档
    detail = router.radix_tree.lookup_debug(msgs)
    return detail if detail is not None else False


def _log_route_diagnosis(command: str, path: str, ctx: RequestContext, backend,
                         pre_hit: bool | dict | None = None) -> None:
    """打印每个 chat 请求的路由诊断：分配到的 rank、策略来源、是否 radix 命中。
    仅 DEBUG_RANK 开启时调用。backend 为 None 表示走了透传(异常情况)。
    pre_hit: 决策前 radix 命中明细（dict=命中, False=冷启动, None=非radix）。
    """
    if backend is None:
        sys.stderr.write(f"{_ts()} [route] {command} {path} -> NoBackend(透传/无路由)\n")
        return
    rank = backend.dp_rank
    msgs = ctx.messages
    n_msgs = len(msgs) if isinstance(msgs, list) else 0
    sk = extract_session_key(ctx)
    sk_str = f" session={sk!r}" if sk else ""
    radix_hit_str = ""
    if isinstance(pre_hit, dict):
        d = pre_hit
        segs = d.get("segs")
        walk = d.get("walk")
        found_rank = d.get("found_rank")
        found_depth = d.get("found_depth")
        break_index = d.get("break_index")
        full_leaf = d.get("full_leaf")
        effective_rank = d.get("effective_rank")
        subtree_rank = d.get("subtree_rank")
        if full_leaf:
            # 情况3: 命中叶子节点 → 过载保护
            radix_hit_str = f" radix命中(leaf walk={walk}/{segs} rank={rank})"
        elif walk == 0:
            # 情况1: 完全未命中 (walk=0) → argmin(最闲)
            radix_hit_str = (
                f" radix冷启动(完全未命中 walk=0/{segs}) → argmin rank={rank}"
            )
        elif effective_rank is not None:
            # effective_rank 存在 = radix 命中(情况2.1子树回退 / 情况2.2本应但这里不可能 / 末尾追加)
            if subtree_rank is not None:
                # 情况2.1: 子树回退 (walk/total > 50%) → 过载保护
                radix_hit_str = (
                    f" radix命中(子树回退 walk={walk}/{segs} subtree_rank={subtree_rank}) "
                    f"final-rank={rank}"
                )
            elif found_rank is not None and break_index >= walk:
                # 末尾追加: found_rank 非空且断点在末尾
                radix_hit_str = (
                    f" radix命中(末尾追加 walk={walk}/{segs} miss={segs-walk} "
                    f"rank={found_rank}->{rank})"
                )
            else:
                # 不应到达: effective_rank 存在但既非子树回退也非末尾追加
                radix_hit_str = (
                    f" radix命中(未知 walk={walk}/{segs} found={found_rank} "
                    f"subtree={subtree_rank} eff={effective_rank}) final-rank={rank}"
                )
        else:
            # effective_rank=None: radix 未命中 → argmin(最闲)
            if walk > 0 and segs > 0 and walk / segs <= 0.5:
                # 情况2.2: 共享前缀占比 ≤ 50% → 冷启动(argmin)
                radix_hit_str = (
                    f" radix冷启动(前缀占比≤50% walk={walk}/{segs}) → argmin rank={rank}"
                )
            else:
                # 真正的异常中间断档 (walk>0, effective_rank=None, walk/segs > 50% 但子树也没rank)
                radix_hit_str = (
                    f" radix冷启动(中间断档 walk={walk}/{segs} found={found_rank}) → argmin rank={rank}"
                )
            
    elif pre_hit is False:
        radix_hit_str = f" radix冷启动(无消息) → rank={rank}"
    sys.stderr.write(
        f"{_ts()} [route] {command} {path} -> rank={rank}{radix_hit_str}{sk_str} msgs={n_msgs}\n"
    )


def _from_request(h: http.server.BaseHTTPRequestHandler, raw: bytes | None) -> RequestContext:
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
    return RequestContext(
        # 统一转小写：HTTP header 名不区分大小写（如 X-Session-Id vs x-session-id），
        # dict 的 .get() 是大小写敏感的，必须归一化，否则 extract_session_key 等按小写匹配会漏。
        headers={k.lower(): v for k, v in h.headers.items()},
        raw_body=raw or b"",
        parsed_body=parsed,
    )


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s [ingress] %s %s\n" % (_ts(), self.command, self.path))

    def _handle(self):
        raw = _read_body(self)
        if raw is _BODY_TOO_LARGE:
            # 请求体超限 → 413，不转发
            self.send_response(413, "Payload Too Large")
            body = b'{"error":"request body too large"}'
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Router/Dispatcher 内部异常 → 显式 500，避免客户端悬挂/半开
        try:
            ctx = _from_request(self, raw)
            # 仅 chat/completions 才做前缀路由(加 rank 头)；其它(/models、/health等)直接透传
            if _is_chat_request(self.command, self.path):
                # 决策前探测 radix 是否命中(用于准确诊断"本次是命中还是冷启动")
                pre_hit = _pre_check_radix_hit(ctx)
                backend = ROUTER(ctx) if ROUTER else None
                if DEBUG_RANK:
                    _log_route_diagnosis(self.command, self.path, ctx, backend, pre_hit)
            else:
                backend = None
                if DEBUG_RANK:
                    sys.stderr.write(f"{_ts()} [route] {self.command} {self.path} -> 透传(非chat, 不加rank)\n")
            status, reason, resp_headers, body_iter = DISPATCHER(
                self.command, self.path, ctx.headers, raw, backend, self.connection,
            )
        except Exception as e:
            try:
                sys.stderr.write(f"{_ts()} [ingress] internal error: {type(e).__name__}: {e}\n")
                body = json.dumps({"error": {"type": "internal_error",
                                             "message": f"{type(e).__name__}: {e}"}}).encode()
                self.send_response(500, "Internal Server Error")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                pass
            return

        # 判断响应是否有 Content-Length；若没有(流式/SSE/无界)，客户端只能靠连接关闭
        # 确定 body 结束 → 必须真正 close_connection，否则客户端(如 curl)会一直等导致超时。
        has_content_length = any(h.lower() == "content-length" for h, _ in resp_headers)
        if not has_content_length:
            self.close_connection = True

        self.send_response_only(status, reason)
        for hk, hv in resp_headers:
            self.send_header(hk, hv)
        self.end_headers()
        try:
            for chunk in body_iter:
                # 若发送时用户已断开，write 会抛错
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                if hasattr(body_iter, "close"):
                    body_iter.close()
            except Exception:
                pass

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle
    do_OPTIONS = _handle
    do_HEAD = _handle   # HEAD 也透传(不加 rank)，避免健康检查/探活返回 501


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """有界并发线程数：防止高并发下 ThreadingMixIn 无限起线程把内存打爆（review 2.5）。"""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args, max_workers: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        # 在服务器主循环里先用信号量限流：并发连接达到上限后，新连接阻塞在 acquire，
        # 等某个 worker 结束才放行，从而控制同时处理的连接数。
        self._sema = threading.BoundedSemaphore(max_workers)

    def process_request(self, request, client_address):
        # 先占并发槽；线程真正结束（process_request_thread)时 release。
        self._sema.acquire()
        try:
            super().process_request(request, client_address)
        except Exception:
            # 线程创建失败等极端情况要释放信号量，避免槽位泄漏
            self._sema.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._sema.release()


def run_server(
    router: Callable,
    *,
    host: str = "0.0.0.0",
    port: int = 38294,
    upstream: str = "http://localhost:9100/v1",
    timeout: int = 600,
    max_body_size: int = 0,
    max_workers: int = 256,
    debug_rank: bool = False,
):
    global ROUTER, DISPATCHER, MAX_BODY_SIZE, DEBUG_RANK
    ROUTER = router
    MAX_BODY_SIZE = max_body_size  # 0 = 不限制
    DEBUG_RANK = debug_rank        # 打印每个请求的路由诊断(rank/radix命中/会话键)
    opts = ForwardOptions(upstream=upstream, timeout=timeout)
    DISPATCHER = lambda m, p, hs, b, be, cs: forward_response(m, p, hs, b, be, opts, cs)
    # 读超时：防止慢客户端/半开连接无限阻塞占线程（socketserver 用类属性做 settimeout）
    _Handler.timeout = timeout
    httpd = _ThreadingServer((host, port), _Handler, max_workers=max_workers)
    print("=" * 60)
    print("prefix-hash-router Ingress 已启动")
    print("  监听: %s:%d" % (host, port))
    print("  上游: %s" % upstream)
    print("  读/上游超时: %ss" % timeout)
    print("  并发上限: %s" % max_workers)
    print("  请求体上限: %s" % (f"{max_body_size} bytes" if max_body_size else "不限制"))
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[ingress] stopped")
