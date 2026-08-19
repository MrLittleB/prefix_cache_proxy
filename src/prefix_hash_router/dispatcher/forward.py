"""Dispatcher · 透传转发：注入 Backend 对应的 X-data-parallel-rank 头，原样转发。

支持:
  - SSE 流式逐块透传（含 [DONE]）
  - 用户端断开时主动 detect 并关闭上游连接（避免 vLLM 继续空跑）
"""
from __future__ import annotations

import http.client
import json
import select
import socket
import ssl
import urllib.parse
from dataclasses import dataclass

from ..router.backend import Backend

HOP_BY_HOP = {
    "connection", "keep-alive", "transfer-encoding", "upgrade",
    "proxy-connection", "proxy-authenticate", "proxy-authorization", "te", "trailer",
}

@dataclass
class ForwardOptions:
    upstream: str
    timeout: int = 600
    read_size: int = 8192


# 转发诊断开关（由 main.py 在 --debug-rank 时开启）。定位"502/Connection reset"等链路问题。
_FORWARD_DEBUG = False


def set_forward_debug(on: bool = True) -> None:
    """开启/关闭转发链路诊断输出。"""
    global _FORWARD_DEBUG
    _FORWARD_DEBUG = on


def _dbg(msg: str) -> None:
    if _FORWARD_DEBUG:
        import sys
        sys.stderr.write(f"[forward-debug] {msg}\n")


def _body_structure_summary(body: bytes | None) -> str:
    """提取请求体的 JSON 结构摘要（不打印真实内容，只打印字段/统计），
    用于定位是哪个特征（大字段/tool_calls/多轮等）触发上游 reset。"""
    if not body:
        return "body=None"
    import json
    try:
        data = json.loads(body)
    except Exception as e:
        return f"body=非JSON({type(e).__name__}, {len(body)}B)"
    if not isinstance(data, dict):
        return f"body=非dict({type(data).__name__}, {len(body)}B)"
    # 顶层字段（含长度）
    top = {}
    for k, v in data.items():
        if isinstance(v, str):
            top[k] = f"str({len(v)})"
        elif isinstance(v, (list, dict)):
            top[k] = f"{type(v).__name__}({len(v)})"
        elif v is None:
            top[k] = "null"
        else:
            top[k] = type(v).__name__
    # messages 统计
    msgs = data.get("messages")
    extra = ""
    if isinstance(msgs, list):
        roles = {}
        has_tool = False
        for m in msgs:
            if isinstance(m, dict):
                r = str(m.get("role", ""))
                roles[r] = roles.get(r, 0) + 1
                if m.get("tool_calls") or m.get("tool_call_id") or m.get("tool_results"):
                    has_tool = True
        extra = f" messages={len(msgs)} roles={roles} has_tool={has_tool}"
    # 响应/输入字段
    if "input" in data and isinstance(data["input"], list):
        extra += f" input_len={len(data['input'])}"
    return f"top={top}{extra} size={len(body)}B"


def _parse_upstream(upstream: str):
    up = urllib.parse.urlsplit(upstream)
    scheme = up.scheme
    host = up.hostname
    port = up.port or (443 if scheme == "https" else 80)
    base = up.path.rstrip("/")
    if not base.startswith("/"):
        base = "/" + base
    return scheme, host, port, base


def build_target_path(client_path: str, upstream_base: str = "") -> str:
    """构造转发给上游的路径：客户端路径 + 上游 base 前缀（若客户端未包含）。

    upstream_base 为上游 URL 的 path 前缀（如 '/v1'）。若客户端路径已包含该前缀
    （proxy 与上游同 path 挂载，最常见），则原样透传；否则 prepend，支持
    "代理挂在 /，上游挂在 /v1" 的 path 映射场景。
    """
    parsed = urllib.parse.urlsplit(client_path)
    path = parsed.path or "/"
    # 注意用路径边界判断（base.rstrip('/')+'/'），避免 /v10 被误判为命中 base /v1
    if upstream_base and upstream_base != "/":
        base_norm = upstream_base.rstrip("/")
        if path == base_norm or path.startswith(base_norm + "/"):
            return urllib.parse.urlunsplit(("", "", path, parsed.query, ""))
        path = base_norm + "/" + path.lstrip("/")
    return urllib.parse.urlunsplit(("", "", path, parsed.query, ""))


def forward_response(
    method: str,
    client_path: str,
    headers: dict,
    body: bytes | None,
    backend: Backend | None,
    opts: ForwardOptions,
    client_sock=None,
):
    """转发到上游，注入 backend.dp_rank 头。返回 (status, reason, resp_headers, body_iter)。

    client_sock: 用户端 socket（可选）。用于流式转发时检测用户断开：
      一旦用户断开，立即 close 上游并中止，避免 vLLM 继续空跑。
    """
    scheme, host, port, base = _parse_upstream(opts.upstream)
    target = build_target_path(client_path, upstream_base=base)

    fwd = {}
    for k, v in headers.items():
        if k.lower() in HOP_BY_HOP:
            continue
        fwd[k] = v
    if backend is not None and backend.inject_rank:
        fwd["X-data-parallel-rank"] = str(backend.dp_rank)
    # 统一设置 Host：先移除客户端透传的 host(任意大小写)以免与 Host 冲突(HTTP头不区分大小写，
    # 若同时有 'host' 和 'Host' 两个键, http.client 会重复发送导致上游 Host 匹配错误)。
    fwd.pop("host", None)
    fwd.pop("Host", None)
    fwd["Host"] = host

    full_url = f"{scheme}://{host}:{port}{target}"
    _dbg(f"完整转发地址: {full_url}")
    _dbg(f"upstream={opts.upstream} -> scheme={scheme} host={host} port={port} base={base!r} "
         f"client_path={client_path!r} target={target!r} "
         f"method={method} body_bytes={len(body) if body else 0}")
    _dbg(f"inject_rank={backend.dp_rank if backend and backend.inject_rank else '无'} "
         f"fwd_headers={ {k: (v[:40]+'...' if isinstance(v,str) and len(v)>40 else v) for k,v in fwd.items()} }")
    _dbg(f"body结构摘要: {_body_structure_summary(body)}")

    if scheme == "https":
        conn = http.client.HTTPSConnection(
            host, port, context=ssl._create_unverified_context(), timeout=opts.timeout)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=opts.timeout)

    try:
        conn.request(method, target, body=body if body else None, headers=fwd,
                     encode_chunked=False)
        resp = conn.getresponse()
        resp_headers = [(hk, hv) for hk, hv in resp.getheaders()
                        if hk.lower() not in HOP_BY_HOP]
        _dbg(f"upstream 响应: status={resp.status} reason={resp.reason} "
             f"headers={ {k:v for k,v in resp_headers} }")
    except Exception as exc:
        # 上游连接失败/超时 → 显式返回 502 + JSON 错误体（而非异常冒出导致不可靠半开）
        import traceback
        _dbg(f"!! 上游连接异常: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        try:
            conn.close()
        except Exception:
            pass
        err_body = json.dumps({"error": {"type": "upstream_unavailable",
                                         "message": f"upstream error: {exc}"}}).encode()
        resp_headers_dummy = [("Content-Type", "application/json"),
                              ("Content-Length", str(len(err_body)))]
        return 502, "Bad Gateway", resp_headers_dummy, iter([err_body])

    # 判断是否流式（SSE）：Content-Type: text/event-stream
    content_type = ""
    for k, v in resp.getheaders():
        if k.lower() == "content-type":
            content_type = v.lower()
            break
    is_stream = "text/event-stream" in content_type

    def body_iter():
        try:
            if client_sock is not None and is_stream:
                yield from _stream_with_disconnect(conn, resp, opts, client_sock)
            else:
                while True:
                    chunk = resp.read(opts.read_size)
                    if not chunk:
                        break
                    yield chunk
        finally:
            try:
                conn.close()
            except Exception:
                pass

    return resp.status, resp.reason, resp_headers, body_iter()


def _stream_with_disconnect(conn, resp, opts, client_sock):
    """流式读取上游（用 resp.read1 增量读），每块间隙非阻塞探测用户 socket 是否断开。

    一旦用户断开（对端 FIN/EOF），立即 close 上游连接并中止——让 vLLM 感知 disconnect，
    触发其 with_cancellation 取消该请求（否则 vLLM 会继续空跑，浪费算力）。

    注意：不能用 resp.read(size) + select(底层sock)，因为 http.client 有内部 buffered
    reader，数据可能已在其 buffer 里，导致 select 看不到底层可读而卡住。read1() 绕开该问题。
    """
    # read1 只在本方法用；若不可用(极端)退化为 read
    read = getattr(resp, "read1", None) or (lambda n: resp.read(n))

    while True:
        # 每块间隙先非阻塞探测用户是否断开 → 及时中止（不用等上游下一个块）
        if _client_closed(client_sock):
            return  # 用户断开 → finally 会 close 上游
        # read1 返回最多 n 字节的可用数据（有数据即返回，不会因 select 冲突卡住）
        chunk = read(opts.read_size)
        if not chunk:
            return  # 上游结束
        yield chunk
        # read1 可能一次返回多块；跨块间由下一循环顶部再次检查用户断开


def _client_closed(sock) -> bool:
    """非阻塞探测用户 socket 是否已断开：select 可读且 recv 返回 EOF 则断开。"""
    if sock is None:
        return False
    r, _, _ = select.select([sock], [], [], 0)  # 非阻塞，不等待
    if not r:
        return False  # 无可读 → 未断开
    try:
        data = sock.recv(1, socket.MSG_PEEK)
        return data == b""  # 空 → 对端 EOF/断开
    except (BlockingIOError, InterruptedError):
        return False
    except (ConnectionResetError, ConnectionAbortedError, OSError):
        return True
