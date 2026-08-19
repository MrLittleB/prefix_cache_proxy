"""Dispatcher 测试：透传、rank 头注入、hop-by-hop 过滤、路径构造。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import http.server, socketserver, threading, json
from prefix_hash_router.dispatcher.forward import (
    ForwardOptions, build_target_path, forward_response, HOP_BY_HOP,
)
from prefix_hash_router.router.backend import Backend


def test_build_target_path():
    assert build_target_path("/v1/chat/completions?foo=1") == "/v1/chat/completions?foo=1"


def test_build_target_path_with_upstream_base():
    """上游 base(/v1) 与客户端 path 的前缀合并逻辑。"""
    # 客户端 path 已包含 base → 原样透传
    assert build_target_path("/v1/chat/completions", "/v1") == "/v1/chat/completions"
    assert build_target_path("/v1/chat/completions", "/v1/") == "/v1/chat/completions"
    # 客户端 path 未包含 base → prepend
    assert build_target_path("/chat/completions", "/v1") == "/v1/chat/completions"
    assert build_target_path("/chat/completions?x=1", "/v1") == "/v1/chat/completions?x=1"
    # 边界：/v10 不应被误判为命中 base /v1
    assert build_target_path("/v10/foo", "/v1") == "/v1/v10/foo"
    # base == "/" 不加前缀
    assert build_target_path("/chat/completions", "/") == "/chat/completions"
    # 无 base 参数 → 不变
    assert build_target_path("/v1/chat/completions") == "/v1/chat/completions"


def test_hop_by_hop():
    assert "authorization" not in HOP_BY_HOP
    assert "connection" in HOP_BY_HOP


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0) or 0)
        body = self.rfile.read(n) if n else b""
        resp = json.dumps({
            "rank": self.headers.get("X-data-parallel-rank"),
            "auth": self.headers.get("Authorization"),
            "body_len": len(body),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


def _start_echo():
    srv = socketserver.TCPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_forward_injects_rank_and_preserves_auth():
    srv, port = _start_echo()
    try:
        opts = ForwardOptions(upstream=f"http://127.0.0.1:{port}")
        status, _, _, body_iter = forward_response(
            "POST", "/v1/chat/completions",
            {"Authorization": "Bearer secret", "Content-Type": "application/json"},
            b'{"m":1}', Backend(dp_rank=3), opts,
        )
        data = json.loads(b"".join(body_iter))
        assert status == 200
        assert data["rank"] == "3"
        assert data["auth"] == "Bearer secret"
    finally:
        srv.shutdown()


def test_no_backend_no_rank_header():
    srv, port = _start_echo()
    try:
        opts = ForwardOptions(upstream=f"http://127.0.0.1:{port}")
        _, _, _, body_iter = forward_response(
            "POST", "/v1/chat/completions", {"Content-Type": "application/json"},
            b"{}", Backend(dp_rank=0, inject_rank=False), opts,
        )
        data = json.loads(b"".join(body_iter))
        assert data["rank"] is None
    finally:
        srv.shutdown()


def test_forward_sets_single_host_no_conflict():
    """转发时 Host 头应只设一个(上游 host)，不残留客户端透传的 host 导致 Host 冲突。

    复现 bug: 客户端 headers 透传了 'host'(小写), 后加 'Host'(大写), http.client
    会重复发送两个 host 键, 上游按 Host 匹配时返回 404。
    """
    class _HostEcho(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            host_keys = [k for k in self.headers.keys() if k.lower() == "host"]
            resp = json.dumps({"host_keys": host_keys,
                               "host": self.headers.get("Host")}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        def log_message(self, *a): pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), _HostEcho)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        opts = ForwardOptions(upstream=f"http://127.0.0.1:{port}")
        # 模拟 Ingress 透传的 headers：含小写 'host'(来自客户端) 和 Authorization
        headers = {
            "host": "clienthost:12345",   # 客户端原始 Host(小写, Ingress 归一化后)
            "Authorization": "Bearer x",
        }
        _, _, _, body_iter = forward_response("GET", "/v1/models", headers,
                                              None, None, opts)
        data = json.loads(b"".join(body_iter))
        # 上游只应收到一个 host 键, 且为上游 host(不含端口)
        assert len(data["host_keys"]) == 1, f"应有1个host头, 实际 {data['host_keys']}"
        assert data["host"] == "127.0.0.1", f"Host 应为上游host, 实际 {data['host']!r}"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    test_build_target_path()
    test_build_target_path_with_upstream_base()
    test_hop_by_hop()
    test_forward_injects_rank_and_preserves_auth()
    test_no_backend_no_rank_header()
    test_forward_sets_single_host_no_conflict()
    print("dispatcher 全部测试通过")
