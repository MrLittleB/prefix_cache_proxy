"""Ingress 端到端 smoke：HTTP 请求 → Router 决策 → Dispatcher 转发 → 回传。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import http.server, socketserver, threading, json, urllib.request, urllib.error

from prefix_hash_router.app import build_router
from prefix_hash_router.ingress.server import run_server, _ThreadingServer, _Handler

# 临时 mock 上游：回显收到的 rank 头
class _Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0) or 0)
        body = self.rfile.read(n) if n else b""
        resp = json.dumps({"rank": self.headers.get("X-data-parallel-rank")}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a): pass


def test_e2e():
    # 1) 起 mock 上游
    ups = socketserver.TCPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=ups.serve_forever, daemon=True).start()
    up_port = ups.server_address[1]

    # 2) 起 Ingress（路由器指向该上游）
    router = build_router(dp_size=8, mode="prefix_hash")
    # monkey 注入到 server 模块全局
    import prefix_hash_router.ingress.server as isv
    isv.ROUTER = router
    opts = isv.ForwardOptions(upstream=f"http://127.0.0.1:{up_port}")
    isv.DISPATCHER = lambda m,p,hs,b,be,cs: isv.forward_response(m,p,hs,b,be,opts,cs)
    srv = _ThreadingServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    in_port = srv.server_address[1]

    try:
        def send(system):
            body = json.dumps({"messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": "hi"},
            ]}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{in_port}/v1/chat/completions",
                data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
            return json.loads(urllib.request.urlopen(req, timeout=10).read())

        # 相同 system 前缀 → 上游收到的 rank 一致
        r1 = send("共享前缀" * 50)
        r2 = send("共享前缀" * 50)
        assert r1["rank"] == r2["rank"]
        assert r1["rank"] is not None
    finally:
        srv.shutdown(); ups.shutdown()


def test_body_size_limit():
    """请求体超过 MAX_BODY_SIZE → 返回 413，不转发。"""
    import prefix_hash_router.ingress.server as isv
    # 设一个很小的上限
    old = isv.MAX_BODY_SIZE
    isv.MAX_BODY_SIZE = 100
    try:
        ups = socketserver.TCPServer(("127.0.0.1", 0), _Upstream)
        threading.Thread(target=ups.serve_forever, daemon=True).start()
        up_port = ups.server_address[1]
        isv.ROUTER = build_router(dp_size=8)
        opts = isv.ForwardOptions(upstream=f"http://127.0.0.1:{up_port}")
        isv.DISPATCHER = lambda m,p,hs,b,be,cs: isv.forward_response(m,p,hs,b,be,opts,cs)
        srv = _ThreadingServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        in_port = srv.server_address[1]
        try:
            # 超过100字节的 body
            big = json.dumps({"messages":[{"role":"user","content":"x"*300}]}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{in_port}/v1/chat/completions",
                data=big, headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=10)
                raise AssertionError("should have got 413")
            except urllib.error.HTTPError as e:
                assert e.code == 413, e.code
                return  # 通过
        finally:
            srv.shutdown(); ups.shutdown()
    finally:
        isv.MAX_BODY_SIZE = old


def test_chat_only_injects_rank():
    """仅为 chat/completions 加 rank 头；/models、/embeddings、GET 等非 chat 请求透传不加。"""
    import prefix_hash_router.ingress.server as isv

    class _Echo(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            resp = json.dumps({"rank": self.headers.get("X-data-parallel-rank")}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(resp)))
            self.end_headers(); self.wfile.write(resp)
        def do_POST(self):
            n = int(self.headers.get("content-length", 0) or 0); self.rfile.read(n)
            resp = json.dumps({"rank": self.headers.get("X-data-parallel-rank")}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(resp)))
            self.end_headers(); self.wfile.write(resp)
        def log_message(self, *a): pass

    ups = socketserver.TCPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=ups.serve_forever, daemon=True).start()
    isv.ROUTER = build_router(dp_size=8)
    opts = isv.ForwardOptions(upstream=f"http://127.0.0.1:{ups.server_address[1]}")
    isv.DISPATCHER = lambda m,p,hs,b,be,cs: isv.forward_response(m,p,hs,b,be,opts,cs)
    srv = _ThreadingServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def get(path):
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5).read())
    def post(path, body=None):
        if body is None:
            body = {"messages": [{"role": "user", "content": "hi"}]}
        data = json.dumps(body).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=5).read())

    try:
        # 三个消息接口(OpenAI chat / Anthropic messages / OpenAI responses)都加 rank
        assert post("/v1/chat/completions")["rank"] is not None
        assert post("/chat/completions")["rank"] is not None
        assert post("/v1/messages", {"system": "s", "messages": [{"role": "user", "content": "hi"}]})["rank"] is not None
        assert post("/v1/responses", {"instructions": "r", "input": "hello"})["rank"] is not None
        # 透传接口不加 rank
        assert post("/v1/embeddings", {})["rank"] is None
        assert post("/v1/completions", {})["rank"] is None
        assert post("/v1/models", {})["rank"] is None
        assert get("/v1/models")["rank"] is None
        assert get("/models")["rank"] is None
    finally:
        srv.shutdown(); ups.shutdown()


if __name__ == "__main__":
    test_e2e()
    test_body_size_limit()
    test_chat_only_injects_rank()
    print("ingress e2e 全部通过")
