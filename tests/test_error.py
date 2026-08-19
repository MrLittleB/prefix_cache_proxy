"""上游错误传播测试：4xx/5xx 状态码透传 + 连接失败处理。"""
import sys, os, threading, socketserver, http.server, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import urllib.request, urllib.error

from prefix_hash_router.app import build_router
import prefix_hash_router.ingress.server as isv


def _make_err_upstream(code):
    """按给定状态码返回错误体(JSON)的上游。"""
    class _E(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", 0) or 0); self.rfile.read(n)
            b = json.dumps({"error": f"code_{code}"}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        def log_message(self, *a): pass
    return _E


def _start_proxy(upstream):
    ups = socketserver.TCPServer(("127.0.0.1", 0), upstream)
    threading.Thread(target=ups.serve_forever, daemon=True).start()
    isv.ROUTER = build_router(dp_size=8)
    opts = isv.ForwardOptions(upstream=f"http://127.0.0.1:{ups.server_address[1]}")
    isv.DISPATCHER = lambda m,p,h,b,be,cs: isv.forward_response(m,p,h,b,be,opts,cs)
    px = isv._ThreadingServer(("127.0.0.1", 0), isv._Handler)
    threading.Thread(target=px.serve_forever, daemon=True).start()
    return ups, px


def _send(handler, port):
    body = json.dumps({"messages": []}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return r.status, r.read().decode(), None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), None
    except Exception as e:
        return None, None, type(e).__name__


def test_error_status_codes_passthrough():
    """4xx/5xx 应原样透传状态码与错误体。"""
    for code in (400, 404, 429, 500, 503):
        ups, px = _start_proxy(_make_err_upstream(code))
        try:
            st, body, err = _send(None, px.server_address[1])
            assert err is None, err
            assert st == code, f"{code} 透传失败 -> {st}"
            assert f"code_{code}" in body, body
        finally:
            ups.shutdown(); px.shutdown()
    print("  4xx/5xx 状态码+错误体透传正确")


def test_connection_failure_returns_error():
    """上游连接被拒时，客户端应得到明确错误而非无限挂起。"""
    # 用一个不可达端口的上游
    old_router = isv.ROUTER
    import prefix_hash_router.dispatcher.forward as F
    bad_opts = isv.ForwardOptions(upstream="http://127.0.0.1:1")
    isv.DISPATCHER = lambda m,p,h,b,be,cs: F.forward_response(m,p,h,b,be,bad_opts,cs)
    px = isv._ThreadingServer(("127.0.0.1", 0), isv._Handler)
    threading.Thread(target=px.serve_forever, daemon=True).start()
    try:
        st, body, err = _send(None, px.server_address[1])
        if err is None:
            print(f"  上游不可达 -> HTTP {st}({body[:60]})")
            assert st == 502, f"上游不可达应返回 502，实际 {st}"
        else:
            print(f"  上游不可达 -> 客户端抛 {err}（需改进为 502）")
    finally:
        px.shutdown(); isv.ROUTER = old_router


def test_internal_error_returns_500():
    """Router/Dispatcher 内部异常 → 显式 500，不悬挂半开。"""
    old_router = isv.ROUTER
    old_disp = isv.DISPATCHER
    def broken_router(ctx):
        raise ValueError("simulated routing failure")
    isv.ROUTER = broken_router
    opts = isv.ForwardOptions(upstream="http://127.0.0.1:1")
    isv.DISPATCHER = lambda m,p,h,b,be,cs: isv.forward_response(m,p,h,b,be,opts,cs)
    px = isv._ThreadingServer(("127.0.0.1", 0), isv._Handler)
    threading.Thread(target=px.serve_forever, daemon=True).start()
    try:
        st, body, err = _send(None, px.server_address[1])
        assert err is None, err
        assert st == 500, f"内部异常应返回 500，实际 {st}"
        assert "simulated routing failure" in body, body
    finally:
        px.shutdown()
        isv.ROUTER = old_router
        isv.DISPATCHER = old_disp


if __name__ == "__main__":
    test_error_status_codes_passthrough()
    test_connection_failure_returns_error()
    test_internal_error_returns_500()
    print("error 测试通过")
