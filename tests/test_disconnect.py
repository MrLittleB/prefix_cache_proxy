"""用户端断开 → 上游中止 测试（修复：流式时检测断开并关闭上游）。"""
import sys, os, threading, socketserver, http.server, time, json, socket
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from prefix_hash_router.app import build_router
import prefix_hash_router.ingress.server as isv


class _SSE(http.server.BaseHTTPRequestHandler):
    """慢流式上游，记录是否被中断(write_fail)。"""
    protocol_version = "HTTP/1.1"
    broken = False
    def do_POST(self):
        _SSE.broken = False
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for i in range(200):
                p = f"data: {{'x':{i}}}\n\n".encode()
                self.wfile.write(f"{len(p):x}\r\n".encode() + p + b"\r\n")
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            _SSE.broken = True
        self.close_connection = True
    def log_message(self, *a): pass


def _start():
    ups = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _SSE)
    ups.daemon_threads = True
    threading.Thread(target=ups.serve_forever, daemon=True).start()
    up_port = ups.server_address[1]

    isv.ROUTER = build_router(dp_size=8)
    opts = isv.ForwardOptions(upstream=f"http://127.0.0.1:{up_port}")
    isv.DISPATCHER = lambda m,p,h,b,be,cs: isv.forward_response(m,p,h,b,be,opts,cs)
    px = isv._ThreadingServer(("127.0.0.1", 0), isv._Handler)
    px.daemon_threads = True
    threading.Thread(target=px.serve_forever, daemon=True).start()
    return up_port, px.server_address[1], ups


def test_user_disconnect_aborts_upstream():
    up_port, proxy_port, ups = _start()
    try:
        body = json.dumps({"messages": []}).encode()
        s = socket.create_connection(("127.0.0.1", proxy_port), timeout=5)
        req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost:x\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(body)}\r\n\r\n").encode() + body
        s.sendall(req)
        s.settimeout(0.5)
        got = b""
        try:
            while len(got) < 200:
                try:
                    d = s.recv(4096)
                    if not d: break
                    got += d
                except socket.timeout: break
        except Exception:
            pass
        # 用户真实断开(TCP FIN)
        s.close()
        time.sleep(1.0)
        assert _SSE.broken is True, "用户断开后上游仍空跑！(bug)"
    finally:
        ups.shutdown()


def test_stream_complete_when_user_stays():
    """用户不断开 → 完整读到 [DONE]。"""
    up_port, proxy_port, ups = _start()
    try:
        body = json.dumps({"messages": []}).encode()
        s = socket.create_connection(("127.0.0.1", proxy_port), timeout=5)
        req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost:x\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(body)}\r\n\r\n").encode() + body
        s.sendall(req)
        s.settimeout(5)
        full = b""
        while True:
            try:
                d = s.recv(4096)
                if not d: break
                full += d
            except socket.timeout:
                break
        s.close()
        # 上游只有200块无[DONE]，应读到相当多块且不卡死；此处验证能读满且不抛错
        assert len(full) > 500  # 应读到大量数据
        assert _SSE.broken is False  # 用户没断开，上游不应 broken
    finally:
        ups.shutdown()


def test_stream_ends_when_upstream_finishes():
    """上游正常发完 [DONE] 并 chunked 结束 → 代理应 close 连接(无 Content-Length)，
    客户端靠 EOF 确定流结束，不能一直等导致超时(修复 'Deep diving... 不结束')。"""
    class _FiniteSSE(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def do_POST(self):
            n = int(self.headers.get("content-length", 0) or 0); self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for e in ["data: Deep diving...\n\n", "data: 结果\n\n", "data: [DONE]\n\n"]:
                b = e.encode()
                self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            self.close_connection = True
        def log_message(self, *a): pass

    ups = socketserver.TCPServer(("127.0.0.1", 0), _FiniteSSE)
    threading.Thread(target=ups.serve_forever, daemon=True).start()
    px = None
    try:
        isv.ROUTER = build_router(dp_size=8)
        opts = isv.ForwardOptions(upstream=f"http://127.0.0.1:{ups.server_address[1]}", timeout=5)
        isv.DISPATCHER = lambda m,p,h,b,be,cs: isv.forward_response(m,p,h,b,be,opts,cs)
        px = isv._ThreadingServer(("127.0.0.1", 0), isv._Handler)
        threading.Thread(target=px.serve_forever, daemon=True).start()
        body = json.dumps({"messages": []}).encode()
        s = socket.create_connection(("127.0.0.1", px.server_address[1]), timeout=3)
        req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost:x\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(body)}\r\n\r\n").encode() + body
        s.sendall(req)
        s.settimeout(5)
        full = b""
        got_eof = False
        t0 = time.time()
        while time.time() - t0 < 4:
            try:
                d = s.recv(4096)
                if not d:
                    got_eof = True   # 收到 EOF → 连接被 close
                    break
                full += d
            except socket.timeout:
                break
        s.close()
        assert b"[DONE]" in full, "应收到 [DONE]"
        assert got_eof, "无 Content-Length 的流式响应后，代理应 close 连接(否则客户端等EOF超时)"
    finally:
        ups.shutdown()
        if px: px.shutdown()


if __name__ == "__main__":
    _SSE.broken = False
    test_user_disconnect_aborts_upstream()
    _SSE.broken = False
    test_stream_complete_when_user_stays()
    test_stream_ends_when_upstream_finishes()
    print("disconnect 全部测试通过")
