"""MetricsCollector 测试：解析真实 vLLM /metrics 格式，running/waiting 加权负载。"""
import sys, os, threading, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import http.server, socketserver
from prefix_hash_router.metrics_collector import MetricsCollector
from prefix_hash_router.router.context import RequestContext
from prefix_hash_router.app import build_router


# ---- 解析测试 ----
def test_parse_weighted_load():
    sample = """
vllm:num_requests_running{engine="0",model_name="M"} 2.0
vllm:num_requests_running{engine="1",model_name="M"} 0.0
vllm:num_requests_waiting{engine="0",model_name="M"} 1.0
vllm:num_requests_waiting{engine="1",model_name="M"} 5.0
"""
    mc = MetricsCollector("dummy", dp_size=2, waiting_weight=4.0)
    mc._parse(sample)
    # 渐进权重(max_w=4): waiting=1→2, waiting=5→2+3+3*4=17
    # engine0: running2 + score(1)=2 -> 4 ; engine1: running0 + score(5)=17 -> 17
    assert mc.load() == [4.0, 17.0]


def test_parse_ignores_unknown():
    sample = "content-length 1\nvllm:other 3.0\n"
    mc = MetricsCollector("dummy", dp_size=4)
    mc._parse(sample)
    assert mc.load() == [0, 0, 0, 0]

def test_progressive_waiting_weight():
    """渐进权重：第1个waiting=2、第2个=3、第3个起=4（max_w=4时），避免小waiting被夸大。"""
    from prefix_hash_router.metrics_collector import progressive_waiting_score
    assert progressive_waiting_score(0, 4.0) == 0.0
    assert progressive_waiting_score(1, 4.0) == 2.0    # 第1个
    assert progressive_waiting_score(2, 4.0) == 5.0    # 2+3
    assert progressive_waiting_score(3, 4.0) == 9.0    # 2+3+4
    assert progressive_waiting_score(4, 4.0) == 13.0   # 2+3+4+4
    assert progressive_waiting_score(5, 4.0) == 17.0   # 2+3+4+4+4

    # 关键边界：非空载时 (base=4, running=4, waiting=1) 负载=4+2=6 不夸大
    assert 4 + progressive_waiting_score(1, 4.0) == 6.0
    # running=4, waiting=2 -> 4+5=9（确实有积压才显得高）
    assert 4 + progressive_waiting_score(2, 4.0) == 9.0


# ---- 真实 metrics HTTP 服务测试 ----
class _MetricsSrv(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = ('vllm:num_requests_running{engine="0",model_name="M"} 3.0\n'
                'vllm:num_requests_running{engine="1",model_name="M"} 0.0\n'
                'vllm:num_requests_waiting{engine="0",model_name="M"} 0.0\n'
                'vllm:num_requests_waiting{engine="1",model_name="M"} 8.0\n').encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass


def test_fetch_once_from_http():
    srv = socketserver.TCPServer(("127.0.0.1", 0), _MetricsSrv)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        mc = MetricsCollector(f"http://127.0.0.1:{port}/metrics", dp_size=2, waiting_weight=4.0)
        load = mc.fetch_once()
        # 渐进权重(max_w=4): waiting=0->0, waiting=8->2+3+6*4=29
        # engine0: running3 -> 3 ; engine1: running0 + score(8)=29 -> 29
        assert load == [3.0, 29.0], load
    finally:
        srv.shutdown()


def test_load_provider_into_router():
    """真实 running/waiting 负载喂给 build_router → 过载保护生效。"""
    # 模拟：engine1 负载极高(32)，engine0 低(3)，engine1 应过载 spill
    load = [3.0, 32.0, 0, 0, 0, 0, 0, 0]  # dp=8
    r = build_router(dp_size=8, mode="prefix_hash", load=load, load_skew=1.5)
    # 构造一个基础应落到 engine1 的前缀（若不过载会落1）
    from prefix_hash_router.router.policies._hash import hash_to_rank
    sys_text = next(f"hit-{i}"*20 for i in range(3000) if hash_to_rank(f"hit-{i}"*20, 8)==1)
    ctx = RequestContext(headers={}, raw_body=b"", parsed_body={
        "messages":[{"role":"system","content":sys_text}]})
    b = r(ctx)
    assert b.dp_rank != 1, "engine1 过载应 spill"


def test_load_none_before_data():
    """从未成功拉取时 load() 返回 None（区分"真实全0"与"无数据"），过载保护应退化。"""
    mc = MetricsCollector("dummy", dp_size=4)
    assert mc.load() is None          # 无数据
    assert mc.has_data() is False
    mc._parse("")                     # 空文本也算"解析过" → has_data True, 但全0
    assert mc.has_data() is True
    assert mc.load() == [0, 0, 0, 0]  # 真实全0


def test_main_like_fail_open_on_no_metrics():
    """模拟 main.py: 初始拉取失败 → 不启用过载保护(load_provider 保持 None)。"""
    from prefix_hash_router.metrics_collector import MetricsCollector
    # 用一个连不上的 metrics 地址
    mc = MetricsCollector("http://127.0.0.1:1/metrics", dp_size=8)
    try:
        mc.fetch_once()
        fetched = True
    except Exception:
        fetched = False
    # main.py 里：fetch 失败则 mc=None → load_provider 保持 None → 过载不启用
    load_provider = mc.load if fetched else None
    assert load_provider is None      # fail-open


def test_engine_diagnosis_mismatch():
    """dp 诊断：实际 engine 数 > 配置 dp_size → 报告 mismatch，发现漏服务。"""
    mc = MetricsCollector("dummy", dp_size=4)
    mc._parse("""
vllm:num_requests_running{engine="0"} 1
vllm:num_requests_running{engine="1"} 0
vllm:num_requests_waiting{engine="2"} 0
vllm:num_requests_waiting{engine="3"} 0
vllm:num_requests_running{engine="4"} 1
vllm:num_requests_waiting{engine="5"} 0
""")
    d = mc.diagnosis()
    assert d["configured_dp"] == 4
    assert d["observed"] == [0, 1, 2, 3]
    assert d["dropped"] == [4, 5]        # 因 idx>=dp 被丢弃
    assert d["actual_dp"] == 6
    assert d["mismatch"] is True


def test_engine_diagnosis_ok():
    """dp 诊断：配置与实际匹配 → 无 mismatch。"""
    mc = MetricsCollector("dummy", dp_size=8)
    mc._parse('vllm:num_requests_running{engine="0"} 1\nvllm:num_requests_waiting{engine="3"} 0\n')
    d = mc.diagnosis()
    assert d["mismatch"] is False
    assert d["dropped"] == []


if __name__ == "__main__":
    test_parse_weighted_load()
    test_parse_ignores_unknown()
    test_progressive_waiting_weight()
    test_fetch_once_from_http()
    test_load_provider_into_router()
    test_load_none_before_data()
    test_main_like_fail_open_on_no_metrics()
    test_engine_diagnosis_mismatch()
    test_engine_diagnosis_ok()
    print("metrics collector 全部测试通过")
