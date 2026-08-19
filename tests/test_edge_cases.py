"""边界/异常测试：空 body、无 messages、dp_size 校验、一致性哈希均匀性、键提取边界。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from prefix_hash_router.router.context import RequestContext
from prefix_hash_router.router import ChainedPolicy
from prefix_hash_router.router.policies import (
    PrefixHashPolicy, SessionAffinityPolicy, ConsistentHashPolicy, RoundRobinPolicy,
)
from prefix_hash_router.router.policies._hash import hash_to_rank, consistent_hash_ring
from prefix_hash_router.router.policies._keys import (
    extract_session_key, extract_session_key_debug, extract_prefix_key,
)
from prefix_hash_router.ingress.server import _read_body, _is_chat_request


def _ctx(parsed=None, headers=None):
    return RequestContext(headers=headers or {}, raw_body=b"", parsed_body=parsed, session_key=None)


def test_empty_body_prefix_key():
    p = ChainedPolicy([SessionAffinityPolicy(8), PrefixHashPolicy(8)])
    r = p.route(_ctx(parsed={}))
    assert 0 <= r.dp_rank < 8
    r2 = p.route(_ctx(parsed={"messages": []}))
    assert 0 <= r2.dp_rank < 8


def test_empty_body_consistent_rank():
    p = ChainedPolicy([SessionAffinityPolicy(8), PrefixHashPolicy(8)])
    assert p.route(_ctx(parsed={})).dp_rank == p.route(_ctx(parsed={})).dp_rank


def test_hash_invalid_dp():
    for fn in (lambda: hash_to_rank("k", 0), lambda: consistent_hash_ring("k", -1)):
        try:
            fn()
            raise AssertionError("should have raised")
        except ValueError:
            pass


def test_consistent_ring_uniform():
    counts = [0]*8
    for i in range(256):
        counts[consistent_hash_ring(f"key-{i}", 8)] += 1
    assert min(counts) > 0
    assert max(counts) < 32*3


def test_session_key_from_body_and_header():
    assert extract_session_key(_ctx(parsed={"session_id": "abc"}, headers={})) == "abc"
    assert extract_session_key(_ctx(parsed={}, headers={"x-session-id": "hdr"})) == "hdr"
    assert extract_session_key(_ctx(parsed={}, headers={})) is None


def test_prefix_key_only_system():
    key = extract_prefix_key(_ctx(parsed={"messages": [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]}))
    assert "SYS" in key
    assert "USER" not in key


def test_prefix_key_multimodal_content():
    """多模态 content(list) 仅取文本部分作哈希键，排除二进制/图像。"""
    key = extract_prefix_key(_ctx(parsed={"messages": [
        {"role": "system", "content": [
            {"type": "text", "text": "多模态系统提示"},
            {"type": "image_url", "image_url": {"url": "data:image/..."}},
        ]},
    ]}))
    assert "多模态系统提示" in key
    assert "image" not in key
    assert "data:image" not in key


def test_session_key_lowercase_headers_normalized():
    """server 侧会把 header 统一转小写，extract_session_key 应能匹配常见小写形式。"""
    assert extract_session_key(_ctx(parsed={}, headers={"x-session-id": "hdr"})) == "hdr"
    assert extract_session_key(_ctx(parsed={}, headers={"x-smg-routing-key": "routing-1"})) == "routing-1"
    assert extract_session_key(_ctx(parsed={}, headers={"x-session_id": "underscore"})) == "underscore"


def test_conversation_key_header():
    """OpenClaw / Clawdbot / Claude / Codex / DeepSeek 常用 x-conversation-id。"""
    assert extract_session_key(_ctx(parsed={}, headers={"x-conversation-id": "conv-1"})) == "conv-1"
    assert extract_session_key(_ctx(parsed={}, headers={"x-conversation_id": "conv-2"})) == "conv-2"
    assert extract_session_key(_ctx(parsed={}, headers={"conversation-id": "conv-3"})) == "conv-3"


def test_conversation_key_body():
    """DSH / DeepSeek / OpenClaw body 里带 conversation_id。"""
    assert extract_session_key(_ctx(parsed={"conversation_id": "c1"}, headers={})) == "c1"
    assert extract_session_key(_ctx(parsed={"conversationId": "c2"}, headers={})) == "c2"


def test_thread_key():
    """OpenAI Assistants / 线程化客户端：thread_id / threadId。"""
    assert extract_session_key(_ctx(parsed={"thread_id": "t1"}, headers={})) == "t1"
    assert extract_session_key(_ctx(parsed={"threadId": "t2"}, headers={})) == "t2"
    assert extract_session_key(_ctx(parsed={}, headers={"x-thread-id": "t3"})) == "t3"
    assert extract_session_key(_ctx(parsed={}, headers={"x-thread_id": "t4"})) == "t4"


def test_pi_session_key():
    """Pi (Inflection)：x-pi-session-id。"""
    assert extract_session_key(_ctx(parsed={}, headers={"x-pi-session-id": "pi-1"})) == "pi-1"


def test_user_key():
    """OpenAI 兼容：user_id / x-user-id 同用户粘同一 rank。"""
    assert extract_session_key(_ctx(parsed={"user_id": "u1"}, headers={})) == "u1"
    assert extract_session_key(_ctx(parsed={"userId": "u2"}, headers={})) == "u2"
    assert extract_session_key(_ctx(parsed={}, headers={"x-user-id": "u3"})) == "u3"


def test_header_priority_over_body():
    """header 的会话键优先于 body 的会话键（粘性更强、更显式）。"""
    assert extract_session_key(_ctx(
        parsed={"conversation_id": "body-conv"},
        headers={"x-session-id": "header-sess"},
    )) == "header-sess"


def test_prompt_cache_key_still_works():
    """prompt_cache_key（OpenRouter 兜底链）仍应命中。"""
    assert extract_session_key(_ctx(parsed={"prompt_cache_key": "pck-1"}, headers={})) == "pck-1"


def test_session_key_debug():
    """diagnostic 版本在有会话键时应打印来源，无键时返回 None 不抛错。"""
    import io, contextlib
    # 有键：打印 stderr，返回该键
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        key = extract_session_key_debug(_ctx(parsed={}, headers={"x-conversation-id": "c9"}))
    assert key == "c9"
    assert "header 'x-conversation-id'" in buf.getvalue()
    # 无键：打印 no found，返回 None
    buf2 = io.StringIO()
    with contextlib.redirect_stderr(buf2):
        key2 = extract_session_key_debug(_ctx(parsed={}, headers={}))
    assert key2 is None
    assert "no session key found" in buf2.getvalue()
    # 关闭诊断：不打印但仍返回
    buf3 = io.StringIO()
    with contextlib.redirect_stderr(buf3):
        key3 = extract_session_key_debug(_ctx(parsed={}, headers={"x-session-id": "s1"}), enabled=False)
    assert key3 == "s1"
    assert buf3.getvalue() == ""


def test_chained_requires_policies():
    try:
        ChainedPolicy([])
        raise AssertionError("should have raised")
    except ValueError:
        pass


# ---- chunked body 解析（server._read_body）----
class _FakeRFile:
    def __init__(self, data):
        self.data = data
        self.pos = 0
    def readline(self):
        end = self.data.find(b"\r\n", self.pos)
        if end == -1:
            end = len(self.data)
        line = self.data[self.pos:end + 2]
        self.pos = end + 2
        return line
    def read(self, n):
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk


class _FakeHeaders(dict):
    pass


class _FakeHandler:
    def __init__(self, data):
        self.headers = _FakeHeaders({"transfer-encoding": "chunked"})
        self.rfile = _FakeRFile(data)


def test_chunked_body_plain():
    assert _read_body(_FakeHandler(b"5\r\nhello\r\n0\r\n\r\n")) == b"hello"


def test_chunked_body_with_chunk_extensions():
    """RFC7230 §4.1.1: chunk size 允许带扩展 (size;ext)。修复: 之前会解析失败返回空。"""
    result = _read_body(_FakeHandler(b"5;ext=value\r\nhello\r\n6;foo=bar\r\n world\r\n0\r\n\r\n"))
    assert result == b"hello world", result


def test_chunked_body_multiple():
    assert _read_body(_FakeHandler(b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n")) == b"abcde"


def test_non_chunked_no_content_length():
    # 无 transfer-encoding、无 content-length → None（无 body）
    class _FakeH2(_FakeHandler):
        def __init__(self):
            self.headers = _FakeHeaders({})
            self.rfile = _FakeRFile(b"")
    assert _read_body(_FakeH2()) is None


# ---- 消息接口多格式提取（messages 标准化）----
def test_messages_openai_chat():
    """OpenAI chat：messages 原样返回。"""
    ctx = _ctx(parsed={"messages": [{"role": "user", "content": "hi"}]})
    assert ctx.messages == [{"role": "user", "content": "hi"}]


def test_messages_anthropic_system_prepended():
    """Anthropic messages：顶层 system 前置，不重复已含 system。"""
    ctx = _ctx(parsed={"system": "sys-x", "messages": [{"role": "user", "content": "hi"}]})
    assert ctx.messages == [{"role": "system", "content": "sys-x"},
                            {"role": "user", "content": "hi"}]
    # messages 已含 system 则不重复
    ctx2 = _ctx(parsed={"system": "sys-x",
                        "messages": [{"role": "system", "content": "in-msg"}]})
    assert ctx2.messages[0] == {"role": "system", "content": "in-msg"}
    assert len(ctx2.messages) == 1


def test_messages_openai_responses():
    """OpenAI responses：instructions 作 system 前置 + input。"""
    ctx = _ctx(parsed={"instructions": "rule", "input": "hello"})
    assert ctx.messages == [{"role": "system", "content": "rule"},
                            {"role": "user", "content": "hello"}]
    # input 是 list
    ctx2 = _ctx(parsed={"input": [{"role": "user", "content": "a"}]})
    assert ctx2.messages == [{"role": "user", "content": "a"}]
    # 无 instructions 无 input → None
    assert _ctx(parsed={"model": "m"}).messages is None


# ---- 接口分类（哪些加 rank 头 / 哪些透传）----
def test_is_chat_request_classification():
    """精确白名单：仅 POST 消息生成接口加 rank；真实 vLLM 路由其它全部透传。"""
    # 消息生成接口 → True
    assert _is_chat_request("POST", "/v1/chat/completions")
    assert _is_chat_request("POST", "/chat/completions")
    assert _is_chat_request("POST", "/v1/messages")
    assert _is_chat_request("POST", "/messages")
    assert _is_chat_request("POST", "/v1/responses")
    assert _is_chat_request("POST", "/responses")
    # 真实 vLLM 边界变体(基于启动日志) → 透传 False
    for p in ["/v1/chat/completions/batch", "/v1/chat/completions/render",
              "/v1/chat/completions/derender", "/v1/completions/render",
              "/v1/completions/derender", "/v1/messages/count_tokens"]:
        assert not _is_chat_request("POST", p), f"{p} 应透传"
    # 查询/取消 responses
    assert not _is_chat_request("GET", "/v1/responses/{response_id}")
    assert not _is_chat_request("POST", "/v1/responses/{response_id}/cancel")
    # 其它真实生成/工具 → 透传
    for p in ["/v1/completions", "/invocations", "/tokenize", "/detokenize",
              "/generative_scoring", "/scale_elastic_ep", "/is_scaling_elastic_ep",
              "/inference/v1/generate", "/load", "/version", "/v1/models", "/health"]:
        assert not _is_chat_request("POST", p), f"{p} 应透传"
    # 防误判：/x/messages 不是消息生成
    assert not _is_chat_request("POST", "/foo/messages")
    assert not _is_chat_request("POST", "/bar/responses")
    # 非 POST → False
    assert not _is_chat_request("GET", "/v1/chat/completions")
    assert not _is_chat_request("OPTIONS", "/v1/messages")
    assert not _is_chat_request("HEAD", "/v1/models")


if __name__ == "__main__":
    test_empty_body_prefix_key()
    test_empty_body_consistent_rank()
    test_hash_invalid_dp()
    test_consistent_ring_uniform()
    test_session_key_from_body_and_header()
    test_prefix_key_only_system()
    test_prefix_key_multimodal_content()
    test_session_key_lowercase_headers_normalized()
    test_conversation_key_header()
    test_conversation_key_body()
    test_thread_key()
    test_pi_session_key()
    test_user_key()
    test_header_priority_over_body()
    test_prompt_cache_key_still_works()
    test_session_key_debug()
    test_chained_requires_policies()
    test_chunked_body_plain()
    test_chunked_body_with_chunk_extensions()
    test_chunked_body_multiple()
    test_non_chunked_no_content_length()
    test_messages_openai_chat()
    test_messages_anthropic_system_prepended()
    test_messages_openai_responses()
    test_is_chat_request_classification()
    print("edge cases 全部测试通过")
