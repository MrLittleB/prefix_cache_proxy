"""共享：前缀键/会话键提取。

核心（D2 难点）：哈希对象必须是**稳定共享前缀**（system 等固定 preamble），
而非每轮变化的 user tail，保证同一会话/共享 system 前缀的各轮请求 key 一致。
"""
from __future__ import annotations

from ..context import RequestContext

# 显式会话/routing key 来源（参考 SGLang x-smg-routing-key + OpenRouter 兜底链）
# x-smg-routing-key 是 SGLang 的标准会话/项目级路由键（见 #31170）。
#
# 覆盖的客户端生态（业界常见）：
#   - OpenClaw / Clawdbot:   x-conversation-id / body conversation_id
#   - Pi (Inflection):        x-session-id / session_id / x-pi-session-id
#   - Claude Code:            一般无显式会话键（靠完整 messages + cache_control），但也会带 conversation_id
#   - Codex:                  一般无显式会话键，可能带 x-conversation-id
#   - DSH / DeepSeek:         x-smg-routing-key / session_id / conversation_id
#   - OpenAI Assistants:      thread_id / threadId
#   - OpenAI 兼容:            session_id / sessionId / prompt_cache_key / user_id
#
# 顺序按"粘性好 + 特异性强"排列：routing key 优先，其次是 session/conversation/thread。
# server 侧已把 header 统一转小写，这里用小写匹配即可。
_SESSION_HEADERS = (
    # SGLang 标准：项目/会话级路由键，最推荐
    "x-smg-routing-key",
    # 通用 session
    "x-session-id", "x-session_id", "session-id",
    # conversation（OpenClaw / Claude 系 / Codex / DeepSeek）
    "x-conversation-id", "x-conversation_id", "conversation-id",
    # thread（OpenAI Assistants / 线程化客户端）
    "x-thread-id", "x-thread_id", "thread-id",
    # Pi / 其他专有
    "x-pi-session-id",
    # user 级（OpenAI 兼容，同用户粘同一 rank）
    "x-user-id", "user-id",
)
_SESSION_BODY_KEYS = (
    "session_id", "sessionId", "prompt_cache_key",
    "conversation_id", "conversationId",
    "thread_id", "threadId",
    "user_id", "userId",
)


def extract_session_key(ctx: RequestContext) -> str | None:
    for name in _SESSION_HEADERS:
        v = ctx.headers.get(name)
        if v and str(v).strip():
            return str(v).strip()
    pb = ctx.parsed_body or {}
    # 防御：parsed_body 可能不是 dict（如 JSON 字符串/数组），避免 .get 崩溃
    if isinstance(pb, dict):
        for name in _SESSION_BODY_KEYS:
            v = pb.get(name)
            if v:
                return str(v).strip()
    return None


def extract_session_key_debug(ctx: RequestContext, enabled: bool = True) -> str | None:
    """同 extract_session_key，但在命中时打印命中的键名（来源 header/body）便于实测各客户端。

    当 enabled=True 时，命中的会话键会把来源写到 stderr，方便在没有真实流量抓包的情况下，
    直观看到某客户端实际带了哪个字段，确认覆盖是否命中。
    """
    import sys
    for name in _SESSION_HEADERS:
        v = ctx.headers.get(name)
        if v and str(v).strip():
            val = str(v).strip()
            if enabled:
                sys.stderr.write(f"[session-debug] header {name!r} = {val!r}\n")
            return val
    pb = ctx.parsed_body or {}
    if isinstance(pb, dict):
        for name in _SESSION_BODY_KEYS:
            v = pb.get(name)
            if v:
                val = str(v).strip()
                if enabled:
                    sys.stderr.write(f"[session-debug] body {name!r} = {val!r}\n")
                return val
    if enabled:
        sys.stderr.write("[session-debug] no session key found\n")
    return None


def _msg_text(m) -> str:
    if isinstance(m, str):
        return m
    if isinstance(m, dict):
        content = m.get("content")
        # content 缺失或为 None（如 Anthropic assistant 停止时 content=null）
        # → 返回空串而非 str(None)="None"，避免 "None" 污染树
        if content is None:
            return ""
        # OpenAI/Anthropic 多模态：content 可能是 [{"type":"text","text":...},...]
        # 仅取文本部分作哈希键，避免把非稳定/二进制部分（如图片）纳入
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif isinstance(part, str):
                    texts.append(part)
            return "\n".join(texts)
        return str(content)
    return str(m)


def extract_prefix_key(
    ctx: RequestContext,
    max_msgs: int = 1,
    max_chars: int = 500,
    roles=("system", "developer"),
) -> str:
    """取稳定共享前缀（默认 system/developer 稳定指令角色）构造哈希键。"""
    msgs = ctx.messages or []
    # 防御：messages 若不是 list（例如字符串/整型），直接返回空，避免迭代字符导致错误 key
    if not msgs or not isinstance(msgs, list):
        return ""
    parts = []
    for m in msgs:
        role = m.get("role") if isinstance(m, dict) else None
        if role in roles and len(parts) < max_msgs:
            text = _msg_text(m).strip()
            if text:
                parts.append(text[:max_chars])
    return "\x1f".join(parts)
