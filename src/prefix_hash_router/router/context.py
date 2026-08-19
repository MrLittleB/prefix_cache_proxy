"""RequestContext：统一请求中间结构，解耦 Router 与 HTTP 框架。"""
from __future__ import annotations

from dataclasses import dataclass, field


# 稳定指令前缀的角色：system / developer（如 OpenAI responses 的 developer message、
# ChatGPT 的 developer role）都应作为稳定前缀开头，避免共享失效。
_INSTRUCTION_ROLES = ("system", "developer")


def _has_instruction_role(msgs: list) -> bool:
    """messages 列表里是否已有稳定指令前缀消息(system/developer)。

    用于判断是否需要前置 Anthropic 顶层 system / Responses instructions——
    若 messages 已含则没必要重复。
    """
    for m in msgs:
        if isinstance(m, dict) and m.get("role") in _INSTRUCTION_ROLES:
            return True
    return False


@dataclass
class RequestContext:
    headers: dict[str, str] = field(default_factory=dict)
    raw_body: bytes = b""
    parsed_body: dict | None = None
    session_key: str | None = None

    @property
    def messages(self) -> list | None:
        """从请求体提取标准化消息序列（[{role, content}, ...]），供 radix/prefix 统一使用。

        支持三种消息类接口格式，均向前兼容"标准 messages"原样返回：
          1. OpenAI chat        : {messages:[...]}              → 原样返回 messages
          2. Anthropic messages : {system:..., messages:[...]}   → system(置前) + messages
          3. OpenAI responses   : {instructions:..., input:...}  → instructions(置前,system) + input
        parsed_body 可能不是 dict（如客户端发 JSON 字符串/数组），防御性处理避免崩溃。
        """
        if not self.parsed_body or not isinstance(self.parsed_body, dict):
            return None
        body = self.parsed_body

        # 格式2: Anthropic — 顶层 system + messages
        if "messages" in body:
            msgs = body["messages"]
            if not isinstance(msgs, list):
                return None
            # 顶层 system 存在且不是空 → 前置为 system 消息（Anthropic 特有）
            sysv = body.get("system")
            if sysv and not _has_instruction_role(msgs):
                return [{"role": "system", "content": sysv}] + msgs
            return msgs

        # 格式3: OpenAI responses — instructions(置前) + input
        if "input" in body:
            inp = body["input"]
            prompt = body.get("instructions")
            head = []
            if prompt and not _has_instruction_role(
                    inp if isinstance(inp, list) else []):
                head = [{"role": "system", "content": prompt}]
            if isinstance(inp, str):
                return head + [{"role": "user", "content": inp}]
            if isinstance(inp, list):
                return head + list(inp)
            return head or None

        return None
