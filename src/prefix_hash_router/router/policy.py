"""RoutingPolicy 接口 + ChainedPolicy 组合。

策略 = 可插拔 Policy 类的集合（而非 if/elif）。每个 Policy 实现 route(ctx)->Backend。
route() 返回 None 表示"本策略不适用，交给下一个"。
"""
from __future__ import annotations

from typing import Protocol

from .context import RequestContext
from .backend import Backend


class RoutingPolicy(Protocol):
    def route(self, ctx: RequestContext) -> Backend | None: ...


class ChainedPolicy:
    """按优先级依次尝试策略，直到返回非 None 的 Backend。"""

    def __init__(self, policies: list[RoutingPolicy]):
        if not policies:
            raise ValueError("ChainedPolicy needs at least one policy")
        self._policies = policies

    def route(self, ctx: RequestContext) -> Backend:
        for p in self._policies:
            b = p.route(ctx)
            if b is not None:
                return b
        raise RuntimeError("no policy produced a backend")

    def __repr__(self) -> str:
        return f"ChainedPolicy({[type(p).__name__ for p in self._policies]})"
