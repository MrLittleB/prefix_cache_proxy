"""RoundRobinPolicy：兜底策略，轮转分配。"""
from __future__ import annotations

import itertools

from ..context import RequestContext
from ..backend import Backend


class RoundRobinPolicy:
    def __init__(self, dp_size: int):
        self._dp = dp_size
        self._counter = itertools.count(0)

    def route(self, ctx: RequestContext) -> Backend | None:
        n = next(self._counter) % self._dp
        return Backend(dp_rank=n)
