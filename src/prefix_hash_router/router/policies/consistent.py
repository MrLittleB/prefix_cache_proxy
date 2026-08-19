"""ConsistentHashPolicy：一致性哈希环，DP 扩缩容时最小重映射（演进项）。"""
from __future__ import annotations

from ..context import RequestContext
from ..backend import Backend
from ._keys import extract_prefix_key
from ._hash import consistent_hash_ring


class ConsistentHashPolicy:
    def __init__(self, dp_size: int, replicas: int = 100):
        self._dp = dp_size
        self._replicas = replicas

    def route(self, ctx: RequestContext) -> Backend | None:
        key = extract_prefix_key(ctx)
        return Backend(dp_rank=consistent_hash_ring(key, self._dp, self._replicas))
