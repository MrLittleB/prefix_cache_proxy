"""PrefixHashPolicy：无 session key 时的 fallback — 对稳定共享前缀做 blake2b % DP。

注意（SGLang #26612/#31170 共识）：它只能保证"相同前缀→同 rank 命中缓存"，
无法区分"共享同一大 system 前缀的不同会话"——那些会话会塌到同一 rank（热点）。
因此：
  - 它是 fallback，不是首选；
  - 真正要分散不同会话，需用 SessionAffinityPolicy（routing key）或过载保护（OverloadGuard）。
"""
from __future__ import annotations

from ..context import RequestContext
from ..backend import Backend
from ._keys import extract_prefix_key
from ._hash import hash_to_rank


class PrefixHashPolicy:
    def __init__(self, dp_size: int, max_msgs: int = 1, max_chars: int = 500):
        self._dp = dp_size
        self._max_msgs = max_msgs
        self._max_chars = max_chars

    def route(self, ctx: RequestContext) -> Backend | None:
        key = extract_prefix_key(ctx, self._max_msgs, self._max_chars)
        return Backend(dp_rank=hash_to_rank(key, self._dp))
