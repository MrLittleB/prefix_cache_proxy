"""SessionAffinityPolicy：有显式会话/routing key 时用它（最强粘性，且避免共享前缀塌缩）。

参考 SGLang #31170 `prefix_affinity`：
  - 不同 session 给不同 x-smg-routing-key（或等价会话键）→ 各自落不同 rank，
    从根本上避免"共享大 system 前缀"把大量无关会话挤到同一 rank 的热点。
  - 用 HRW(Rendezvous) 哈希，rank 用稳定 id 评分，拓扑变化只重映射受影响者。
若无显式 key，返回 None，交给下一个策略（如 PrefixHash —— 仅作无 key 时的 fallback）。
"""
from __future__ import annotations

from ..context import RequestContext
from ..backend import Backend
from ._keys import extract_session_key
from ._hash import hrw_hash_rank


class SessionAffinityPolicy:
    def __init__(self, dp_size: int):
        self._dp = dp_size

    def route(self, ctx: RequestContext) -> Backend | None:
        key = extract_session_key(ctx)
        if key is None:
            return None
        return Backend(dp_rank=hrw_hash_rank(key, self._dp))
