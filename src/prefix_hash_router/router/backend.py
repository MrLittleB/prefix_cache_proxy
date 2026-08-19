"""Backend：路由目标抽象（不再只是 int rank）。

从"返回 rank 整数"升级为"返回目标后端"，为未来加多上游/健康检查/非 DP 场景留空间。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Backend:
    dp_rank: int
    inject_rank: bool = True
    # 未来可扩展: upstream_url: str | None, healthy: bool, timeout: int ... 等

    def __repr__(self) -> str:
        return f"Backend(dp_rank={self.dp_rank}, inject_rank={self.inject_rank})"
