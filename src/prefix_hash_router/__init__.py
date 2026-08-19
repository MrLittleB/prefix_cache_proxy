"""prefix-hash-router · 三层架构：Ingress / Router / Dispatcher。

为 vLLM 多 DP 集群实现前缀感知哈希路由，把相同前缀/会话钉到同一 DP，
注入 X-data-parallel-rank 头。
"""
__version__ = "0.1.0"

from .router.context import RequestContext
from .router.backend import Backend
from .router.policy import RoutingPolicy, ChainedPolicy

__all__ = ["RequestContext", "Backend", "RoutingPolicy", "ChainedPolicy"]
