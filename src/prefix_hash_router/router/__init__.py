"""Router（路由核心）：纯逻辑，RequestContext -> Backend，与 HTTP 无关。"""
from .context import RequestContext
from .backend import Backend
from .policy import RoutingPolicy, ChainedPolicy

__all__ = ["RequestContext", "Backend", "RoutingPolicy", "ChainedPolicy"]
