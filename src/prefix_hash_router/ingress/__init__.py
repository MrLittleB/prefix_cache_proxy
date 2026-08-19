"""Ingress（接入层）：收 HTTP 请求、读 body、构造 RequestContext，并组合 Router+Dispatcher。"""
from .server import run_server

__all__ = ["run_server"]
