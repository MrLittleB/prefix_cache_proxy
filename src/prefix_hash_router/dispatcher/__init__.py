"""Dispatcher（转发层）：把 Backend 变成实际 HTTP 转发，注入 rank 头、透传。"""
from .forward import ForwardOptions, forward_response, build_target_path

__all__ = ["ForwardOptions", "forward_response", "build_target_path"]
