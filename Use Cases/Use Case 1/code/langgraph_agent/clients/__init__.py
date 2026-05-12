"""
Client-side tool access abstractions for the LangGraph agent.

This package is the transition seam between the orchestration layer and
the execution layer. Today it can call local Python tools, but the same
interface can later be backed by an MCP client without changing node logic.
"""

from .tool_gateway import ToolGateway, LocalToolGateway, get_tool_gateway

__all__ = [
    "ToolGateway",
    "LocalToolGateway",
    "get_tool_gateway",
]
