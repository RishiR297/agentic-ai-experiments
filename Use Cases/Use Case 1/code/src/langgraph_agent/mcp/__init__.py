"""
MCP (Model Context Protocol) Integration Module

This module provides enhanced context preservation and reference resolution
for multi-turn medical conversations using the Model Context Protocol.

Key Components:
- MCPContextManager: Manages MCP context items and persistence
- MCPMedicalAssistantAgent: Enhanced agent with MCP integration
- MCP-enhanced processing nodes for LangGraph workflow

Features:
- Structured context item storage
- Advanced reference resolution ("her", "next patient", "that appointment")
- Relevance-based context scoring
- Temporal context decay
- Cross-session context sharing when appropriate
"""

from .context_manager import MCPContextManager, MCPContextItem
from .mcp_agent import MCPMedicalAssistantAgent
from .mcp_nodes import (
    mcp_enhanced_context_resolver_node,
    mcp_enhanced_memory_manager_node,
    get_mcp_enhanced_context_for_llm,
    create_mcp_enhanced_graph
)

__all__ = [
    "MCPContextManager",
    "MCPContextItem", 
    "MCPMedicalAssistantAgent",
    "mcp_enhanced_context_resolver_node",
    "mcp_enhanced_memory_manager_node",
    "get_mcp_enhanced_context_for_llm",
    "create_mcp_enhanced_graph"
]
