# tools/__init__.py
"""
LangGraph Medical Assistant - Tools Module

Essential tools for database operations and context management.
Contains core utilities for patient data, appointments, and MCP integration.

Core Tools:
- database.py: Patient ID resolution, database queries, appointment operations
- mcp_context_manager.py: Model Context Protocol integration and context management
"""

from .database import (
    get_or_create_patient_id,
    execute_query
)

from .mcp_context_manager import (
    MCPContextManager
)

__all__ = [
    "get_or_create_patient_id",
    "execute_query",
    "MCPContextManager"
]
