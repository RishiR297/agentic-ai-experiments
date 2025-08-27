"""
MCP-Enhanced Processing Nodes for LangGraph Agent

This module integrates Model Context Protocol (MCP) context management
into the LangGraph processing pipeline for improved context preservation.
"""

import asyncio
from typing import Dict, Any
from ..core.state import AgentState
from ..core.config import AgentConfig
from ..mcp.context_manager import MCPContextManager
from ..memory.conversation_memory import ConversationMemory
from ..nodes.processing import (
    context_resolver_node as original_context_resolver,
    memory_manager_node as original_memory_manager
)
import logging

logger = logging.getLogger(__name__)

# Initialize MCP Context Manager
from langgraph_agent.memory.conversation_memory import ConversationMemory
conversation_memory = ConversationMemory()
mcp_manager = MCPContextManager(conversation_memory)


def mcp_enhanced_context_resolver_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Enhanced context resolver with MCP integration.
    
    This node combines traditional context resolution with MCP context retrieval
    for better reference resolution and context awareness.
    """
    logger.info("MCP-enhanced context resolver processing...")
    
    session_id = state.get("session_id", "default")
    current_query = state["current_query"]
    
    try:
        # Step 1: Run original context resolution
        state = original_context_resolver(state, config)
        
        # Step 2: Enhance with MCP context resolution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Get MCP-based reference resolution
            mcp_references = loop.run_until_complete(
                mcp_manager.resolve_references_with_mcp(session_id, current_query)
            )
            
            # Get relevant MCP context for prompt enhancement
            mcp_context_string = loop.run_until_complete(
                mcp_manager.get_mcp_context_for_prompt(session_id, current_query)
            )
            
            # Merge MCP references with existing resolved references
            existing_references = state.get("resolved_references", {})
            enhanced_references = {**existing_references, **mcp_references}
            state["resolved_references"] = enhanced_references
            
            # Add MCP context to state for use in other nodes
            state["mcp_context"] = mcp_context_string
            state["mcp_references"] = mcp_references
            
            logger.info(f"MCP context resolved - Enhanced references: {enhanced_references}")
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"MCP context resolution error: {e}")
        # Fall back to original context resolution
        state["mcp_context"] = ""
        state["mcp_references"] = {}
    
    return state


def mcp_enhanced_memory_manager_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Enhanced memory manager with MCP context persistence.
    
    This node updates both traditional conversation memory and MCP context items
    for comprehensive context preservation.
    """
    logger.info("MCP-enhanced memory manager processing...")
    
    session_id = state.get("session_id", "default")
    
    try:
        # Step 1: Run original memory management
        state = original_memory_manager(state, config)
        
        # Step 2: Update MCP context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Update MCP context from this conversation turn
            loop.run_until_complete(
                mcp_manager.update_context_from_conversation_turn(
                    session_id=session_id,
                    user_input=state["current_query"],
                    agent_response=state["formatted_response"],
                    intent=state.get("query_intent", "unknown"),
                    tool_used=state.get("selected_tool", "unknown"),
                    context_resolved=state.get("resolved_references", {})
                )
            )
            
            # Get context summary for metadata
            context_summary = mcp_manager.get_context_summary(session_id)
            state["mcp_context_summary"] = context_summary
            
            logger.info(f"MCP context updated - Summary: {context_summary}")
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"MCP memory management error: {e}")
        state["mcp_context_summary"] = {"error": str(e)}
    
    return state


def get_mcp_enhanced_context_for_llm(state: AgentState) -> str:
    """
    Generate enhanced context string for LLM prompts that includes MCP context.
    
    This function can be used in any node that needs to provide context to the LLM.
    """
    context_parts = []
    
    # Add traditional context
    if state.get("conversation_memory", {}).get("conversation_flow"):
        context_parts.append("=== RECENT CONVERSATION ===")
        recent_flow = state["conversation_memory"]["conversation_flow"][-3:]  # Last 3 turns
        for i, turn in enumerate(recent_flow, 1):
            context_parts.append(f"Turn {i}: {turn}")
    
    # Add MCP context if available
    if state.get("mcp_context"):
        context_parts.append(state["mcp_context"])
    
    # Add implicit references
    if state.get("resolved_references"):
        context_parts.append("=== RESOLVED REFERENCES ===")
        for ref, value in state["resolved_references"].items():
            context_parts.append(f"{ref} -> {value}")
    
    return "\n".join(context_parts)


def create_mcp_enhanced_graph():
    """
    Create a LangGraph workflow with MCP-enhanced nodes.
    
    This replaces the original context_resolver_node and memory_manager_node
    with MCP-enhanced versions.
    """
    from langgraph.graph import StateGraph
    from ..nodes.processing import (
        tool_selector_node,
        sql_generator_node,
        response_formatter_node,
    )
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add MCP-enhanced nodes
    workflow.add_node("context_resolver", mcp_enhanced_context_resolver_node)
    workflow.add_node("tool_selector", tool_selector_node)
    workflow.add_node("sql_generator", sql_generator_node)
    workflow.add_node("response_formatter", response_formatter_node)
    workflow.add_node("memory_manager", mcp_enhanced_memory_manager_node)
    
    # Define the flow
    workflow.set_entry_point("context_resolver")
    workflow.add_edge("context_resolver", "tool_selector")
    workflow.add_edge("tool_selector", "sql_generator")
    workflow.add_edge("sql_generator", "response_formatter")
    workflow.add_edge("response_formatter", "memory_manager")
    workflow.set_finish_point("memory_manager")
    
    return workflow.compile()


# Utility functions for MCP context management

async def get_session_mcp_summary(session_id: str) -> Dict[str, Any]:
    """Get comprehensive MCP context summary for a session."""
    return mcp_manager.get_context_summary(session_id)


async def cleanup_session_mcp_context(session_id: str, max_age_hours: int = 24):
    """Clean up old MCP contexts for a session."""
    await mcp_manager.cleanup_old_contexts(session_id, max_age_hours)


def get_mcp_context_manager() -> MCPContextManager:
    """Get the global MCP context manager instance."""
    return mcp_manager
