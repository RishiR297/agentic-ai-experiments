"""
LangGraph Graph Definition for Multi-Turn Medical Assistant

This module creates the main conversation flow graph that handles
multi-turn conversations with context and memory management.
"""

import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from .base import AgentState, AgentConfig
from .state import create_initial_state
from ..nodes.processing import (
    context_resolver_node,
    tool_selector_node,
    sql_generator_node,
    response_formatter_node,
    memory_manager_node,
    slot_validator_node
)
from ..nodes.multi_step_planner import multi_step_planner_node

logger = logging.getLogger(__name__)


def should_continue_processing(state: AgentState) -> str:
    """Determine if processing should continue or end."""
    if state["has_errors"] and len(state["errors"]) > 3:
        return "end"
    if not state["current_query"].strip():
        return "end"
    return "continue"


def create_medical_agent_graph() -> StateGraph:
    """
    Create the main LangGraph for the medical assistant.
    
    The graph follows this flow:
    1. Context Resolution - Resolve implicit references and context
    2. Tool Selection - Choose appropriate tool and parameters
    3. SQL Generation - Generate and execute database queries
    4. Response Formatting - Format results into natural language
    5. Memory Management - Update conversation memory and context
    """
    
    # Create the configuration
    config = AgentConfig()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes with config binding

    workflow.add_node("context_resolver", lambda state: context_resolver_node(state, config))
    workflow.add_node("tool_selector", lambda state: tool_selector_node(state, config))
    workflow.add_node("multi_step_planner", lambda state: multi_step_planner_node(state, config))
    workflow.add_node("slot_validator", lambda state: slot_validator_node(state, config))
    # Add backend_lookup_node if missing
    from ..nodes.processing import backend_lookup_node
    workflow.add_node("backend_lookup_node", lambda state: backend_lookup_node(state, config))
    workflow.add_node("sql_generator", lambda state: sql_generator_node(state, config))
    workflow.add_node("response_formatter", lambda state: response_formatter_node(state, config))
    workflow.add_node("memory_manager", lambda state: memory_manager_node(state, config))

    # Set entry point
    workflow.set_entry_point("context_resolver")

    # Define the flow
    workflow.add_edge("context_resolver", "tool_selector")
    workflow.add_edge("tool_selector", "multi_step_planner")
    workflow.add_edge("multi_step_planner", "slot_validator")
    # Replace slot_routing with direct conditional routing from slot_validator
    def route_from_slot_validator(state):
        # If all required fields are present, go to sql_generator
        if state.get("slot_validation", {}).get("status") == "ok":
            return "sql_generator"
        # If missing natural fields, go to multi_step_planner
        if state.get("slot_validation", {}).get("status") == "missing":
            return "multi_step_planner"
        # If missing backend fields, go to backend_lookup_node
        if state.get("slot_validation", {}).get("status") == "missing_backend":
            return "backend_lookup_node"
        # Fallback: go to multi_step_planner
        return "multi_step_planner"

    workflow.add_conditional_edges("slot_validator", {
        "sql_generator": lambda state: route_from_slot_validator(state) == "sql_generator",
        "multi_step_planner": lambda state: route_from_slot_validator(state) == "multi_step_planner",
        "backend_lookup_node": lambda state: route_from_slot_validator(state) == "backend_lookup_node"
    })
    workflow.add_edge("sql_generator", "response_formatter")
    workflow.add_edge("response_formatter", "memory_manager")
    workflow.add_edge("memory_manager", END)

    # Compile the graph
    return workflow.compile()


class MedicalAssistantAgent:
    """
    Main agent class that manages conversation state and processing.
    """
    
    def __init__(self):
        self.graph = create_medical_agent_graph()
        self.config = AgentConfig()
        self.sessions: Dict[str, AgentState] = {}
    
    def get_or_create_session(
        self, 
        session_id: str, 
        user_role: str, 
        doctor_id: str = None
    ) -> AgentState:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = create_initial_state(
                user_role=user_role,
                doctor_id=doctor_id,
                session_id=session_id
            )
        return self.sessions[session_id]
    
    def process_message(
        self, 
        message: str, 
        session_id: str, 
        user_role: str, 
        doctor_id: str = None,
        identity_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the agent graph.
        
        Returns:
            Dict containing response, metadata, and conversation state
        """
        try:
            # Get or create session state
            state = self.get_or_create_session(session_id, user_role, doctor_id)
            
            # Add identity context to state if provided
            if identity_context:
                state["identity_context"] = identity_context
                logger.info(f"Added identity context: {identity_context}")
            
            # Add user message to state
            state["current_query"] = message
            state["messages"].append(HumanMessage(content=message))
            
            # Reset processing state
            state["errors"] = []
            state["has_errors"] = False
            state["tool_results"] = None
            state["formatted_response"] = ""
            state["sql_metadata"] = {}
            
            logger.info(f"Processing message for session {session_id}: {message}")
            
            # Run the graph
            result = self.graph.invoke(state)
            
            # Update session state
            self.sessions[session_id] = result

            # Ensure clarification_prompt is copied into formatted_response if present and formatted_response is empty
            if "clarification_prompt" in result and not result.get("formatted_response"):
                result["formatted_response"] = result["clarification_prompt"]

            # Debug logging
            logger.info(f"Graph result - formatted_response: '{result['formatted_response']}'")
            logger.info(f"Graph result - has_errors: {result['has_errors']}")
            logger.info(f"Graph result keys: {list(result.keys())}")
            logger.info(f"Graph result sql_metadata: {result.get('sql_metadata', 'NOT_FOUND')}")

            # Return processed result
            return {
                "success": not result["has_errors"],
                "result": result["formatted_response"],
                "metadata": result["response_metadata"],
                "tool_name": result.get("selected_tool", "unknown"),
                "sql_metadata": result.get("sql_metadata", {}),
                "session_id": session_id,
                "conversation_context": {
                    "patient_context": result.get("patient_context"),
                    "query_intent": result.get("query_intent"),
                    "resolved_references": result.get("resolved_references")
                }
            }
            
        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return {
                "success": False,
                "result": f"I encountered an error processing your request: {str(e)}",
                "metadata": {"error": str(e)},
                "tool_name": "error_handler",
                "session_id": session_id
            }
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get current context for a session, including all MCP context fields for diagnostics."""
        if session_id in self.sessions:
            state = self.sessions[session_id]
            # Gather all relevant context fields for diagnostics
            context = {
                "patient_context": state.get("patient_context"),
                "doctor_context": state.get("doctor_context"),
                "conversation_memory": state.get("conversation_memory", {}),
                "message_count": len(state.get("messages", [])),
                "recent_queries": state.get("recent_queries", []),
                "resolved_references": state.get("resolved_references", {}),
                "query_intent": state.get("query_intent"),
                "identity_context": state.get("identity_context"),
                "selected_tool": state.get("selected_tool"),
                "sql_metadata": state.get("sql_metadata", {}),
                "tool_results": state.get("tool_results"),
                "errors": state.get("errors", []),
                "has_errors": state.get("has_errors", False),
                "response_metadata": state.get("response_metadata", {}),
            }
            # Optionally include any other custom MCP context fields here
            return context
        return {}
    
    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def list_active_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
