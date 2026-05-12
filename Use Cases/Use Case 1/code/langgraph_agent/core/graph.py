import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from .state import AgentState, create_initial_state
from .config import AgentConfig
from ..nodes.processing import (
    context_resolver_node,
    rbac_evaluation_node,
    tool_selector_node,
    sql_generator_node,
    tool_executor_node,
    response_formatter_node,
    memory_manager_node,
    slot_validator_node,
    backend_lookup_node
)

logger = logging.getLogger(__name__)

def create_medical_agent_graph():
    """Create the medical assistant graph using TypedDict AgentState."""
    config = AgentConfig()
    workflow = StateGraph(AgentState)
    
    # Add nodes directly - TypedDict handles field validation
    workflow.add_node("context_resolver", lambda state: context_resolver_node(state, config))
    workflow.add_node("tool_selector", lambda state: tool_selector_node(state, config))
    workflow.add_node("rbac_evaluation", lambda state: rbac_evaluation_node(state, config))
    workflow.add_node("slot_validator", lambda state: slot_validator_node(state, config))
    workflow.add_node("backend_lookup", lambda state: backend_lookup_node(state, config))
    workflow.add_node("sql_generator", lambda state: sql_generator_node(state, config))
    workflow.add_node("tool_executor", lambda state: tool_executor_node(state, config))
    workflow.add_node("response_formatter", lambda state: response_formatter_node(state, config))
    workflow.add_node("memory_manager", lambda state: memory_manager_node(state, config))

    # Set entry point
    workflow.set_entry_point("context_resolver")

    # Add edges
    workflow.add_edge("context_resolver", "tool_selector")
    workflow.add_edge("tool_selector", "slot_validator")
    
    # Conditional routing from slot_validator
    def route_after_validation(state):
        status = state.get("slot_validation", {}).get("status")
        tool = state.get("selected_tool")
        logger.info(f"Routing after validation: status={status}, tool={tool}")
        
        if status == "ok":
            return "sql_generator"
        elif status == "missing_backend":
            return "backend_lookup"
        elif status == "backend_complete":
            # For tools like schedule_query that don't need SQL generation
            return "tool_executor"
        else:
            return "response_formatter"
    
    workflow.add_conditional_edges(
        "slot_validator",
        route_after_validation,
        {
            "sql_generator": "sql_generator",
            "backend_lookup": "backend_lookup",
            "tool_executor": "tool_executor",
            "response_formatter": "response_formatter"
        }
    )
    
    # Conditional routing from backend_lookup
    def route_after_backend_lookup(state):
        status = state.get("slot_validation", {}).get("status")
        tool = state.get("selected_tool")
        rbac_error = state.get("rbac_error", False)
        logger.info(f"Routing after backend lookup: status={status}, tool={tool}")
        
        # Check for RBAC denial first
        if rbac_error or status == "rbac_denied":
            return "response_formatter"  # Skip RBAC and tool execution, go directly to response formatting
        
        # Hard stop on validation failure - do not continue to SQL generation or execution
        if status == "validation_failed":
            return "response_formatter"
        
        if status == "backend_complete":
            # Tools that don't need SQL generation (use direct function calls)
            if tool in [
                "schedule_query",
                "appointment_lookup",
                "appointment_rescheduling",
                "appointment_cancellation",
                "cancel_appointment",
                "appointment_query_executor",
                "calendar_summary",
                "doctor_availability",
            ]:
                # Check if RBAC evaluation is needed
                if tool in ["appointment_rescheduling", "appointment_cancellation", "cancel_appointment"]:
                    return "rbac_evaluation"
                else:
                    return "tool_executor"
            else:
                # Tools that need SQL generation (appointment_lookup, etc.)
                return "sql_generator"
        elif status == "error":
            # For appointment management tools, even with backend errors, try to proceed with RBAC and tool execution
            if tool in ["appointment_rescheduling", "appointment_cancellation", "cancel_appointment"]:
                return "rbac_evaluation"
            else:
                # Other tools with errors go to SQL generation as fallback
                return "sql_generator"
        else:
            # Default: proceed to SQL generation
            return "sql_generator"
    
    workflow.add_conditional_edges(
        "backend_lookup",
        route_after_backend_lookup,
        {
            "sql_generator": "sql_generator",
            "rbac_evaluation": "rbac_evaluation", 
            "tool_executor": "tool_executor",
            "response_formatter": "response_formatter"
        }
    )
    
    # Conditional routing from sql_generator - Check if RBAC is needed
    def route_after_sql_generation(state):
        tool = state.get("selected_tool")
        logger.info(f"Routing after SQL generation: tool={tool}")
        
        # RBAC evaluation required for appointment management operations
        if tool in ["appointment_rescheduling", "appointment_cancellation", "cancel_appointment"]:
            return "rbac_evaluation"
        else:
            return "tool_executor"
    
    workflow.add_conditional_edges(
        "sql_generator",
        route_after_sql_generation,
        {
            "rbac_evaluation": "rbac_evaluation",
            "tool_executor": "tool_executor"
        }
    )
    
    # Conditional routing from RBAC evaluation
    def route_after_rbac(state):
        rbac_status = state.get("rbac_evaluation")
        logger.info(f"Routing after RBAC: status={rbac_status}")
        
        if rbac_status == "approved":
            return "tool_executor"
        else:
            # RBAC denied or error - skip tool execution, go directly to response formatting
            return "response_formatter"
    
    workflow.add_conditional_edges(
        "rbac_evaluation", 
        route_after_rbac,
        {
            "tool_executor": "tool_executor",
            "response_formatter": "response_formatter"
        }
    )
    
    workflow.add_edge("tool_executor", "response_formatter")
    workflow.add_edge("response_formatter", "memory_manager")
    workflow.add_edge("memory_manager", END)

    logger.info("Compiling graph...")
    return workflow.compile()


class MedicalAssistantAgent:
    def __init__(self):
        self.graph = create_medical_agent_graph()
        self.config = AgentConfig()
        self.sessions: Dict[str, AgentState] = {}

    def get_or_create_session(self, session_id: str, user_role: str, doctor_id: Optional[str] = None) -> AgentState:
        doctor_id = doctor_id or ""
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
        doctor_id: Optional[str] = None,
        identity_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            state = self.get_or_create_session(session_id, user_role, doctor_id)
            if identity_context:
                state["identity_context"] = identity_context

            # Reset relevant state
            state["current_query"] = message
            state["messages"].append(HumanMessage(content=message))
            state["errors"] = []
            state["has_errors"] = False
            state["tool_results"] = None
            state["formatted_response"] = ""
            state["sql_metadata"] = {}

            logger.info(f"Processing message for session {session_id}: {message}")
            result = self.graph.invoke(state)
            self.sessions[session_id] = result

            # Ensure clarification_prompt is copied to formatted_response if needed
            if "clarification_prompt" in result and not result.get("formatted_response"):
                result["formatted_response"] = result["clarification_prompt"]

            return {
                "success": not result.get("has_errors", False),
                "result": result.get("formatted_response", ""),
                "metadata": result.get("response_metadata", {}),
                "tool_name": result.get("selected_tool", "unknown"),
                "sql_metadata": result.get("sql_metadata", {}),
                "session_id": session_id,
                "conversation_context": {
                    "patient_context": result.get("patient_context", {}),
                    "query_intent": result.get("query_intent", ""),
                    "resolved_references": result.get("resolved_references", {})
                }
            }
        except Exception as e:
            logger.exception(f"Agent processing error: {e}")
            return {
                "success": False,
                "result": f"I encountered an error processing your request: {str(e)}",
                "metadata": {"error": str(e)},
                "tool_name": "error_handler",
                "session_id": session_id
            }
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get current context for a session."""
        if session_id in self.sessions:
            state = self.sessions[session_id]
            return {
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
        return {}
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def list_active_sessions(self) -> List[str]:
        return list(self.sessions.keys())
