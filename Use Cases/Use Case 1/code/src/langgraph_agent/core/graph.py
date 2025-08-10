import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from .state import AgentState, create_initial_state
from .config import AgentConfig

# Import deterministic validation nodes/tools (pure logic, no LLM)
from ..nodes.processing import (
    context_resolver_node,
    tool_selector_node,
    sql_generator_node,
    tool_executor_node,
    response_formatter_node,
    memory_manager_node,
    slot_validator_node,
    backend_lookup_node,
    appointment_overlap_check_node,
    doctor_schedule_check_node
)

logger = logging.getLogger(__name__)


def should_continue_processing(state: AgentState) -> str:
    """Determine if processing should continue or end."""
    if state["has_errors"] and len(state["errors"]) > 3:
        return "end"
    if not state["current_query"].strip():
        return "end"
    return "continue"



def create_medical_agent_graph():
    config = AgentConfig()
    workflow = StateGraph(AgentState)

    # Node wrappers for sync compatibility
    def context_resolver_sync(state):
        return context_resolver_node(state, config)
    def tool_selector_sync(state):
        return tool_selector_node(state, config)
    def slot_validator_sync(state):
        return slot_validator_node(state, config)
    def backend_lookup_sync(state):
        return backend_lookup_node(state, config)
    def appointment_overlap_check_sync(state):
        return appointment_overlap_check_node(state, config)
    def doctor_schedule_check_sync(state):
        return doctor_schedule_check_node(state, config)
    def sql_generator_sync(state):
        return sql_generator_node(state, config)
    def tool_executor_sync(state):
        return tool_executor_node(state, config)
    def response_formatter_sync(state):
        return response_formatter_node(state, config)
    def memory_manager_sync(state):
        return memory_manager_node(state, config)

    # Register nodes (deterministic validation nodes are independent)
    workflow.add_node("context_resolver", context_resolver_sync)
    workflow.add_node("tool_selector", tool_selector_sync)
    workflow.add_node("slot_validator", slot_validator_sync)
    workflow.add_node("backend_lookup", backend_lookup_sync)
    workflow.add_node("appointment_overlap_check", appointment_overlap_check_sync)
    workflow.add_node("doctor_schedule_check", doctor_schedule_check_sync)
    workflow.add_node("sql_generator", sql_generator_sync)
    workflow.add_node("tool_executor", tool_executor_sync)
    workflow.add_node("response_formatter", response_formatter_sync)
    workflow.add_node("memory_manager", memory_manager_sync)

    # Example: Linear deterministic validation flow (can be orchestrated by LLM supervisor node)
    workflow.add_edge("context_resolver", "tool_selector")
    workflow.add_edge("tool_selector", "slot_validator")
    workflow.add_edge("slot_validator", "backend_lookup")
    workflow.add_edge("backend_lookup", "appointment_overlap_check")
    workflow.add_edge("appointment_overlap_check", "doctor_schedule_check")
    workflow.add_edge("doctor_schedule_check", "sql_generator")
    workflow.add_edge("sql_generator", "tool_executor")
    workflow.add_edge("tool_executor", "response_formatter")
    workflow.add_edge("response_formatter", "memory_manager")
    workflow.add_edge("memory_manager", END)

    # Set entry point after all edges
    workflow.set_entry_point("context_resolver")

    # Optional: Debug prints
    logger.debug("[DEBUG] Registered nodes: %s", list(workflow.nodes.keys()))
    logger.debug("[DEBUG] Registered edges: %s", list(workflow.edges))

    compiled_graph = workflow.compile()
    return compiled_graph


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
                logger.info(f"Added identity context: {identity_context}")

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

            logger.info(f"Graph result - formatted_response: '{result['formatted_response']}'")

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
                "error": str(e)
            }

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def list_active_sessions(self) -> List[str]:
        return list(self.sessions.keys())
