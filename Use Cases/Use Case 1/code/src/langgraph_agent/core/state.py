"""
Agent State Definition for Multi-Turn Medical Assistant

This module defines the state structure that maintains conversation context,
patient information, and medical history across multiple turns.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class PatientContext(TypedDict):
    """Context information about a patient being discussed"""
    patient_id: Optional[str]
    patient_name: Optional[str]
    appointment_id: Optional[str]
    appointment_date: Optional[str]
    last_mentioned: datetime
    relevant_details: Dict[str, Any]


class DoctorContext(TypedDict):
    """Context information about the current doctor"""
    doctor_id: str
    doctor_name: Optional[str]
    specialization: Optional[str]
    current_appointments: List[Dict[str, Any]]
    last_queried_date: Optional[str]


class ConversationMemory(TypedDict):
    """Memory of recent conversation elements"""
    recent_queries: List[str]
    recent_results: List[Dict[str, Any]]
    conversation_flow: List[str]
    implicit_references: Dict[str, Any]  # Tracks "next patient", "her", "that appointment", etc.


class AgentState(TypedDict):
    """
    Complete state for the multi-turn medical assistant agent.
    
    This state is maintained across conversation turns and includes:
    - Message history with LangGraph's message management
    - Patient context for reference resolution
    - Doctor context for role-based access
    - Conversation memory for implicit references
    - Tool execution results
    """
    
    # Core message handling with LangGraph
    messages: Annotated[List[BaseMessage], add_messages]
    
    # User and role information
    user_role: str  # "doctor" or "assistant"
    doctor_id: Optional[str]
    session_id: str
    identity_context: Optional[Dict[str, Any]]  # Role-based identity information
    
    # Context management
    patient_context: Optional[PatientContext]
    doctor_context: Optional[DoctorContext]
    conversation_memory: ConversationMemory
    
    # Current request processing
    current_query: str
    query_intent: Optional[str]  # "next_patient", "patient_history", "schedule", etc.
    resolved_references: Dict[str, Any]  # What "she", "next patient", etc. refer to
    
    # Tool execution state
    selected_tool: Optional[str]
    tool_parameters: Dict[str, Any]
    tool_results: Any
    sql_query: Optional[str]
    sql_metadata: Dict[str, Any]  # LLM SQL generation metadata
    
    # Response generation
    formatted_response: str
    response_metadata: Dict[str, Any]
    
    # Error handling
    errors: List[str]
    has_errors: bool


def create_initial_state(
    user_role: str,
    doctor_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> AgentState:
    """Create initial agent state for a new conversation."""
    if session_id is None:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return AgentState(
        messages=[],
        user_role=user_role,
        doctor_id=doctor_id,
        session_id=session_id,
        identity_context=None,
        patient_context=None,
        doctor_context=DoctorContext(
            doctor_id=doctor_id or "",
            doctor_name=None,
            specialization=None,
            current_appointments=[],
            last_queried_date=None
        ) if doctor_id else None,
        conversation_memory=ConversationMemory(
            recent_queries=[],
            recent_results=[],
            conversation_flow=[],
            implicit_references={}
        ),
        current_query="",
        query_intent=None,
        resolved_references={},
        selected_tool=None,
        tool_parameters={},
        tool_results=None,
        sql_query=None,
        sql_metadata={},
        formatted_response="",
        response_metadata={},
        errors=[],
        has_errors=False
    )


def update_patient_context(
    state: AgentState,
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    appointment_id: Optional[str] = None,
    appointment_date: Optional[str] = None,
    **additional_details
) -> AgentState:
    """Update patient context in the agent state."""
    current_context = state.get("patient_context")
    
    if current_context is None:
        current_context = PatientContext(
            patient_id=patient_id,
            patient_name=patient_name,
            appointment_id=appointment_id,
            appointment_date=appointment_date,
            last_mentioned=datetime.now(),
            relevant_details=additional_details
        )
    else:
        # Update existing context
        if patient_id:
            current_context["patient_id"] = patient_id
        if patient_name:
            current_context["patient_name"] = patient_name
        if appointment_id:
            current_context["appointment_id"] = appointment_id
        if appointment_date:
            current_context["appointment_date"] = appointment_date
        
        current_context["last_mentioned"] = datetime.now()
        current_context["relevant_details"].update(additional_details)
    
    state["patient_context"] = current_context
    return state


def add_to_conversation_memory(
    state: AgentState,
    query: str,
    result: Any,
    flow_step: str
) -> AgentState:
    """Add information to conversation memory for context tracking."""
    memory = state["conversation_memory"]
    
    # Add to recent queries (keep last 5)
    memory["recent_queries"].append(query)
    if len(memory["recent_queries"]) > 5:
        memory["recent_queries"].pop(0)
    
    # Add to recent results (keep last 5)
    if result:
        memory["recent_results"].append(result)
        if len(memory["recent_results"]) > 5:
            memory["recent_results"].pop(0)
    
    # Add to conversation flow
    memory["conversation_flow"].append(flow_step)
    if len(memory["conversation_flow"]) > 10:
        memory["conversation_flow"].pop(0)
    
    state["conversation_memory"] = memory
    return state
