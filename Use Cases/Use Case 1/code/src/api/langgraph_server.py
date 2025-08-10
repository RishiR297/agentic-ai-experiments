"""
FastAPI Server for LangGraph Multi-Turn Medical Assistant

This server integrates the LangGraph agent with FastAPI to provide
RESTful endpoints for the medical assistant with context and memory.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import sys
import os
from pathlib import Path

# Add the parent directory to the Python path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph_agent.core.graph import MedicalAssistantAgent

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="LangGraph Medical Assistant API",
    description="Multi-turn conversational medical assistant with context and memory",
    version="4.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent
agent = MedicalAssistantAgent()

# Request/Response models
class ChatRequest(BaseModel):
    message: str  # Changed from user_input for consistency
    session_id: Optional[str] = None
    doctor_id: Optional[str] = None
    user_role: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    result: str
    session_id: str
    metadata: Dict[str, Any] = {}
    tool_name: str = "unknown"
    conversation_context: Dict[str, Any] = {}
    sql_metadata: Optional[Dict[str, Any]] = None  # Added for observability
    identity_context: Optional[Dict[str, Any]] = None  # Added for role tracking

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agent_status: str
    active_sessions: int

# --- Slot Validation API Models ---
class SlotValidationRequest(BaseModel):
    """
    Request model for /slot_validation endpoint.
    Example:
    {
        "session_id": "doctor_1_20250714",
        "user_role": "doctor",
        "doctor_id": "1",
        "tool": "appointment_booking",
        "tool_parameters": {"service_name": "Facial", "patient_name": "Eva Davis", "start_time": "2025-07-30 14:00:00"},
        "resolved_references": {}
    }
    """
    session_id: Optional[str] = None
    user_role: Optional[str] = None
    doctor_id: Optional[str] = None
    tool: str
    tool_parameters: Dict[str, Any] = {}
    resolved_references: Dict[str, Any] = {}

class SlotValidationResponse(BaseModel):
    """
    Response model for /slot_validation endpoint.
    Example:
    {
        "success": true,
        "slot_validation": {"status": "missing", "fields": ["start_time"]},
        "clarification_prompt": "What time would you like to book the appointment?",
        "required_lookups": ["StatusId", "PatientId", "BranchName", "ServiceId", "BranchId"],
        "tool_parameters": {"service_name": "Facial", "patient_name": "Eva Davis"},
        "has_errors": false
    }
    """
    success: bool
    slot_validation: Dict[str, Any]
    clarification_prompt: Optional[str] = None
    required_lookups: Optional[list] = None
    tool_parameters: Dict[str, Any] = {}
    has_errors: bool = False

class ClarifyRequest(BaseModel):
    """
    Request model for /clarify endpoint.
    Example:
    {
        "missing_fields": ["start_time"],
        "tool": "appointment_booking",
        "tool_parameters": {"service_name": "Facial", "patient_name": "Eva Davis"},
        "context": {"user_role": "doctor", "doctor_id": "1"}
    }
    """
    missing_fields: list
    tool: str
    tool_parameters: Dict[str, Any] = {}
    context: Dict[str, Any] = {}

class ClarifyResponse(BaseModel):
    """
    Response model for /clarify endpoint.
    Example:
    {
        "clarification_prompt": "What time would you like to book the appointment?"
    }
    """
    clarification_prompt: str
# --- Slot Validation Endpoint ---
@app.post("/slot_validation", response_model=SlotValidationResponse)
async def slot_validation_endpoint(request: SlotValidationRequest):
    """
    Validate required user-facing fields for a tool and generate clarification prompt if needed.
    Returns slot validation status, clarification prompt, required backend lookups, and merged tool parameters.

    Example request:
    {
        "session_id": "doctor_1_20250714",
        "user_role": "doctor",
        "doctor_id": "1",
        "tool": "appointment_booking",
        "tool_parameters": {"service_name": "Facial", "patient_name": "Eva Davis"},
        "resolved_references": {}
    }
    """
    try:
        # Use or create session for context
        session_id = request.session_id or f"slotval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        user_role = request.user_role or "assistant"
        doctor_id = request.doctor_id
        # Build a minimal agent state for slot validation
        state = {
            "session_id": session_id,
            "user_role": user_role,
            "doctor_id": doctor_id,
            "selected_tool": request.tool,
            "tool_parameters": request.tool_parameters,
            "resolved_references": request.resolved_references,
            "errors": [],
            "has_errors": False
        }
        # Call the slot_validator_node directly
        from langgraph_agent.nodes.processing import slot_validator_node
        from langgraph_agent.core.config import AgentConfig
        config = AgentConfig()
        state = slot_validator_node(state, config)
        return SlotValidationResponse(
            success=True,
            slot_validation=state.get("slot_validation", {}),
            clarification_prompt=state.get("clarification_prompt", None),
            required_lookups=state.get("required_lookups", []),
            tool_parameters=state.get("tool_parameters", {}),
            has_errors=state.get("has_errors", False)
        )
    except Exception as e:
        logger.error(f"Slot validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Clarification Prompt Endpoint ---
@app.post("/clarify", response_model=ClarifyResponse)
async def clarify_endpoint(request: ClarifyRequest):
    """
    Generate a personalized clarification prompt for missing fields using the LLM.
    Returns a natural language prompt for the user.

    Example request:
    {
        "missing_fields": ["start_time"],
        "tool": "appointment_booking",
        "tool_parameters": {"service_name": "Facial", "patient_name": "Eva Davis"},
        "context": {"user_role": "doctor", "doctor_id": "1"}
    }
    """
    try:
        from langgraph_agent.core.config import AgentConfig
        config = AgentConfig()
        # Compose the LLM prompt as in slot_validator_node
        system_prompt = (
            "You are a helpful medical assistant agent. "
            "Given the current tool, parameters, and context, generate a polite, context-aware message asking ONLY for the missing user-facing fields. "
            "Use available context (patient name, doctor, service, date, time) to make your question natural and specific. "
            "If only one field is missing, ask for it directly. If multiple are missing, ask for all together. Never ask for fields that are already filled. "
            "Respond in plain English, not JSON."
        )
        llm_input = {
            "tool": request.tool,
            "parameters": request.tool_parameters,
            "missing_fields": request.missing_fields,
            "context": request.context
        }
        from langchain_core.messages import SystemMessage, HumanMessage
        import json
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(llm_input, default=str))
        ]
        response = config.llm.invoke(messages)
        clarification_prompt = response.content.strip()
        return ClarifyResponse(clarification_prompt=clarification_prompt)
    except Exception as e:
        logger.error(f"Clarification prompt error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Role validation
VALID_ROLES = ["doctor", "assistant"]

def validate_role(role: str) -> bool:
    """Validate user role."""
    return role in VALID_ROLES

def extract_identity_context(
    x_user_role: Optional[str] = None,
    x_doctor_id: Optional[str] = None, 
    x_user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Extract identity context from headers."""
    return {
        "role": x_user_role or "assistant",
        "doctor_id": x_doctor_id,
        "user_id": x_user_id,
        "timestamp": datetime.now().isoformat()
    }

# Utility functions
def extract_user_context(request: Request, headers: Dict[str, str]) -> Dict[str, Any]:
    """Extract user context from request headers."""
    return {
        "user_role": headers.get("x-user-role", "assistant"),
        "doctor_id": headers.get("x-doctor-id"),
        "session_id": headers.get("x-session-id")
    }

def generate_session_id(user_role: str, doctor_id: str = None) -> str:
    """Generate a session ID if not provided."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if doctor_id:
        return f"{user_role}_{doctor_id}_{timestamp}"
    return f"{user_role}_{timestamp}"


# --- Additional API Models ---
class ConversationReplayResponse(BaseModel):
    session_id: str
    history: list

class PlannerResponse(BaseModel):
    session_id: str
    next_steps: list
    current_state: Dict[str, Any]

class SessionSummaryRequest(BaseModel):
    session_id: str

class SessionSummaryResponse(BaseModel):
    summary: str
    details: Dict[str, Any] = {}

class ContextResolutionResponse(BaseModel):
    session_id: str
    resolved_references: Dict[str, Any]

class MemoryMutationRequest(BaseModel):
    session_id: str
    updates: Dict[str, Any]

class MemoryMutationResponse(BaseModel):
    session_id: str
    updated_context: Dict[str, Any]

# API Endpoints
@app.get("/conversation/{session_id}/replay", response_model=ConversationReplayResponse)
async def conversation_replay(session_id: str):
    """
    Retrieve the full conversation history for a session.
    Example response:
    {"session_id": "doctor_1_20250714", "history": [{"role": "doctor", "message": "..."}, ...]}
    """
    try:
        # Stub: implement get_conversation_history in agent if not present
        if not hasattr(agent, "get_conversation_history"):
            def get_conversation_history(session_id):
                ctx = agent.get_session_context(session_id)
                return ctx.get("conversation_history", []) if ctx else []
            agent.get_conversation_history = get_conversation_history
        history = agent.get_conversation_history(session_id)
        return ConversationReplayResponse(session_id=session_id, history=history)
    except Exception as e:
        logger.error(f"Conversation replay error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/planner/{session_id}", response_model=PlannerResponse)
async def get_planner_state(session_id: str):
    """
    Get the planned next steps and current state for a session.
    Example response:
    {"session_id": "doctor_1_20250714", "next_steps": ["slot_validation", "backend_lookup"], "current_state": {...}}
    """
    try:
        # Stub: implement get_next_steps in agent if not present
        if not hasattr(agent, "get_next_steps"):
            def get_next_steps(session_id):
                ctx = agent.get_session_context(session_id)
                return ctx.get("planned_steps", []) if ctx else []
            agent.get_next_steps = get_next_steps
        state = agent.get_session_context(session_id)
        next_steps = agent.get_next_steps(session_id)
        return PlannerResponse(session_id=session_id, next_steps=next_steps, current_state=state)
    except Exception as e:
        logger.error(f"Planner state error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/summary", response_model=SessionSummaryResponse)
async def session_summary(request: SessionSummaryRequest):
    """
    Generate a summary of the session's conversation and actions.
    Example request:
    {"session_id": "doctor_1_20250714"}
    Example response:
    {"summary": "You booked 2 appointments for Eva Davis.", "details": {...}}
    """
    try:
        # Stub: implement generate_session_summary in agent if not present
        if not hasattr(agent, "generate_session_summary"):
            def generate_session_summary(session_id):
                return "Session summary not implemented.", {}
            agent.generate_session_summary = generate_session_summary
        summary, details = agent.generate_session_summary(request.session_id)
        return SessionSummaryResponse(summary=summary, details=details)
    except Exception as e:
        logger.error(f"Session summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context/{session_id}/resolved", response_model=ContextResolutionResponse)
async def get_resolved_references(session_id: str):
    """
    Get resolved references/entities for a session.
    Example response:
    {"session_id": "doctor_1_20250714", "resolved_references": {"patient_id": 5, "service_id": 2}}
    """
    try:
        context = agent.get_session_context(session_id)
        resolved = context.get("resolved_references", {}) if context else {}
        return ContextResolutionResponse(session_id=session_id, resolved_references=resolved)
    except Exception as e:
        logger.error(f"Context resolution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/{session_id}/mutate", response_model=MemoryMutationResponse)
async def mutate_memory(request: MemoryMutationRequest):
    """
    Mutate the session memory/context (for testing/admin only).
    Example request:
    {"session_id": "doctor_1_20250714", "updates": {"patient_context": {"name": "Eva Davis"}}}
    Example response:
    {"session_id": "doctor_1_20250714", "updated_context": {...}}
    """
    try:
        # Stub: implement mutate_session_context in agent if not present
        if not hasattr(agent, "mutate_session_context"):
            def mutate_session_context(session_id, updates):
                ctx = agent.get_session_context(session_id)
                if ctx is not None:
                    ctx.update(updates)
                return ctx
            agent.mutate_session_context = mutate_session_context
        updated = agent.mutate_session_context(request.session_id, request.updates)
        return MemoryMutationResponse(session_id=request.session_id, updated_context=updated)
    except Exception as e:
        logger.error(f"Memory mutation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agent_status="operational",
        active_sessions=len(agent.list_active_sessions())
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    x_user_role: Optional[str] = Header(None, alias="X-User-Role"),
    x_doctor_id: Optional[str] = Header(None, alias="X-Doctor-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Main chat endpoint with role-based access control and observability.
    
    Headers:
    - X-User-Role: Role of the user (doctor, nurse, admin)
    - X-Doctor-ID: Doctor identifier (required for doctor role)
    - X-User-ID: User identifier
    """
    try:
        # Extract identity from headers (prioritize headers over request body)
        user_role = x_user_role or request.user_role or "assistant"
        doctor_id = x_doctor_id or request.doctor_id
        
        # Validate role
        if not validate_role(user_role):
            raise HTTPException(
                status_code=403, 
                detail=f"Invalid role: {user_role}. Valid roles: {VALID_ROLES}"
            )
        
        # Role-specific validation
        if user_role == "doctor" and not doctor_id:
            raise HTTPException(
                status_code=400, 
                detail="Doctor ID required for doctor role"
            )
        
        # Extract identity context
        identity_context = extract_identity_context(x_user_role, x_doctor_id, x_user_id)
        
        # Generate session ID if not provided
        session_id = request.session_id
        if not session_id:
            session_id = generate_session_id(user_role, doctor_id)
        
        logger.info(f"🔐 Chat request from {user_role} (session: {session_id})")
        logger.info(f"📝 Message: {request.message}")
        
        # Process with agent
        result = agent.process_message(
            message=request.message,
            session_id=session_id,
            user_role=user_role,
            doctor_id=doctor_id,
            identity_context=identity_context
        )
        
        # Extract SQL metadata for observability
        sql_metadata = result.get("sql_metadata", {})
        if sql_metadata:
            print(f"🗄️  SQL QUERY: {sql_metadata.get('raw_query', 'N/A')}")
            print(f"📊 PARAMETERS: {sql_metadata.get('parameters', [])}")
        
        # Debug logging for response data
        print("=" * 60)
        print("🔍 API RESPONSE DEBUG")
        print("=" * 60)
        print(f"📊 Result keys: {list(result.keys())}")
        print(f"🆔 Session ID: {session_id}")
        print(f"🗄️  SQL Metadata present: {bool(sql_metadata)}")
        print(f"👤 Identity Context present: {bool(identity_context)}")
        print(f"💬 Conversation Context present: {bool(result.get('conversation_context', {}))}")
        print(f"📝 Metadata present: {bool(result.get('metadata', {}))}")
        print("=" * 60)
        
        response = ChatResponse(
            success=True,
            result=result.get("result", "No response generated"),
            session_id=session_id,
            metadata=result.get("metadata", {}),
            tool_name=result.get("tool_name", "unknown"),
            conversation_context=result.get("conversation_context", {}),
            sql_metadata=sql_metadata,
            identity_context=identity_context
        )
        
        # Debug the actual response being sent
        print(f"🚀 Response SQL metadata: {response.sql_metadata}")
        print(f"🚀 Response session_id: {response.session_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/execute", response_model=ChatResponse)
async def execute_tool(
    request: ChatRequest,
    x_user_role: Optional[str] = Header(None),
    x_doctor_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None)
):
    """
    Backward compatibility endpoint that maps to the chat endpoint.
    Maintains compatibility with existing Streamlit frontend.
    """
    return await chat_endpoint(request, x_user_role, x_doctor_id, x_session_id)

@app.get("/context/{session_id}")
async def get_context(session_id: str):
    """
    Get current context state for a given session.
    Returns conversation memory and reference resolution.
    """
    try:
        context = agent.get_session_context(session_id)
        if not context:
            raise HTTPException(status_code=404, detail="Session not found")

        # Import the MCP context manager and get full MCP context items for this session
        from langgraph_agent.tools.mcp_context_manager import mcp_context_manager
        mcp_items = mcp_context_manager.get_session_context(session_id)
        mcp_context = [
            {
                "id": item.id,
                "type": item.type,
                "content": item.content,
                "created_at": item.created_at.isoformat(),
                "expires_at": item.expires_at.isoformat() if item.expires_at else None,
                "session_id": item.session_id,
                "relevance_score": item.relevance_score
            }
            for item in mcp_items
        ]

        # Also provide a summary for convenience
        mcp_summary = mcp_context_manager.get_context_summary(session_id)

        return {
            "session_id": session_id,
            "context": context,
            "mcp_context": mcp_context,
            "mcp_context_summary": mcp_summary,
            "timestamp": datetime.now().isoformat(),
            "memory_summary": {
                "message_count": context.get("message_count", 0),
                "has_patient_context": bool(context.get("patient_context")),
                "has_doctor_context": bool(context.get("doctor_context")),
                "recent_queries": context.get("recent_queries", []),
                "conversation_memory": context.get("conversation_memory", {})
            }
        }
    except Exception as e:
        logger.error(f"Get context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/context")
async def get_session_context(session_id: str):
    """Get current context for a specific session."""
    try:
        context = agent.get_session_context(session_id)
        if not context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "context": context
        }
        
    except Exception as e:
        logger.error(f"Get context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session."""
    try:
        agent.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    try:
        sessions = agent.list_active_sessions()
        session_info = []
        
        for session_id in sessions:
            context = agent.get_session_context(session_id)
            session_info.append({
                "session_id": session_id,
                "message_count": context.get("message_count", 0),
                "has_patient_context": bool(context.get("patient_context")),
                "user_role": session_id.split("_")[0] if "_" in session_id else "unknown"
            })
        
        return {
            "active_sessions": len(sessions),
            "sessions": session_info
        }
        
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def get_available_tools(
    x_user_role: Optional[str] = Header(None),
    x_doctor_id: Optional[str] = Header(None)
):
    """
    Get available tools for the current user role.
    Maintains compatibility with existing frontend.
    """
    try:
        user_role = x_user_role or "assistant"
        
        # Get allowed tools from config
        allowed_tools = agent.config.get_role_permissions(user_role)
        
        # Format tools information
        tools_info = []
        tool_descriptions = {
            "appointment_lookup": "Find specific appointments by patient, doctor, date, or ID",
            "schedule_query": "Get doctor's schedule for specific dates/times",
            "patient_history": "Retrieve patient medical history and past appointments",
            "doctor_availability": "Check when doctors are available",
            "calendar_summary": "Summarize schedule for a day/week"
        }
        
        for tool in allowed_tools:
            tools_info.append({
                "name": tool,
                "description": tool_descriptions.get(tool, "Medical assistant tool")
            })
        
        return {
            "user_role": user_role,
            "tools_available": len(tools_info),
            "tools": tools_info
        }
        
    except Exception as e:
        logger.error(f"Get tools error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/summary")
async def get_mcp_summary():
    """
    Get MCP (Model Context Protocol) system summary.
    Shows structured context metadata and relevance scoring.
    """
    try:
        # Get MCP context from agent if available
        mcp_summary = {
            "mcp_enabled": True,
            "context_preservation": "enhanced",
            "reference_resolution": "advanced",
            "context_scoring": "relevance_based",
            "max_context_items": 5,
            "features": [
                "Multi-turn conversation memory",
                "Context preservation across turns",
                "Advanced reference resolution",
                "Structured context metadata",
                "Relevance-based context scoring"
            ],
            "active_sessions": len(agent.list_active_sessions()),
            "timestamp": datetime.now().isoformat()
        }
        
        return mcp_summary
        
    except Exception as e:
        logger.error(f"MCP summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/demo/test")
async def demo_test():
    """
    Demo endpoint for API testing.
    Returns sample request/response format, including new slot validation, clarify, and advanced endpoints.
    """
    return {
        "api_name": "LangGraph Medical Assistant API",
        "version": "4.0",
        "sample_request": {
            "method": "POST",
            "endpoint": "/chat",
            "headers": {
                "X-User-Role": "doctor",
                "X-Doctor-ID": "1",
                "Content-Type": "application/json"
            },
            "body": {
                "message": "Who's my next patient?",
                "session_id": "doctor_1_20250714"
            }
        },
        "sample_response": {
            "success": True,
            "result": "Your next patient is Eva Davis at 2:00 PM for a Facial.",
            "session_id": "doctor_1_20250714",
            "tool_name": "appointment_lookup",
            "sql_metadata": {
                "raw_query": "SELECT * FROM View_Appointments WHERE...",
                "parameters": [1],
                "result_count": 1,
                "query_type": "next_patient"
            },
            "conversation_context": {
                "query_intent": "next_patient",
                "resolved_references": {}
            }
        },
        "slot_validation_example": {
            "method": "POST",
            "endpoint": "/slot_validation",
            "body": {
                "session_id": "doctor_1_20250714",
                "user_role": "doctor",
                "doctor_id": "1",
                "tool": "appointment_booking",
                "tool_parameters": {"service_name": "Facial", "patient_name": "Eva Davis"},
                "resolved_references": {}
            }
        },
        "clarify_example": {
            "method": "POST",
            "endpoint": "/clarify",
            "body": {
                "missing_fields": ["start_time"],
                "tool": "appointment_booking",
                "tool_parameters": {"service_name": "Facial", "patient_name": "Eva Davis"},
                "context": {"user_role": "doctor", "doctor_id": "1"}
            }
        },
        "conversation_replay_example": {
            "method": "GET",
            "endpoint": "/conversation/doctor_1_20250714/replay"
        },
        "planner_example": {
            "method": "GET",
            "endpoint": "/planner/doctor_1_20250714"
        },
        "session_summary_example": {
            "method": "POST",
            "endpoint": "/session/summary",
            "body": {"session_id": "doctor_1_20250714"}
        },
        "resolved_references_example": {
            "method": "GET",
            "endpoint": "/context/doctor_1_20250714/resolved"
        },
        "memory_mutation_example": {
            "method": "POST",
            "endpoint": "/memory/doctor_1_20250714/mutate",
            "body": {"session_id": "doctor_1_20250714", "updates": {"patient_context": {"name": "Eva Davis"}}}
        },
        "available_endpoints": [
            "POST /chat - Main chat interface",
            "POST /slot_validation - Slot validation and clarification prompt",
            "POST /clarify - Generate clarification prompt for missing fields",
            "GET /context/{session_id} - Session context",
            "GET /context/{session_id}/resolved - Resolved references/entities",
            "GET /conversation/{session_id}/replay - Conversation history",
            "GET /planner/{session_id} - Planner next steps and state",
            "POST /session/summary - Session summary",
            "POST /memory/{session_id}/mutate - Mutate session memory/context",
            "GET /tools - Available tools by role",
            "GET /health - Health check",
            "GET /mcp/summary - MCP system summary",
            "GET /demo/test - This demo endpoint"
        ]
    }

@app.get("/")
async def root():
    """
    Root endpoint with API information and usage guide.
    """
    return {
        "name": "LangGraph Medical Assistant API",
        "version": "4.0",
        "description": "Multi-turn conversational medical assistant with context and memory",
        "features": [
            "LangGraph-based conversation flow",
            "Context preservation across turns", 
            "Advanced reference resolution",
            "Role-based access control",
            "SQL query generation with LLM",
            "MCP integration",
            "Full observability and debugging"
        ],
        "endpoints": {
            "POST /chat": "Main chat interface with role-based access",
            "POST /slot_validation": "Slot validation and clarification prompt",
            "POST /clarify": "Generate clarification prompt for missing fields",
            "GET /context/{session_id}": "Get conversation context and memory",
            "GET /context/{session_id}/resolved": "Resolved references/entities",
            "GET /conversation/{session_id}/replay": "Conversation history",
            "GET /planner/{session_id}": "Planner next steps and state",
            "POST /session/summary": "Session summary",
            "POST /memory/{session_id}/mutate": "Mutate session memory/context",
            "GET /tools": "List available tools for user role",
            "GET /health": "Health check and system status",
            "GET /mcp/summary": "MCP system summary and features",
            "GET /demo/test": "Demo endpoint with sample usage",
            "GET /docs": "Swagger UI documentation",
            "GET /redoc": "ReDoc documentation"
        },
        "usage": {
            "authentication": "Use headers: X-User-Role, X-Doctor-ID",
            "roles": ["doctor", "assistant"],
            "session_management": "Automatic session ID generation if not provided",
            "testing": "Access /docs for interactive API testing"
        },
        "public_access": {
            "swagger_ui": f"http://0.0.0.0:8502/docs",
            "redoc": f"http://0.0.0.0:8502/redoc",
            "ngrok_ready": True,
            "lan_access": "0.0.0.0:8502"
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting LangGraph Medical Assistant API")
    logger.info("Agent initialized and ready for multi-turn conversations")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting LangGraph Medical Assistant server on port 8001...")
    uvicorn.run(app, host="127.0.0.1", port=8001)
