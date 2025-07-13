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
    user_input: str
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

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agent_status: str
    active_sessions: int

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

# API Endpoints

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
    x_user_role: Optional[str] = Header(None),
    x_doctor_id: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None)
):
    """
    Main chat endpoint for multi-turn conversations.
    
    Supports context preservation and implicit reference resolution.
    """
    try:
        # Extract user context
        user_role = request.user_role or x_user_role or "assistant"
        doctor_id = request.doctor_id or x_doctor_id
        session_id = request.session_id or x_session_id
        
        # Generate session ID if not provided
        if not session_id:
            session_id = generate_session_id(user_role, doctor_id)
        
        # Validate role-based access
        if user_role == "doctor" and not doctor_id:
            raise HTTPException(
                status_code=400, 
                detail="Doctor ID required for doctor role"
            )
        
        logger.info(f"Chat request - Session: {session_id}, Role: {user_role}, Query: {request.user_input}")
        
        # Process through agent
        result = agent.process_message(
            message=request.user_input,
            session_id=session_id,
            user_role=user_role,
            doctor_id=doctor_id
        )
        
        return ChatResponse(
            success=result["success"],
            result=result["result"],
            session_id=result["session_id"],
            metadata=result["metadata"],
            tool_name=result["tool_name"],
            conversation_context=result.get("conversation_context", {})
        )
        
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
