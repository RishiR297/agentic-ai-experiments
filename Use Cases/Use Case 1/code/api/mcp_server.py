"""
MCP-Enhanced LangGraph Server Demo

This server demonstrates proper MCP (Model Context Protocol) integration
for conversational context preservation in multi-turn medical conversations.

Run this server to test MCP-enhanced context preservation.
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our MCP-enhanced agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_agent.core.config import AgentConfig
from langgraph_agent.mcp.mcp_agent import MCPMedicalAssistantAgent

# Global agent variable
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global agent
    
    # Startup
    logger.info("Starting MCP-Enhanced Medical Assistant API")
    
    try:
        config = AgentConfig()
        agent = MCPMedicalAssistantAgent(config)
        logger.info("MCP-Enhanced agent initialized successfully")
        logger.info("✅ Model Context Protocol (MCP) is ENABLED")
        logger.info("✅ Enhanced context preservation across turns")
        logger.info("✅ Advanced reference resolution")
        logger.info("✅ Structured context metadata")
    except Exception as e:
        logger.error(f"Failed to initialize MCP agent: {e}")
        raise
    
    yield
    
    # Shutdown (if needed)
    pass

# FastAPI app with lifespan
app = FastAPI(
    title="MCP-Enhanced Medical Assistant",
    description="Medical Assistant with Model Context Protocol integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: Optional[MCPMedicalAssistantAgent] = None

# Pydantic models
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_input: str
    user_role: str = "doctor"
    doctor_id: Optional[str] = "11712738-BFDE-436E-950B-2731FA20DDB2"

class ChatResponse(BaseModel):
    success: bool
    result: str
    session_id: str
    metadata: dict
    tool_name: str
    conversation_context: dict

class ContextResponse(BaseModel):
    session_id: str
    traditional_context: dict
    mcp_context_summary: dict
    enhanced_features: dict

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    agent_status: str
    active_sessions: int
    mcp_enabled: bool = True


def generate_session_id(user_role: str, doctor_id: str = None) -> str:
    """Generate a session ID if not provided."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if doctor_id:
        return f"mcp_{user_role}_{doctor_id[-8:]}_{timestamp}"
    return f"mcp_{user_role}_{timestamp}"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with MCP status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        agent_status="operational",
        active_sessions=len(agent.list_active_sessions()) if agent else 0,
        mcp_enabled=True
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    MCP-Enhanced chat endpoint for multi-turn conversations.
    
    Features:
    - Model Context Protocol (MCP) integration
    - Enhanced context preservation across turns
    - Advanced reference resolution (e.g., "her", "next patient", "that appointment")
    - Structured context metadata and scoring
    - Cross-session context sharing when appropriate
    """
    try:
        if not agent:
            raise HTTPException(status_code=500, detail="MCP agent not initialized")
        
        # Generate session ID if not provided
        session_id = request.session_id
        if not session_id:
            session_id = generate_session_id(request.user_role, request.doctor_id)
        
        logger.info(f"MCP Chat request - Session: {session_id}, Role: {request.user_role}, Query: {request.user_input}")
        
        # Process with MCP-enhanced agent
        result = agent.process_message_sync(
            session_id=session_id,
            message=request.user_input,
            user_role=request.user_role,
            doctor_id=request.doctor_id
        )
        
        logger.info(f"MCP Chat response - Success: {result['success']}, MCP contexts: {result['metadata'].get('context_items_used', 0)}")
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"MCP chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/context/{session_id}", response_model=ContextResponse)
async def get_session_context(session_id: str):
    """Get comprehensive session context including MCP data."""
    try:
        if not agent:
            raise HTTPException(status_code=500, detail="MCP agent not initialized")
        
        # Get enhanced context
        context = await agent.get_session_context(session_id)
        
        # Get traditional context for comparison
        traditional_context = {
            "patient_context": context.get("patient_context"),
            "doctor_context": context.get("doctor_context"),
            "conversation_memory": context.get("conversation_memory"),
            "message_count": context.get("message_count", 0)
        }
        
        # MCP-specific context
        mcp_context_summary = context.get("mcp_context_summary", {})
        
        # Enhanced features information
        enhanced_features = {
            "reference_resolution": "Advanced pronoun and implicit reference resolution",
            "context_scoring": "Relevance-based context item scoring",
            "structured_metadata": "Rich context metadata for better retrieval",
            "cross_turn_persistence": "Context preserved across conversation turns",
            "temporal_decay": "Context relevance decreases over time naturally"
        }
        
        return ContextResponse(
            session_id=session_id,
            traditional_context=traditional_context,
            mcp_context_summary=mcp_context_summary,
            enhanced_features=enhanced_features
        )
        
    except Exception as e:
        logger.error(f"Context retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/summary")
async def get_mcp_summary():
    """Get comprehensive MCP context summary across all sessions."""
    try:
        if not agent:
            raise HTTPException(status_code=500, detail="MCP agent not initialized")
        
        summary = await agent.get_mcp_context_summary_for_all_sessions()
        
        return {
            "mcp_enabled": True,
            "summary": summary,
            "features": {
                "context_preservation": "✅ Enabled",
                "reference_resolution": "✅ Enhanced",
                "structured_metadata": "✅ Active",
                "relevance_scoring": "✅ Functional",
                "temporal_decay": "✅ Configured"
            }
        }
        
    except Exception as e:
        logger.error(f"MCP summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session and its MCP context."""
    try:
        if not agent:
            raise HTTPException(status_code=500, detail="MCP agent not initialized")
        
        agent.clear_session(session_id)
        
        return {
            "success": True,
            "message": f"Session {session_id} and its MCP context cleared",
            "mcp_enabled": True
        }
        
    except Exception as e:
        logger.error(f"Session clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo/conversation")
async def demo_conversation():
    """
    Demonstrate MCP-enhanced multi-turn conversation capabilities.
    
    This endpoint shows how context is preserved and enhanced across turns.
    """
    return {
        "demo_scenario": "MCP-Enhanced Multi-Turn Medical Conversation",
        "steps": [
            {
                "step": 1,
                "query": "What are my appointments today?",
                "mcp_features": ["Initial context creation", "Doctor context establishment"]
            },
            {
                "step": 2,
                "query": "Who is my next patient?",
                "mcp_features": ["Reference to 'my' resolved using doctor context", "Appointment context retrieval"]
            },
            {
                "step": 3,
                "query": "Tell me about her medical history",
                "mcp_features": ["Pronoun 'her' resolved to previous patient", "Patient context linkage"]
            },
            {
                "step": 4,
                "query": "What was that appointment about?",
                "mcp_features": ["'that appointment' resolved using conversation context", "Temporal context awareness"]
            }
        ],
        "traditional_vs_mcp": {
            "traditional": "Context stored in session memory, limited reference resolution",
            "mcp_enhanced": "Structured context items, advanced reference resolution, relevance scoring, metadata enrichment"
        },
        "test_url": "/chat",
        "test_session": "mcp_demo_session_001"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting MCP-Enhanced Medical Assistant Server")
    logger.info("📋 Features:")
    logger.info("   ✅ Model Context Protocol (MCP) integration")
    logger.info("   ✅ Enhanced context preservation")
    logger.info("   ✅ Advanced reference resolution")
    logger.info("   ✅ Structured context metadata")
    logger.info("   ✅ Relevance-based context scoring")
    logger.info("")
    logger.info("🔗 Endpoints:")
    logger.info("   POST /chat - MCP-enhanced chat")
    logger.info("   GET /context/{session_id} - View session context")
    logger.info("   GET /mcp/summary - MCP system summary")
    logger.info("   GET /demo/conversation - Demo scenario")
    logger.info("")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
