"""
MCP-Enhanced LangGraph Medical Assistant

This module provides an MCP-enabled version of the medical assistant
with enhanced context preservation using Model Context Protocol.
"""

import logging
from typing import Dict, Any, Optional, List
from langgraph_agent.core.state import AgentState
from langgraph_agent.core.config import AgentConfig
from langgraph_agent.mcp.mcp_nodes import create_mcp_enhanced_graph, get_mcp_context_manager
from langgraph_agent.memory.conversation_memory import ConversationMemory

logger = logging.getLogger(__name__)


class MCPMedicalAssistantAgent:
    """
    MCP-Enhanced Medical Assistant Agent.
    
    This version uses Model Context Protocol for superior context preservation
    and reference resolution across conversation turns.
    
    Key enhancements:
    - MCP context items for structured context storage
    - Enhanced reference resolution using MCP context
    - Cross-turn context persistence with relevance scoring
    - Structured context metadata for better retrieval
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.graph = create_mcp_enhanced_graph()
        self.sessions: Dict[str, AgentState] = {}
        self.mcp_manager = get_mcp_context_manager()
        self.conversation_memory = ConversationMemory()
        
        logger.info("MCP-Enhanced Medical Assistant Agent initialized")
    
    def create_initial_state(
        self, 
        user_role: str, 
        doctor_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AgentState:
        """Create initial state for a new session with MCP integration."""
        
        return {
            "messages": [],
            "current_query": "",
            "user_role": user_role,
            "doctor_id": doctor_id,
            "session_id": session_id,
            "patient_context": None,
            "doctor_context": {
                "doctor_id": doctor_id,
                "doctor_name": None,
                "specialization": None,
                "current_appointments": [],
                "last_queried_date": None
            },
            "conversation_memory": {
                "recent_queries": [],
                "recent_results": [],
                "conversation_flow": [],
                "implicit_references": {}
            },
            "query_intent": "",
            "resolved_references": {},
            "selected_tool": "",
            "tool_parameters": {},
            "sql_query": "",
            "tool_results": [],
            "formatted_response": "",
            "response_metadata": {},
            "errors": [],
            "has_errors": False,
            # MCP-specific fields
            "mcp_context": "",
            "mcp_references": {},
            "mcp_context_summary": {}
        }
    
    async def process_message(
        self, 
        session_id: str, 
        message: str, 
        user_role: str = "assistant", 
        doctor_id: Optional[str] = None,
        identity_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a message with MCP-enhanced context preservation.
        
        This async method provides full MCP integration with:
        - Context retrieval before processing
        - Enhanced reference resolution
        - Context persistence after processing
        """
        try:
            # Get or create session state
            if session_id in self.sessions:
                state = self.sessions[session_id]
                # Load recent MCP context
                await self._load_mcp_context_for_session(session_id, state)
            else:
                state = self.create_initial_state(user_role, doctor_id, session_id)
            
            # Update current message
            state["current_query"] = message
            state["session_id"] = session_id
            
            # Reset processing state
            state["errors"] = []
            state["has_errors"] = False
            state["tool_results"] = None
            state["formatted_response"] = ""
            
            logger.info(f"MCP processing message for session {session_id}: {message}")
            
            # Run the MCP-enhanced graph
            result = self.graph.invoke(state, {"configurable": {"agent_config": self.config}})
            
            # Debug: Log what the graph returns
            logger.info(f"Graph result keys: {list(result.keys())}")
            logger.info(f"Graph result fields: {[k for k, v in result.items() if v is not None]}")
            
            # Update session state
            self.sessions[session_id] = result
            
            # Ensure all required fields are present in the result
            if "patient_context" not in result:
                result["patient_context"] = {}
            if "query_intent" not in result:
                result["query_intent"] = "general"
            if "resolved_references" not in result:
                result["resolved_references"] = {}
            if "response_metadata" not in result:
                result["response_metadata"] = {}
            if "formatted_response" not in result:
                result["formatted_response"] = "I apologize, but I couldn't process your request properly."
            
            # Build the response
            response_data = {
                "success": not result.get("has_errors", False),
                "result": result["formatted_response"],
                "metadata": {
                    **result["response_metadata"],
                    "mcp_context_summary": result.get("mcp_context_summary", {}),
                    "mcp_references_resolved": len(result.get("mcp_references", {})),
                    "context_items_used": result.get("mcp_context_summary", {}).get("total_contexts", 0)
                },
                "tool_name": result.get("selected_tool", "unknown"),
                "session_id": session_id,
                "conversation_context": {
                    "patient_context": result.get("patient_context", {}),
                    "query_intent": result.get("query_intent", "general"),
                    "resolved_references": result.get("resolved_references", {}),
                    "mcp_enhanced": True
                }
            }
            
            # Debug: Log the final response structure
            logger.info(f"Final response keys: {list(response_data.keys())}")
            logger.info(f"Response has conversation_context: {'conversation_context' in response_data}")
            
            # Return enhanced result with MCP metadata
            return response_data
            
        except Exception as e:
            logger.error(f"MCP agent processing error: {e}")
            return {
                "success": False,
                "result": f"I encountered an error processing your request: {str(e)}",
                "metadata": {"error": str(e), "mcp_enhanced": True},
                "tool_name": "error_handler",
                "session_id": session_id,
                "conversation_context": {
                    "patient_context": {},
                    "query_intent": "error_handling",
                    "resolved_references": {},
                    "mcp_enhanced": True,
                    "error": str(e)
                }
            }
    
    def process_message_sync(
        self, 
        session_id: str, 
        message: str, 
        user_role: str = "assistant", 
        doctor_id: Optional[str] = None,
        identity_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_message for compatibility with existing API.
        """
        import asyncio
        import nest_asyncio
        
        try:
            # Enable nested event loops
            nest_asyncio.apply()
            
            # Run the async method directly
            return asyncio.run(self.process_message(session_id, message, user_role, doctor_id))
            
        except Exception as e:
            return {
                "success": False,
                "result": f"Error in process_message_sync: {str(e)}",
                "metadata": {"error": str(e), "mcp_enhanced": True},
                "tool_name": "sync_error_handler",
                "session_id": session_id,
                "conversation_context": {
                    "patient_context": {},
                    "query_intent": "sync_error_handling",
                    "resolved_references": {},
                    "mcp_enhanced": True,
                    "error": str(e)
                }
            }
    
    async def _load_mcp_context_for_session(self, session_id: str, state: AgentState):
        """Load and prepare MCP context for a session."""
        try:
            # Get recent conversation history for context
            recent_turns = self.conversation_memory.get_conversation_history(session_id, limit=5)
            
            if recent_turns:
                # Extract recent queries for context building
                recent_queries = [turn.user_input for turn in recent_turns]
                recent_responses = [turn.agent_response for turn in recent_turns]
                
                # Update conversation memory in state
                state["conversation_memory"]["recent_queries"] = recent_queries
                state["conversation_memory"]["conversation_flow"] = recent_responses
                
                # Get MCP context summary
                context_summary = self.mcp_manager.get_context_summary(session_id)
                state["mcp_context_summary"] = context_summary
                
                logger.info(f"Loaded MCP context for session {session_id}: {context_summary}")
                
        except Exception as e:
            logger.error(f"Error loading MCP context: {e}")
    
    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session context including MCP data."""
        if session_id in self.sessions:
            state = self.sessions[session_id]
            
            # Get MCP context summary
            mcp_summary = self.mcp_manager.get_context_summary(session_id)
            
            return {
                "patient_context": state.get("patient_context"),
                "doctor_context": state.get("doctor_context"),
                "conversation_memory": state["conversation_memory"],
                "message_count": len(state["messages"]),
                "mcp_context_summary": mcp_summary,
                "mcp_enhanced": True
            }
        return {"mcp_enhanced": True}
    
    def clear_session(self, session_id: str):
        """Clear a specific session and its MCP context."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Clean up MCP context for this session
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.mcp_manager.cleanup_old_contexts(session_id, max_age_hours=0)
            )
        finally:
            loop.close()
    
    def list_active_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self.sessions.keys())
    
    async def get_mcp_context_summary_for_all_sessions(self) -> Dict[str, Any]:
        """Get MCP context summary across all active sessions."""
        summaries = {}
        for session_id in self.sessions.keys():
            summaries[session_id] = self.mcp_manager.get_context_summary(session_id)
        
        return {
            "total_active_sessions": len(summaries),
            "session_summaries": summaries,
            "total_contexts": sum(s.get("total_contexts", 0) for s in summaries.values())
        }
