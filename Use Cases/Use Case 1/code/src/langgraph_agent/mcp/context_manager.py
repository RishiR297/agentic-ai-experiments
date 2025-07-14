"""
MCimport json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdictel Context Protocol) Integration for LangGraph Agent

This module implements proper MCP context management for preserving
conversation context across turns using the Model Context Protocol.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from ..memory.conversation_memory import ConversationMemory


@dataclass
class MCPContextItem:
    """Represents a context item in MCP format."""
    context_id: str
    context_type: str  # "conversation", "patient", "appointment", "medical_history"
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    relevance_score: float = 1.0
    
    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert to MCP protocol format."""
        return {
            "id": self.context_id,
            "type": self.context_type,
            "content": self.content,
            "metadata": {
                **self.metadata,
                "created_at": self.created_at.isoformat(),
                "relevance_score": self.relevance_score
            }
        }


class MCPContextManager:
    """
    MCP Context Manager for preserving conversational context.
    
    Implements Model Context Protocol standards for:
    - Context persistence across conversation turns
    - Reference resolution using MCP context items
    - Context scoring and relevance ranking
    - Cross-session context sharing when appropriate
    """
    
    def __init__(self, conversation_memory, max_context_items: int = 5):
        self.conversation_memory = conversation_memory
        self.active_contexts: Dict[str, List[MCPContextItem]] = {}
        self.max_context_items = max_context_items  # Enforce 5-item limit
        
    async def create_context_item(
        self,
        session_id: str,
        context_type: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> MCPContextItem:
        """Create a new MCP context item with automatic pruning."""
        
        context_id = f"{session_id}_{context_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        context_item = MCPContextItem(
            context_id=context_id,
            context_type=context_type,
            content=content,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        # Store in active contexts
        if session_id not in self.active_contexts:
            self.active_contexts[session_id] = []
        
        self.active_contexts[session_id].append(context_item)
        
        # Enforce 5-item limit with temporal decay and relevance scoring
        await self._enforce_context_limit(session_id)
        
        # Persist to conversation memory
        self.conversation_memory.add_mcp_context_mapping(
            session_id=session_id,
            mcp_context_id=context_id,
            context_type=context_type,
            context_content=content,
            relevance_score=context_item.relevance_score
        )
        
        print(f"🧠 MCP: Created context item {context_id} (type: {context_type})")
        print(f"📊 MCP: Session {session_id} now has {len(self.active_contexts[session_id])} context items")
        
        return context_item
    
    async def _enforce_context_limit(self, session_id: str):
        """Enforce the 5-item context limit with temporal decay."""
        if session_id not in self.active_contexts:
            return
        
        context_items = self.active_contexts[session_id]
        
        if len(context_items) > self.max_context_items:
            # Sort by relevance score and recency (temporal decay)
            now = datetime.now()
            
            def context_score(item: MCPContextItem) -> float:
                # Temporal decay: reduce relevance based on age
                age_hours = (now - item.created_at).total_seconds() / 3600
                decay_factor = max(0.1, 1.0 - (age_hours * 0.1))  # 10% decay per hour
                
                return item.relevance_score * decay_factor
            
            # Sort by combined score (relevance + temporal decay)
            sorted_items = sorted(context_items, key=context_score, reverse=True)
            
            # Keep only the top 5 most relevant/recent items
            self.active_contexts[session_id] = sorted_items[:self.max_context_items]
            
            removed_count = len(context_items) - self.max_context_items
            print(f"🗑️  MCP: Pruned {removed_count} old context items from session {session_id}")
            print(f"📊 MCP: Retained {len(self.active_contexts[session_id])} most relevant items")
    
    async def get_relevant_context(
        self,
        session_id: str,
        query: str,
        context_types: Optional[List[str]] = None,
        max_items: int = 5
    ) -> List[MCPContextItem]:
        """Retrieve relevant MCP context items for a query."""
        
        # Load contexts from persistent storage
        await self._load_session_contexts(session_id)
        
        if session_id not in self.active_contexts:
            return []
        
        contexts = self.active_contexts[session_id]
        
        # Filter by context types if specified
        if context_types:
            contexts = [c for c in contexts if c.context_type in context_types]
        
        # Score contexts based on relevance to current query
        scored_contexts = []
        for context in contexts:
            score = await self._calculate_relevance_score(context, query)
            context.relevance_score = score
            scored_contexts.append(context)
        
        # Sort by relevance score and return top items
        scored_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_contexts[:max_items]
    
    async def update_context_from_conversation_turn(
        self,
        session_id: str,
        user_input: str,
        agent_response: str,
        intent: str,
        tool_used: str,
        context_resolved: Dict[str, Any]
    ):
        """Update MCP context based on a conversation turn."""
        
        # Create conversation context item
        conversation_context = await self.create_context_item(
            session_id=session_id,
            context_type="conversation",
            content={
                "user_input": user_input,
                "agent_response": agent_response,
                "intent": intent,
                "tool_used": tool_used,
                "resolved_references": context_resolved
            },
            metadata={
                "turn_timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
        )
        
        # Extract and create patient context items if mentioned
        if "patient" in context_resolved:
            patient_info = context_resolved["patient"]
            await self.create_context_item(
                session_id=session_id,
                context_type="patient",
                content={
                    "patient_name": patient_info.get("name"),
                    "patient_id": patient_info.get("id"),
                    "mentioned_in_context": user_input,
                    "associated_response": agent_response
                },
                metadata={
                    "conversation_turn": conversation_context.context_id,
                    "intent": intent
                }
            )
        
        # Extract appointment context if relevant
        if intent in ["schedule", "next_patient", "appointment_lookup"] and "appointment" in agent_response.lower():
            await self.create_context_item(
                session_id=session_id,
                context_type="appointment",
                content={
                    "query_intent": intent,
                    "tool_used": tool_used,
                    "appointment_data": context_resolved,
                    "response_summary": agent_response[:200]  # Truncated response
                },
                metadata={
                    "conversation_turn": conversation_context.context_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def resolve_references_with_mcp(
        self,
        session_id: str,
        query: str
    ) -> Dict[str, Any]:
        """Resolve implicit references using MCP context."""
        
        # Get relevant context items
        relevant_contexts = await self.get_relevant_context(
            session_id=session_id,
            query=query,
            max_items=10
        )
        
        resolved_references = {}
        
        # Analyze query for implicit references
        query_lower = query.lower()
        
        # Resolve "next patient" references
        if "next" in query_lower and "patient" in query_lower:
            patient_contexts = [c for c in relevant_contexts if c.context_type == "patient"]
            if patient_contexts:
                latest_patient = patient_contexts[0]
                resolved_references["next_patient"] = latest_patient.content.get("patient_name")
        
        # Resolve "her/his/their" references
        pronouns = ["her", "his", "their", "she", "he", "they"]
        if any(pronoun in query_lower for pronoun in pronouns):
            patient_contexts = [c for c in relevant_contexts if c.context_type == "patient"]
            if patient_contexts:
                latest_patient = patient_contexts[0]
                pronoun_found = next((p for p in pronouns if p in query_lower), None)
                resolved_references[pronoun_found] = latest_patient.content.get("patient_name")
        
        # Resolve "that appointment" references
        if "that" in query_lower and "appointment" in query_lower:
            appointment_contexts = [c for c in relevant_contexts if c.context_type == "appointment"]
            if appointment_contexts:
                latest_appointment = appointment_contexts[0]
                resolved_references["that appointment"] = latest_appointment.content.get("appointment_data")
        
        return resolved_references
    
    async def get_mcp_context_for_prompt(
        self,
        session_id: str,
        current_query: str,
        max_context_length: int = 2000
    ) -> str:
        """Generate MCP context string for inclusion in LLM prompts."""
        
        relevant_contexts = await self.get_relevant_context(
            session_id=session_id,
            query=current_query
        )
        
        if not relevant_contexts:
            return ""
        
        context_parts = []
        context_parts.append("=== RELEVANT CONVERSATION CONTEXT (MCP) ===")
        
        for context in relevant_contexts:
            context_summary = f"""
Context Type: {context.context_type}
Relevance: {context.relevance_score:.2f}
Created: {context.created_at.strftime('%H:%M:%S')}
Content: {json.dumps(context.content, indent=2)[:200]}...
"""
            context_parts.append(context_summary)
            
            # Check if we're approaching the length limit
            current_length = len("\n".join(context_parts))
            if current_length > max_context_length:
                break
        
        context_parts.append("=== END MCP CONTEXT ===")
        return "\n".join(context_parts)
    
    async def _load_session_contexts(self, session_id: str):
        """Load MCP contexts from persistent storage."""
        if session_id in self.active_contexts:
            return  # Already loaded
        
        mappings = self.conversation_memory.get_mcp_context_mappings(session_id)
        
        contexts = []
        for mapping in mappings:
            context_item = MCPContextItem(
                context_id=mapping['mcp_context_id'],
                context_type=mapping['context_type'],
                content=mapping['context_content'],
                metadata={},
                created_at=mapping['created_at'],
                relevance_score=mapping['relevance_score']
            )
            contexts.append(context_item)
        
        self.active_contexts[session_id] = contexts
    
    async def _calculate_relevance_score(
        self,
        context: MCPContextItem,
        query: str
    ) -> float:
        """Calculate relevance score for a context item."""
        
        score = 0.0
        query_lower = query.lower()
        
        # Recency score (more recent = higher score)
        time_diff = datetime.now() - context.created_at
        hours_old = time_diff.total_seconds() / 3600
        recency_score = max(0, 1.0 - (hours_old / 24))  # Decay over 24 hours
        
        # Content relevance score
        content_text = json.dumps(context.content).lower()
        common_words = set(query_lower.split()) & set(content_text.split())
        content_score = len(common_words) / max(len(query_lower.split()), 1)
        
        # Context type relevance
        type_score = 1.0
        if context.context_type == "patient" and "patient" in query_lower:
            type_score = 1.5
        elif context.context_type == "appointment" and any(word in query_lower for word in ["appointment", "schedule", "next"]):
            type_score = 1.5
        
        # Combined score
        score = (recency_score * 0.3 + content_score * 0.5 + type_score * 0.2)
        
        return min(score, 1.0)
    
    async def cleanup_old_contexts(self, session_id: str, max_age_hours: int = 24):
        """Clean up old MCP contexts."""
        if session_id not in self.active_contexts:
            return
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Filter out old contexts
        fresh_contexts = [
            c for c in self.active_contexts[session_id]
            if c.created_at > cutoff_time
        ]
        
        self.active_contexts[session_id] = fresh_contexts
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of active MCP contexts for a session."""
        if session_id not in self.active_contexts:
            return {"total_contexts": 0, "context_types": []}
        
        contexts = self.active_contexts[session_id]
        context_types = {}
        
        for context in contexts:
            if context.context_type not in context_types:
                context_types[context.context_type] = 0
            context_types[context.context_type] += 1
        
        return {
            "total_contexts": len(contexts),
            "context_types": context_types,
            "most_recent": max(contexts, key=lambda x: x.created_at).created_at.isoformat() if contexts else None,
            "average_relevance": sum(c.relevance_score for c in contexts) / len(contexts) if contexts else 0
        }
