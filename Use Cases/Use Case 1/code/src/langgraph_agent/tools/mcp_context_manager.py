"""
MCP (Model Context Protocol) Integration for Enhanced Context Preservation

This module provides MCP-compliant context management for maintaining
conversational state across multiple turns with standardized protocols.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MCPContextItem:
    """MCP-compliant context item for conversation memory."""
    id: str
    type: str  # "patient", "appointment", "reference", "session"
    content: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    session_id: str = ""
    relevance_score: float = 1.0


class MCPContextManager:
    """
    MCP-compliant context manager for preserving conversational state.
    
    Features:
    - Standardized context protocols
    - Automatic context expiration
    - Reference resolution tracking
    - Cross-session context sharing
    """
    
    def __init__(self, max_context_items: int = 50):
        self.max_context_items = max_context_items
        self.context_store: Dict[str, MCPContextItem] = {}
        self.session_contexts: Dict[str, List[str]] = {}  # session_id -> context_item_ids
    
    def add_context_item(
        self,
        item_type: str,
        content: Dict[str, Any],
        session_id: str,
        expires_in_hours: Optional[int] = 24,
        relevance_score: float = 1.0
    ) -> str:
        """Add a context item following MCP standards."""
        
        item_id = f"{item_type}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        context_item = MCPContextItem(
            id=item_id,
            type=item_type,
            content=content,
            created_at=datetime.now(),
            expires_at=expires_at,
            session_id=session_id,
            relevance_score=relevance_score
        )
        
        # Store context item
        self.context_store[item_id] = context_item
        
        # Update session tracking
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = []
        self.session_contexts[session_id].append(item_id)
        
        # Clean up old items if needed
        self._cleanup_expired_contexts()
        self._enforce_max_items()
        
        logger.info(f"Added MCP context item: {item_id} (type: {item_type})")
        return item_id
    
    def get_session_context(self, session_id: str) -> List[MCPContextItem]:
        """Get all relevant context items for a session."""
        if session_id not in self.session_contexts:
            return []
        
        context_items = []
        for item_id in self.session_contexts[session_id]:
            if item_id in self.context_store:
                item = self.context_store[item_id]
                # Check if item has expired
                if item.expires_at and datetime.now() > item.expires_at:
                    continue
                context_items.append(item)
        
        # Sort by relevance and recency
        context_items.sort(key=lambda x: (x.relevance_score, x.created_at), reverse=True)
        return context_items
    
    def resolve_reference(
        self,
        reference: str,
        session_id: str,
        context_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        MCP-compliant reference resolution.
        
        Examples:
        - "next patient" -> patient context from recent appointment lookup
        - "her" -> most recent patient mentioned
        - "that appointment" -> most recent appointment discussed
        """
        
        context_items = self.get_session_context(session_id)
        
        # Define reference patterns
        reference_patterns = {
            "next_patient": ["next patient", "upcoming patient", "next appointment"],
            "current_patient": ["her", "him", "the patient", "this patient"],
            "recent_appointment": ["that appointment", "the appointment", "this appointment"],
            "doctor_schedule": ["my schedule", "my appointments", "today's schedule"]
        }
        
        # Find matching pattern
        matched_pattern = None
        for pattern_type, patterns in reference_patterns.items():
            if any(pattern.lower() in reference.lower() for pattern in patterns):
                matched_pattern = pattern_type
                break
        
        if not matched_pattern:
            return None
        
        # Search for relevant context
        for item in context_items:
            if context_type and item.type != context_type:
                continue
                
            if matched_pattern == "next_patient" and item.type == "appointment":
                if "next" in item.content.get("query_intent", ""):
                    return item.content
                    
            elif matched_pattern == "current_patient" and item.type == "patient":
                return item.content
                
            elif matched_pattern == "recent_appointment" and item.type == "appointment":
                return item.content
                
            elif matched_pattern == "doctor_schedule" and item.type == "schedule":
                return item.content
        
        return None
    
    def update_context_relevance(self, item_id: str, new_score: float):
        """Update relevance score for context prioritization."""
        if item_id in self.context_store:
            self.context_store[item_id].relevance_score = new_score
            logger.info(f"Updated relevance for {item_id}: {new_score}")
    
    def add_patient_context(
        self,
        patient_name: str,
        patient_id: str,
        appointment_details: Dict[str, Any],
        session_id: str
    ) -> str:
        """Add patient context following MCP patient data standards."""
        
        patient_context = {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "appointment_details": appointment_details,
            "mentioned_at": datetime.now().isoformat(),
            "context_type": "patient_reference"
        }
        
        return self.add_context_item(
            item_type="patient",
            content=patient_context,
            session_id=session_id,
            expires_in_hours=24,
            relevance_score=0.9
        )
    
    def add_appointment_context(
        self,
        query_intent: str,
        appointments: List[Dict[str, Any]],
        session_id: str
    ) -> str:
        """Add appointment context following MCP appointment standards."""
        
        appointment_context = {
            "query_intent": query_intent,
            "appointments": appointments,
            "query_time": datetime.now().isoformat(),
            "context_type": "appointment_lookup"
        }
        
        return self.add_context_item(
            item_type="appointment",
            content=appointment_context,
            session_id=session_id,
            expires_in_hours=8,
            relevance_score=0.8
        )
    
    def add_schedule_context(
        self,
        schedule_data: List[Dict[str, Any]],
        date: str,
        session_id: str
    ) -> str:
        """Add schedule context following MCP schedule standards."""
        
        schedule_context = {
            "schedule_data": schedule_data,
            "date": date,
            "queried_at": datetime.now().isoformat(),
            "context_type": "schedule_lookup"
        }
        
        return self.add_context_item(
            item_type="schedule",
            content=schedule_context,
            session_id=session_id,
            expires_in_hours=12,
            relevance_score=0.7
        )
    
    def _cleanup_expired_contexts(self):
        """Remove expired context items."""
        now = datetime.now()
        expired_items = []
        
        for item_id, item in self.context_store.items():
            if item.expires_at and now > item.expires_at:
                expired_items.append(item_id)
        
        for item_id in expired_items:
            self._remove_context_item(item_id)
            logger.info(f"Removed expired context item: {item_id}")
    
    def _enforce_max_items(self):
        """Enforce maximum context items limit."""
        if len(self.context_store) <= self.max_context_items:
            return
        
        # Sort by relevance and age, remove lowest scoring items
        items = list(self.context_store.values())
        items.sort(key=lambda x: (x.relevance_score, x.created_at))
        
        items_to_remove = len(items) - self.max_context_items
        for i in range(items_to_remove):
            self._remove_context_item(items[i].id)
    
    def _remove_context_item(self, item_id: str):
        """Remove a context item and update session tracking."""
        if item_id in self.context_store:
            item = self.context_store[item_id]
            del self.context_store[item_id]
            
            # Update session tracking
            if item.session_id in self.session_contexts:
                if item_id in self.session_contexts[item.session_id]:
                    self.session_contexts[item.session_id].remove(item_id)
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get MCP-compliant context summary for a session."""
        context_items = self.get_session_context(session_id)
        
        summary = {
            "session_id": session_id,
            "total_items": len(context_items),
            "context_types": {},
            "recent_references": [],
            "active_patients": [],
            "active_appointments": []
        }
        
        for item in context_items:
            # Count by type
            if item.type not in summary["context_types"]:
                summary["context_types"][item.type] = 0
            summary["context_types"][item.type] += 1
            
            # Extract key information
            if item.type == "patient":
                summary["active_patients"].append({
                    "name": item.content.get("patient_name"),
                    "id": item.content.get("patient_id"),
                    "relevance": item.relevance_score
                })
            elif item.type == "appointment":
                summary["active_appointments"].extend(
                    item.content.get("appointments", [])[:2]  # Top 2 appointments
                )
        
        return summary
    
    def export_session_context(self, session_id: str) -> str:
        """Export session context in MCP-compliant JSON format."""
        context_items = self.get_session_context(session_id)
        
        export_data = {
            "mcp_version": "1.0",
            "session_id": session_id,
            "exported_at": datetime.now().isoformat(),
            "context_items": [
                {
                    "id": item.id,
                    "type": item.type,
                    "content": item.content,
                    "created_at": item.created_at.isoformat(),
                    "relevance_score": item.relevance_score
                }
                for item in context_items
            ]
        }
        
        return json.dumps(export_data, indent=2, default=str)


# Global MCP context manager instance
mcp_context_manager = MCPContextManager()
