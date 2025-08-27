"""
Memory Management for LangGraph Medical Assistant

This module provides conversation memory and context persistence
for multi-turn conversations with enhanced MCP (Model Context Protocol) integration.
"""

import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

# Memory database path - use absolute path to avoid working directory issues
_current_dir = Path(__file__).parent.parent.parent  # Go up to code directory
MEMORY_DB_PATH = str(_current_dir / "db" / "conversation_memory.db")

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    session_id: str
    turn_number: int
    user_input: str
    agent_response: str
    intent: str
    tool_used: str
    context_resolved: Dict[str, Any]
    timestamp: datetime
    mcp_context_ids: Optional[List[str]] = None  # Track associated MCP context items
    
    def __post_init__(self):
        if self.mcp_context_ids is None:
            self.mcp_context_ids = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['context_resolved'] = json.dumps(self.context_resolved)
        data['mcp_context_ids'] = json.dumps(self.mcp_context_ids)
        return data

class ConversationMemory:
    """
    Enhanced conversation memory manager with MCP integration.
    
    Features:
    - Traditional SQLite-based conversation storage
    - MCP context item tracking
    - Cross-reference between conversation turns and MCP contexts
    - Enhanced context retrieval with MCP data
    """
    
    def __init__(self):
        self._init_memory_db()
    
    def _init_memory_db(self):
        """Initialize the memory database with MCP support."""
        # Create directory if it doesn't exist
        Path(MEMORY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        # Create conversation turns table with MCP integration
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                user_input TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                intent TEXT,
                tool_used TEXT,
                context_resolved TEXT,
                mcp_context_ids TEXT,
                timestamp TEXT NOT NULL,
                UNIQUE(session_id, turn_number)
            )
        """)
        
        # Enhanced session context table with MCP references
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_context (
                session_id TEXT PRIMARY KEY,
                patient_context TEXT,
                doctor_context TEXT,
                implicit_references TEXT,
                mcp_context_summary TEXT,
                last_updated TEXT NOT NULL
            )
        """)
        
        # MCP context mapping table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mcp_context_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                mcp_context_id TEXT NOT NULL,
                context_type TEXT NOT NULL,
                context_content TEXT,
                created_at TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                UNIQUE(session_id, mcp_context_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_conversation_turn(
        self, 
        turn: ConversationTurn, 
        mcp_context_ids: Optional[List[str]] = None
    ):
        """Save a conversation turn with MCP context references."""
        if mcp_context_ids:
            turn.mcp_context_ids = mcp_context_ids
            
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        turn_data = turn.to_dict()
        cursor.execute("""
            INSERT OR REPLACE INTO conversation_turns 
            (session_id, turn_number, user_input, agent_response, intent, tool_used, 
             context_resolved, mcp_context_ids, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            turn_data['session_id'],
            turn_data['turn_number'],
            turn_data['user_input'],
            turn_data['agent_response'],
            turn_data['intent'],
            turn_data['tool_used'],
            turn_data['context_resolved'],
            turn_data['mcp_context_ids'],
            turn_data['timestamp']
        ))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(
        self, 
        session_id: str, 
        limit: int = 10,
        include_mcp_context: bool = True
    ) -> List[ConversationTurn]:
        """Get conversation history with optional MCP context enrichment."""
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM conversation_turns 
            WHERE session_id = ? 
            ORDER BY turn_number DESC 
            LIMIT ?
        """, (session_id, limit))
        
        rows = cursor.fetchall()
        
        turns = []
        for row in rows:
            mcp_context_ids = json.loads(row[8]) if row[8] else []
            
            turn = ConversationTurn(
                session_id=row[1],
                turn_number=row[2],
                user_input=row[3],
                agent_response=row[4],
                intent=row[5],
                tool_used=row[6],
                context_resolved=json.loads(row[7]) if row[7] else {},
                timestamp=datetime.fromisoformat(row[9]),
                mcp_context_ids=mcp_context_ids
            )
            
            # Enrich with MCP context if requested
            if include_mcp_context and mcp_context_ids:
                # Here you could add logic to fetch and attach MCP context data
                # from the mcp_context_manager if needed
                pass
                
            turns.append(turn)
        
        conn.close()
        return list(reversed(turns))  # Return in chronological order
    
    def save_session_context(
        self, 
        session_id: str, 
        context: Dict[str, Any],
        mcp_context_summary: Optional[Dict[str, Any]] = None
    ):
        """Save session context with MCP integration."""
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO session_context 
            (session_id, patient_context, doctor_context, implicit_references, 
             mcp_context_summary, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            json.dumps(context.get('patient_context')),
            json.dumps(context.get('doctor_context')),
            json.dumps(context.get('implicit_references')),
            json.dumps(mcp_context_summary) if mcp_context_summary else None,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get session context with MCP integration."""
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT patient_context, doctor_context, implicit_references, mcp_context_summary 
            FROM session_context 
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            context = {
                'patient_context': json.loads(row[0]) if row[0] else None,
                'doctor_context': json.loads(row[1]) if row[1] else None,
                'implicit_references': json.loads(row[2]) if row[2] else {},
                'mcp_context_summary': json.loads(row[3]) if row[3] else None
            }
            return context
        
        return {}
    
    def add_mcp_context_mapping(
        self,
        session_id: str,
        mcp_context_id: str,
        context_type: str,
        context_content: Dict[str, Any],
        relevance_score: float = 1.0
    ):
        """Add mapping between session and MCP context."""
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO mcp_context_mapping 
            (session_id, mcp_context_id, context_type, context_content, created_at, relevance_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            mcp_context_id,
            context_type,
            json.dumps(context_content),
            datetime.now().isoformat(),
            relevance_score
        ))
        
        conn.commit()
        conn.close()
    
    def get_mcp_context_mappings(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all MCP context mappings for a session."""
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT mcp_context_id, context_type, context_content, created_at, relevance_score
            FROM mcp_context_mapping 
            WHERE session_id = ?
            ORDER BY relevance_score DESC, created_at DESC
        """, (session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        mappings = []
        for row in rows:
            mappings.append({
                'mcp_context_id': row[0],
                'context_type': row[1],
                'context_content': json.loads(row[2]),
                'created_at': datetime.fromisoformat(row[3]),
                'relevance_score': row[4]
            })
        
        return mappings
        
    def cleanup_old_sessions(self, days: int = 7):
        """Clean up old session data including MCP mappings."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        # Clean up old conversation turns
        cursor.execute("""
            DELETE FROM conversation_turns 
            WHERE timestamp < ?
        """, (cutoff_date.isoformat(),))
        
        # Clean up old session context
        cursor.execute("""
            DELETE FROM session_context 
            WHERE last_updated < ?
        """, (cutoff_date.isoformat(),))
        
        # Clean up old MCP mappings
        cursor.execute("""
            DELETE FROM mcp_context_mapping 
            WHERE created_at < ?
        """, (cutoff_date.isoformat(),))
        
        conn.commit()
        conn.close()
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT session_id 
            FROM conversation_turns 
            WHERE timestamp > ?
        """, ((datetime.now() - timedelta(hours=24)).isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in rows]
    
    def get_enhanced_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive context summary including MCP data."""
        traditional_context = self.get_session_context(session_id)
        mcp_mappings = self.get_mcp_context_mappings(session_id)
        recent_turns = self.get_conversation_history(session_id, limit=5)
        
        return {
            "session_id": session_id,
            "traditional_context": traditional_context,
            "mcp_mappings_count": len(mcp_mappings),
            "mcp_context_types": list(set(m['context_type'] for m in mcp_mappings)),
            "recent_turns_count": len(recent_turns),
            "last_activity": recent_turns[0].timestamp.isoformat() if recent_turns else None,
            "active_patients": [
                m['context_content'].get('patient_name') 
                for m in mcp_mappings 
                if m['context_type'] == 'patient' and m['context_content'].get('patient_name')
            ]
        }
