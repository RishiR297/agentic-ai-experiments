"""
Enhanced Streamlit Web Interface for LangGraph Medical Assistant

Provides a user-friendly chat interface with session management,
context awareness, and multi-turn conversation support.
"""

import streamlit as st
import requests
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="🏥 LangGraph Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://127.0.0.1:8001"

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "user_role" not in st.session_state:
        st.session_state.user_role = "doctor"
    
    if "doctor_id" not in st.session_state:
        st.session_state.doctor_id = "11712738-BFDE-436E-950B-2731FA20DDB2"  # Default to Dr. Antonella
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = {}

def send_chat_message(message: str, user_role: str, doctor_id: str = None, session_id: str = None) -> Dict[str, Any]:
    """Send a message to the LangGraph backend API."""
    try:
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        
        if user_role == "doctor" and doctor_id:
            headers["X-Doctor-ID"] = doctor_id
            headers["X-User-Role"] = "doctor"
        else:
            headers["X-User-Role"] = "assistant"
        
        if session_id:
            headers["X-Session-ID"] = session_id
        
        # Prepare payload for the /chat endpoint
        payload = {
            "user_input": message,
            "doctor_id": doctor_id if user_role == "doctor" else None,
            "user_role": user_role,
            "session_id": session_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            response_data = response.json()
            return {
                "response": response_data.get("result", "No response"),
                "success": response_data.get("success", False),
                "tool_name": response_data.get("tool_name", "unknown"),
                "metadata": response_data.get("metadata", {}),
                "session_id": response_data.get("session_id"),
                "conversation_context": response_data.get("conversation_context", {})
            }
        else:
            return {
                "response": f"Error: {response.status_code} - {response.text}",
                "success": False
            }
    except requests.exceptions.RequestException as e:
        return {
            "response": f"Connection error: {str(e)}",
            "success": False
        }

def get_session_context(session_id: str) -> Dict[str, Any]:
    """Get current session context from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/session/{session_id}/context", timeout=5)
        if response.status_code == 200:
            return response.json().get("context", {})
    except:
        pass
    return {}

def get_user_tools(user_role: str, doctor_id: str = None) -> List[Dict]:
    """Get available tools for the current user role."""
    try:
        headers = {}
        if user_role == "doctor" and doctor_id:
            headers["X-Doctor-ID"] = doctor_id
        elif user_role == "assistant":
            headers["X-User-Role"] = "assistant"
        
        response = requests.get(f"{API_BASE_URL}/tools", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("tools", [])
    except:
        pass
    return []

def check_system_health() -> Dict[str, bool]:
    """Check the health of backend systems."""
    health_status = {
        "backend": False,
        "backend_details": {}
    }
    
    # Check backend health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_status["backend"] = True
            health_status["backend_details"] = response.json()
    except:
        pass
    
    return health_status

def render_chat_message(role: str, content: str, timestamp: str = None, metadata: Dict = None):
    """Render a chat message with proper styling and context information."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        with st.chat_message("user"):
            st.markdown(f"**You** ({timestamp})")
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Medical Assistant** ({timestamp})")
            st.markdown(content)
            
            # Show metadata if available
            if metadata:
                with st.expander("🔍 Response Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if "intent" in metadata:
                            st.write(f"**Intent:** {metadata['intent']}")
                        if "tool_used" in metadata:
                            st.write(f"**Tool Used:** {metadata['tool_used']}")
                        if "context_resolved" in metadata:
                            context_status = "✅ Yes" if metadata["context_resolved"] else "❌ No"
                            st.write(f"**Context Resolved:** {context_status}")
                    with col2:
                        if "has_errors" in metadata:
                            error_status = "❌ Yes" if metadata["has_errors"] else "✅ No"
                            st.write(f"**Errors:** {error_status}")

def render_context_sidebar():
    """Render the conversation context in the sidebar."""
    context = st.session_state.conversation_context
    
    if context:
        st.subheader("💭 Conversation Context")
        
        # Patient context
        patient_context = context.get("patient_context")
        if patient_context:
            with st.expander("👤 Current Patient", expanded=True):
                if patient_context.get("patient_name"):
                    st.write(f"**Name:** {patient_context['patient_name']}")
                if patient_context.get("appointment_date"):
                    st.write(f"**Appointment:** {patient_context['appointment_date']}")
                if patient_context.get("last_mentioned"):
                    st.write(f"**Last Mentioned:** {patient_context['last_mentioned']}")
        
        # Query intent
        query_intent = context.get("query_intent")
        if query_intent:
            st.write(f"**Last Intent:** {query_intent}")
        
        # Resolved references
        resolved_refs = context.get("resolved_references")
        if resolved_refs:
            with st.expander("🔗 Resolved References"):
                for ref, value in resolved_refs.items():
                    st.write(f"**{ref}:** {value}")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("🏥 LangGraph Medical Assistant")
    st.markdown("*Multi-turn conversational agent with context awareness and memory*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("👤 User Configuration")
        
        # User role selection
        user_role = st.selectbox(
            "Select your role:",
            ["doctor", "assistant"],
            index=0 if st.session_state.user_role == "doctor" else 1,
            key="role_select"
        )
        st.session_state.user_role = user_role
        
        # Doctor ID input (if doctor)
        if user_role == "doctor":
            # Predefined doctor options for easy selection
            doctor_options = {
                "Dr. Antonella": "11712738-BFDE-436E-950B-2731FA20DDB2",
                "Custom Doctor ID": "custom"
            }
            
            selected_doctor = st.selectbox(
                "Select Doctor:",
                options=list(doctor_options.keys()),
                index=0,
                key="doctor_select"
            )
            
            if selected_doctor == "Custom Doctor ID":
                doctor_id = st.text_input(
                    "Custom Doctor ID:",
                    value=st.session_state.doctor_id if st.session_state.doctor_id not in doctor_options.values() else "",
                    help="Enter UUID format: 11712738-BFDE-436E-950B-2731FA20DDB2",
                    key="doctor_id_input"
                )
            else:
                doctor_id = doctor_options[selected_doctor]
            
            st.session_state.doctor_id = doctor_id
            
            # Show role info
            st.info("🩺 **Doctor Access**: Full access to appointments, schedules, and patient information")
        else:
            st.info("🏢 **Assistant Access**: Schedule viewing and availability checking only")
        
        st.divider()
        
        # Session management
        st.header("🔄 Session Management")
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Session", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.conversation_context = {}
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh Context", use_container_width=True):
                context = get_session_context(st.session_state.session_id)
                st.session_state.conversation_context = context
                st.rerun()
        
        # Render conversation context
        render_context_sidebar()
        
        st.divider()
        
        # System status
        st.header("🔧 System Status")
        health_status = check_system_health()
        
        if health_status["backend"]:
            st.success("✅ LangGraph Agent: Online")
            backend_details = health_status["backend_details"]
            if backend_details.get("active_sessions"):
                st.caption(f"Active sessions: {backend_details['active_sessions']}")
        else:
            st.error("❌ LangGraph Agent: Offline")
        
        # Show available tools for current role
        if health_status["backend"]:
            tools = get_user_tools(user_role, st.session_state.doctor_id if user_role == "doctor" else None)
            if tools:
                with st.expander(f"🛠️ Available Tools ({len(tools)})"):
                    for tool in tools:
                        st.write(f"• **{tool['name']}**")
                        st.caption(tool.get('description', 'No description'))
        
        st.divider()
        
        # Multi-turn conversation examples
        st.header("💡 Multi-Turn Examples")
        
        if user_role == "doctor":
            examples = [
                ("What are my appointments today?", "Get today's schedule"),
                ("Who is my next patient?", "Find next appointment"),
                ("Tell me about her medical history", "Use context from previous query"),
                ("When is my next available slot?", "Check availability"),
                ("Summarize my day", "Get schedule overview")
            ]
        else:
            examples = [
                ("Show me Dr. Smith's schedule", "View doctor schedule"),
                ("Is the doctor available at 2 PM?", "Check specific availability"),
                ("What about tomorrow?", "Context-aware follow-up"),
                ("List all cardiologists", "Find specialists"),
                ("Who has openings this week?", "Check multiple availabilities")
            ]
        
        for query, description in examples:
            if st.button(f"💬 {query}", key=f"example_{hash(query)}", use_container_width=True, help=description):
                st.session_state.sample_query = query

    # Main chat interface
    st.header("💬 Multi-Turn Chat Interface")
    
    # Show current role and session info
    role_color = "blue" if user_role == "doctor" else "green"
    role_display = f"Dr. {st.session_state.doctor_id}" if user_role == "doctor" else "Assistant"
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Current Role:** :{role_color}[{role_display}]")
    with col2:
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    with col3:
        has_context = bool(st.session_state.conversation_context.get("patient_context"))
        context_status = "🎯 Active" if has_context else "📝 None"
        st.markdown(f"**Context:** {context_status}")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            render_chat_message(
                message["role"],
                message["content"],
                message.get("timestamp"),
                message.get("metadata")
            )
    
    # Chat input
    if "sample_query" in st.session_state:
        # Use sample query
        user_input = st.session_state.sample_query
        del st.session_state.sample_query
    else:
        # Regular chat input
        placeholder_text = (
            "Ask follow-up questions like 'Tell me about her history' or 'When is my next slot?'..." 
            if user_role == "doctor" 
            else "Ask contextual questions about schedules and availability..."
        )
        user_input = st.chat_input(placeholder_text)
    
    # Process user input
    if user_input:
        # Add user message to chat
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Show user message
        with chat_container:
            render_chat_message("user", user_input, timestamp)
        
        # Send to backend and get response
        with st.spinner("🤔 Processing your request with context awareness..."):
            response_data = send_chat_message(
                user_input,
                st.session_state.user_role,
                st.session_state.doctor_id if st.session_state.user_role == "doctor" else None,
                st.session_state.session_id
            )
        
        # Update session ID if returned
        if response_data.get("session_id"):
            st.session_state.session_id = response_data["session_id"]
        
        # Update conversation context
        if response_data.get("conversation_context"):
            st.session_state.conversation_context = response_data["conversation_context"]
        
        # Add assistant response to chat
        assistant_response = response_data.get("response", "Sorry, I couldn't process your request.")
        response_timestamp = datetime.now().strftime("%H:%M:%S")
        response_metadata = response_data.get("metadata", {})
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": response_timestamp,
            "metadata": response_metadata
        })
        
        # Show assistant response
        with chat_container:
            render_chat_message("assistant", assistant_response, response_timestamp, response_metadata)
        
        # Show success/error status
        if response_data.get("success", True):
            st.success("✅ Request processed with context awareness")
        else:
            st.error("❌ Request failed - check the response for details")
        
        # Rerun to update the chat
        st.rerun()

if __name__ == "__main__":
    main()
