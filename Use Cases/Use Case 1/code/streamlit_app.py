"""
Streamlit Web Interface for LangGraph Medical Assistant
Provides a user-friendly chat interface for doctors and assistants.
"""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="🏥 Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration - Updated to match our refactored backend
API_BASE_URL = "http://127.0.0.1:8001"

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "user_role" not in st.session_state:
        st.session_state.user_role = "doctor"
    
    if "doctor_id" not in st.session_state:
        st.session_state.doctor_id = "1"  # Default to Dr. Antonella (DoctorId 1)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

def send_chat_message(message: str, user_role: str, doctor_id: str = None) -> Dict[str, Any]:
    """Send a message to the backend API with proper role-based headers."""
    try:
        # Prepare headers based on user role
        headers = {"Content-Type": "application/json"}
        
        if user_role == "doctor" and doctor_id:
            headers["X-Doctor-ID"] = doctor_id
            headers["X-User-Role"] = "doctor"
        else:
            headers["X-User-Role"] = "assistant"
        
        # Prepare payload for the /chat endpoint
        payload = {
            "message": message,
            "session_id": f"streamlit_{user_role}_{doctor_id or 'assistant'}"
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
                "metadata": response_data.get("metadata", {})
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
    """Check the health of backend and MCP systems."""
    health_status = {
        "backend": False,
        "mcp": False,
        "backend_details": {},
        "mcp_details": {}
    }
    
    # Check backend health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_status["backend"] = True
            health_status["backend_details"] = response.json()
    except:
        pass
    
    # Check MCP health
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            health_status["mcp"] = True
            health_status["mcp_details"] = response.json()
    except:
        pass
    
    return health_status

def render_chat_message(role: str, content: str, timestamp: str = None, metadata: Dict = None):
    """Render a chat message with proper styling."""
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
                        if "tools_used" in metadata:
                            st.write(f"**Tools Used:** {metadata['tools_used']}")
                        if "tool_used" in metadata:
                            st.write(f"**Tool Used:** {metadata['tool_used']}")
                    with col2:
                        if "user_role" in metadata:
                            st.write(f"**User Role:** {metadata['user_role']}")
                        if "has_errors" in metadata:
                            error_status = "❌ Yes" if metadata["has_errors"] else "✅ No"
                            st.write(f"**Errors:** {error_status}")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("🏥 LangGraph Medical Assistant")
    st.markdown("*Intelligent multi-agent system for medical scheduling and patient management*")
    
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
                "Dr. Antonella (ID: 1)": "1",
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
                    help="Enter Doctor ID (integer or UUID format)",
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
        
        # System status
        st.header("🔧 System Status")
        health_status = check_system_health()
        
        if health_status["backend"]:
            st.success("✅ Backend API: Online")
            backend_details = health_status["backend_details"]
            if backend_details.get("tools_available"):
                st.caption(f"Tools available: {backend_details['tools_available']}")
        else:
            st.error("❌ Backend API: Offline")
        
        if health_status["mcp"]:
            st.success("✅ MCP Server: Online")
        else:
            st.error("❌ MCP Server: Offline")
        
        # Show available tools for current role
        if health_status["backend"]:
            tools = get_user_tools(user_role, st.session_state.doctor_id if user_role == "doctor" else None)
            if tools:
                with st.expander(f"🛠️ Available Tools ({len(tools)})"):
                    for tool in tools:
                        st.write(f"• **{tool['name']}**")
                        st.caption(tool.get('description', 'No description'))
        
        st.divider()
        
        # Sample queries based on role
        st.header("💡 Sample Queries")
        
        if user_role == "doctor":
            sample_queries = [
                "What are my appointments today?",
                "Who is my next patient?",
                "Summarize my calendar today",
                "When is my earliest available slot?",
                "Show me my schedule for tomorrow"
            ]
        else:
            sample_queries = [
                "Show me Dr. Smith's schedule",
                "Is Dr. Johnson available tomorrow?",
                "List available cardiologists",
                "Check Dr. Patel's availability at 2 PM",
                "Who has openings this week?"
            ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{hash(query)}", use_container_width=True):
                st.session_state.sample_query = query
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.conversation_id = None
            st.rerun()

    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Show current role info
    role_color = "blue" if user_role == "doctor" else "green"
    role_display = f"Dr. Antonella (ID: {st.session_state.doctor_id})" if user_role == "doctor" else "Assistant"
    st.markdown(f"**Current Role:** :{role_color}[{role_display}]")
    
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
        placeholder_text = "Ask about appointments, schedules, or patients..." if user_role == "doctor" else "Ask about doctor schedules and availability..."
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
        
        # Show user message immediately
        with chat_container:
            render_chat_message("user", user_input, timestamp)
        
        # Send to backend and get response
        with st.spinner("🤔 Processing your request..."):
            response_data = send_chat_message(
                user_input,
                st.session_state.user_role,
                st.session_state.doctor_id if st.session_state.user_role == "doctor" else None
            )
        
        # Add assistant response to chat
        assistant_response = response_data.get("response", "Sorry, I couldn't process your request.")
        response_timestamp = datetime.now().strftime("%H:%M:%S")
        response_metadata = response_data.get("metadata", {})
        
        # Add role info to metadata for display
        response_metadata["user_role"] = response_data.get("user_role", user_role)
        
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
            st.success("✅ Request processed successfully")
        else:
            st.error("❌ Request failed - check the response for details")
        
        # Rerun to update the chat
        st.rerun()

if __name__ == "__main__":
    main()
