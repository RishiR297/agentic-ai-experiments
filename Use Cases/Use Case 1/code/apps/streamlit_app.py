"""
Streamlit Web Interface for LangGraph Medical Assistant
Provides a user-friendly chat interface for doctors and assistants with detailed diagnostics.
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Explicitly add the `code` directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import robust DB connection utility
from langgraph_agent.tools.database import get_db_connection

# Page configuration
st.set_page_config(
    page_title="🏥 Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration - Updated to match our refactored backend
API_BASE_URL = "http://127.0.0.1:8001"

def get_doctor_mappings():
    """Get doctor ID to name mappings from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Get distinct doctor mappings, prioritizing by recent appointments
        cursor.execute("""
            SELECT DoctorId, DoctorName, COUNT(*) as appointment_count, MAX(StartDateTime) as latest_appointment
            FROM view_appointments 
            GROUP BY DoctorId, DoctorName
            ORDER BY latest_appointment DESC, appointment_count DESC
        """)
        mappings = {}
        for doctor_id, doctor_name, count, latest in cursor.fetchall():
            # Use the first (most recent/active) mapping for each doctor_id
            if doctor_id not in mappings:
                mappings[doctor_id] = {
                    'name': doctor_name,
                    'appointment_count': count,
                    'latest_appointment': latest
                }
        conn.close()
        return mappings
    except Exception as e:
        st.error(f"Error loading doctor mappings: {e}")
        return {11: {'name': 'Antonella', 'appointment_count': 0, 'latest_appointment': None}}

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_role" not in st.session_state:
        st.session_state.user_role = "doctor"
    if "doctor_id" not in st.session_state:
        st.session_state.doctor_id = "11"  # Default to Dr. Antonella (DoctorId 1)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "diagnostics_enabled" not in st.session_state:
        st.session_state.diagnostics_enabled = True
    # Load doctor mappings
    if "doctor_mappings" not in st.session_state:
        st.session_state.doctor_mappings = get_doctor_mappings()

def get_mcp_context(session_id: str) -> Dict[str, Any]:
    """Get MCP context for a session from the MCP server."""
    try:
        response = requests.get(f"http://localhost:8002/context/{session_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "context": {}}
    except Exception as e:
        return {"error": str(e), "context": {}}

def get_mcp_summary() -> Dict[str, Any]:
    """Get overall MCP system summary."""
    try:
        response = requests.get("http://localhost:8002/mcp/summary", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

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
            timeout=90
        )
        
        if response.status_code == 200:
            response_data = response.json()
            return {
                "response": response_data.get("result", "No response"),
                "success": response_data.get("success", False),
                "tool_name": response_data.get("tool_name", "unknown"),
                "metadata": response_data.get("metadata", {}),
                "sql_metadata": response_data.get("sql_metadata", {}),
                "conversation_context": response_data.get("conversation_context", {}),
                "session_id": response_data.get("session_id", "unknown")
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
    """Render a chat message with comprehensive diagnostic information."""
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
            
            # Enhanced diagnostics panel
            if metadata and st.session_state.get("diagnostics_enabled", True):
                with st.expander("🔍 **Response Diagnostics & MCP Context**", expanded=False):
                    
                    # Create tabs for different diagnostic views
                    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Processing Flow", "🧠 MCP Context", "🗄️ SQL Details", "🔧 Technical Metadata"])
                    
                    with tab1:
                        st.subheader("Processing Flow")
                        
                        # Show the processing pipeline
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**🔤 Original Query:**")
                            st.code(metadata.get("original_query", content), language="text")
                            
                            if "intent" in metadata:
                                st.write(f"**🎯 Intent Detected:** `{metadata['intent']}`")
                            
                            if "tool_name" in metadata:
                                st.write(f"**🛠️ Tool Selected:** `{metadata['tool_name']}`")
                        
                        with col2:
                            if "conversation_context" in metadata:
                                ctx = metadata["conversation_context"]
                                if "resolved_references" in ctx:
                                    st.write("**� Reference Resolution:**")
                                    refs = ctx["resolved_references"]
                                    if refs:
                                        for ref, resolution in refs.items():
                                            if isinstance(resolution, str):
                                                st.write(f"  • `{ref}` → {resolution}")
                                            else:
                                                st.write(f"  • `{ref}` → [Complex Object]")
                                    else:
                                        st.write("  • No references to resolve")
                                
                                if "query_intent" in ctx:
                                    st.write(f"**💭 Query Intent:** {ctx['query_intent']}")
                    
                    with tab2:
                        st.subheader("🧠 MCP Context Memory")
                        # Get MCP context for this session
                        session_id = metadata.get("session_id", "streamlit_doctor_1")
                        mcp_data = get_mcp_context(session_id)
                        # Always show raw MCP context JSON for debugging
                        st.write("**Raw MCP Context (JSON):**")
                        st.code(json.dumps(mcp_data, indent=2, default=str), language="json")
                        if "error" not in mcp_data:
                            # Prefer 'mcp_context' if present, else fallback to 'context' for backward compatibility
                            context_items = mcp_data.get("mcp_context")
                            if context_items is None:
                                context_items = mcp_data.get("context", {})
                                # If context is a dict, convert to list of items for uniformity
                                if isinstance(context_items, dict):
                                    context_items = list(context_items.values())
                            if context_items:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Context Items", len(context_items))
                                    # Show context types
                                    context_types = {}
                                    for item in context_items:
                                        item_type = item.get("context_type", "unknown")
                                        context_types[item_type] = context_types.get(item_type, 0) + 1
                                    st.write("**Context Types:**")
                                    for ctx_type, count in context_types.items():
                                        st.write(f"  • {ctx_type}: {count}")
                                with col2:
                                    # Show recent context items (limit to 5 most recent, sorted by queried_at if present)
                                    def get_queried_at(item):
                                        val = item.get("queried_at", "")
                                        return val or ""
                                    sorted_items = sorted(
                                        enumerate(context_items),
                                        key=lambda x: get_queried_at(x[1]),
                                        reverse=True
                                    )[:5]
                                    st.write("**Recent Context (Last 5):**")
                                    for idx, item in sorted_items:
                                        item_type = item.get("context_type", "unknown")
                                        timestamp_str = item.get("queried_at", "")
                                        if timestamp_str:
                                            try:
                                                ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                                time_display = ts.strftime("%H:%M:%S")
                                            except:
                                                time_display = timestamp_str
                                        else:
                                            time_display = "unknown"
                                        st.write(f"  • **{item_type}** ({time_display})")
                                        if item_type == "schedule" and "schedule_data" in item:
                                            patient_count = len(item["schedule_data"])
                                            st.write(f"    └── {patient_count} appointments")
                                # Detailed context view
                                st.write("**📋 Detailed Context Data:**")
                                for i, item in enumerate([x[1] for x in sorted_items]):
                                    st.write(f"**Context {i+1}: {item.get('context_type', 'unknown')}**")
                                    # Show context item details
                                    if item.get("context_type") == "schedule" and "schedule_data" in item:
                                        schedule_data = item["schedule_data"]
                                        st.write(f"📅 **Schedule for {item.get('date', 'unknown date')}:**")
                                        for appointment in schedule_data:
                                            patient_name = appointment.get("PatientName", "Unknown")
                                            start_time = appointment.get("StartDateTime", "")
                                            service = appointment.get("ServiceName", "Unknown")
                                            if start_time:
                                                try:
                                                    time_obj = datetime.fromisoformat(start_time.replace(' ', 'T'))
                                                    time_display = time_obj.strftime("%I:%M %p")
                                                except:
                                                    time_display = start_time
                                            else:
                                                time_display = "Unknown time"
                                            st.write(f"  • {patient_name} at {time_display} for {service}")
                                    else:
                                        # Show raw context data in a code block
                                        st.code(json.dumps(item, indent=2, default=str), language="json")
                                    st.write("")  # Add spacing between items
                            else:
                                st.info("No MCP context available for this session")
                        else:
                            st.error(f"Failed to retrieve MCP context: {mcp_data.get('error', 'Unknown error')}")
                    
                    with tab3:
                        st.subheader("🗄️ SQL Query Details")
                        
                        if "sql_metadata" in metadata and metadata["sql_metadata"]:
                            sql_meta = metadata["sql_metadata"]
                            
                            # LLM Reasoning section
                            if "llm_reasoning" in sql_meta:
                                st.write("**🧠 LLM SQL Generation Reasoning:**")
                                st.info(sql_meta["llm_reasoning"])
                            
                            if "query_type" in sql_meta:
                                st.write(f"**🔍 Query Type:** `{sql_meta['query_type']}`")
                            
                            if "execution_method" in sql_meta:
                                method = sql_meta["execution_method"]
                                method_color = "🤖" if method == "llm_generated_sql" else "🔧"
                                st.write(f"**{method_color} Execution Method:** `{method}`")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if "raw_query" in sql_meta:
                                    st.write("**Generated SQL Query:**")
                                    st.code(sql_meta["raw_query"], language="sql")
                                
                                if "parameters" in sql_meta:
                                    st.write("**Query Parameters:**")
                                    params = sql_meta["parameters"]
                                    if params:
                                        for i, param in enumerate(params):
                                            st.write(f"  • Parameter {i+1}: `{param}`")
                                    else:
                                        st.write("  • No parameters")
                                
                                # Query evaluation context
                                if "query_evaluation" in sql_meta:
                                    eval_data = sql_meta["query_evaluation"]
                                    st.write("**📊 Query Evaluation:**")
                                    st.write(f"  • Intent: `{eval_data.get('intent', 'unknown')}`")
                                    st.write(f"  • Context Used: `{eval_data.get('context_used', False)}`")
                                    if eval_data.get("resolved_references"):
                                        st.write(f"  • References: `{len(eval_data['resolved_references'])} resolved`")
                            
                            with col2:
                                if "result_count" in sql_meta:
                                    st.metric("Results Returned", sql_meta["result_count"])
                                
                                if "execution_time" in sql_meta:
                                    st.metric("Execution Time", f"{sql_meta['execution_time']}ms")
                                
                                if "generated_at" in sql_meta:
                                    gen_time = sql_meta["generated_at"]
                                    st.write(f"**⏰ Generated At:** {gen_time[:19]}")
                                
                                # Parameter mapping details
                                if "parameter_mapping" in sql_meta:
                                    param_map = sql_meta["parameter_mapping"]
                                    if param_map.get("doctor_uuid_mapping"):
                                        st.write("**🔗 Doctor Mapping:**")
                                        st.code(param_map["doctor_uuid_mapping"], language="text")
                                
                                if "tool_name" in sql_meta:
                                    st.write(f"**Tool Used:** {sql_meta['tool_name']}")
                        else:
                            st.info("No SQL query was executed for this response")
                    
                    with tab4:
                        st.subheader("🔧 Technical Metadata")
                        
                        # Show all metadata in a structured way
                        if metadata:
                            # User context
                            if "identity_context" in metadata:
                                id_ctx = metadata["identity_context"]
                                st.write("**👤 User Identity:**")
                                st.write(f"  • Role: {id_ctx.get('role', 'unknown')}")
                                st.write(f"  • Doctor ID: {id_ctx.get('doctor_id', 'N/A')}")
                                st.write(f"  • Session: {id_ctx.get('timestamp', 'N/A')}")
                            
                            # Response metadata
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**📊 Response Metadata:**")
                                for key, value in metadata.items():
                                    if key not in ["conversation_context", "sql_metadata", "identity_context"]:
                                        if isinstance(value, (str, int, float, bool)):
                                            st.write(f"  • {key}: `{value}`")
                            
                            with col2:
                                # Show conversation context
                                if "conversation_context" in metadata:
                                    conv_ctx = metadata["conversation_context"]
                                    st.write("**💬 Conversation Context:**")
                                    for key, value in conv_ctx.items():
                                        if isinstance(value, (str, int, float, bool)):
                                            st.write(f"  • {key}: `{value}`")
                            
                            # Raw metadata (collapsible)
                            st.write("**🔍 Raw Metadata (JSON):**")
                            st.code(json.dumps(metadata, indent=2, default=str), language="json")
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
            # Build doctor options from database mappings
            doctor_mappings = st.session_state.doctor_mappings
            doctor_options = {}
            
            # Add doctors from database
            for doctor_id, info in doctor_mappings.items():
                doctor_name = info['name']
                appointment_count = info['appointment_count']
                latest_appointment = info['latest_appointment']
                
                # Format display name with additional info
                if latest_appointment:
                    try:
                        latest_date = datetime.fromisoformat(latest_appointment.replace(' ', 'T')).strftime("%Y-%m-%d")
                        display_name = f"Dr. {doctor_name} (ID: {doctor_id}, {appointment_count} appts, latest: {latest_date})"
                    except:
                        display_name = f"Dr. {doctor_name} (ID: {doctor_id}, {appointment_count} appointments)"
                else:
                    display_name = f"Dr. {doctor_name} (ID: {doctor_id}, {appointment_count} appointments)"
                
                doctor_options[display_name] = str(doctor_id)
            
            # Add custom option
            doctor_options["Custom Doctor ID"] = "custom"
            
            # Show doctor mappings info
            st.write("**📋 Available Doctors from Database:**")
            for doctor_id, info in doctor_mappings.items():
                st.write(f"  • **{info['name']}** (ID: {doctor_id}) - {info['appointment_count']} appointments")
            
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
            st.info("🏢 **Assistant Access**: Full appointment management for all doctors (booking, rescheduling, cancellation with doctor specification)")
        
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
                st.subheader(f"🛠️ Available Tools ({len(tools)})")
                for tool in tools:
                    st.write(f"• **{tool['name']}**")
                    st.caption(tool.get('description', 'No description'))
        
        st.divider()
        
        # Diagnostics Controls
        st.header("🔍 Diagnostics & MCP")
        
        # Toggle for diagnostics
        diagnostics_enabled = st.checkbox(
            "Enable Response Diagnostics",
            value=st.session_state.get("diagnostics_enabled", True),
            help="Show detailed processing flow, MCP context, and SQL queries for each response"
        )
        st.session_state.diagnostics_enabled = diagnostics_enabled
        
        if diagnostics_enabled:
            st.success("✅ Diagnostics Enabled")
            st.caption("Response details will show: Processing flow, MCP context, and SQL queries, and technical metadata")
        else:
            st.info("ℹ️ Diagnostics Disabled")
        
        # MCP System Summary
        if health_status["mcp"]:
            st.subheader("🧠 MCP System Summary")
            mcp_summary = get_mcp_summary()
            if "error" not in mcp_summary:
                if "active_sessions" in mcp_summary:
                    st.metric("Active Sessions", mcp_summary["active_sessions"])
                if "total_context_items" in mcp_summary:
                    st.metric("Total Context Items", mcp_summary["total_context_items"])
                if "context_types" in mcp_summary:
                    st.write("**Context Types Distribution:**")
                    for ctx_type, count in mcp_summary["context_types"].items():
                        st.write(f"  • {ctx_type}: {count}")
            else:
                st.error(f"MCP Summary Error: {mcp_summary.get('error', 'Unknown')}")
        
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
            
            # Add multi-turn conversation examples
            st.write("**🔄 Multi-Turn Conversation Examples:**")
            st.write("Try these sequences to see MCP context in action:")
            
            st.write("**💬 Example 1: Reference Resolution**")
            st.write("1. 'Who is my next patient?'")
            st.write("2. 'What's her medical history?' ← MCP resolves 'her'")
            st.write("3. 'Reschedule that appointment' ← MCP resolves 'that appointment'")
            
            st.write("**💬 Example 2: Context Building**")
            st.write("1. 'Show me my schedule for today'")
            st.write("2. 'Who's at 2 PM?' ← Uses schedule context")
            st.write("3. 'Move him to tomorrow' ← Resolves patient reference")
                
            st.write("**💬 Example 3: Date Context**")
            st.write("1. 'What appointments do I have tomorrow?'")
            st.write("2. 'Who is my first patient?' ← Uses tomorrow's context")
            st.write("3. 'How long is that appointment?' ← References first patient")
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
    if user_role == "doctor":
        doctor_mappings = st.session_state.doctor_mappings
        doctor_id = st.session_state.doctor_id
        doctor_name = None
        # Debug: print doctor_mappings and doctor_id type
        print(f"[DEBUG] doctor_mappings keys: {list(doctor_mappings.keys())}")
        print(f"[DEBUG] doctor_id value: {doctor_id}, type: {type(doctor_id)}")
        # Try both int and str keys for robustness
        doctor_info = None
        try:
            doctor_id_int = int(doctor_id)
            doctor_info = doctor_mappings.get(doctor_id_int)
        except Exception:
            doctor_info = None
        print(f"[DEBUG] doctor_info for doctor_id {doctor_id}: {doctor_info}")
        if doctor_info and doctor_info.get('name'):
            doctor_name = doctor_info['name']
            role_display = f"Dr. {doctor_name} (ID: {doctor_id})"
        else:
            role_display = f"Dr. Unknown Doctor (ID: {doctor_id})"
    else:
        role_display = "Assistant"
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
        
        # Collect comprehensive metadata for diagnostics
        response_metadata = {
            "original_query": user_input,
            "session_id": response_data.get("session_id", "unknown"),
            "tool_name": response_data.get("tool_name", "unknown"),
            "intent": response_data.get("metadata", {}).get("intent", "unknown"),
            "conversation_context": response_data.get("conversation_context", {}),
            "sql_metadata": response_data.get("sql_metadata", {}),
            "identity_context": response_data.get("identity_context", {}),
            "user_role": response_data.get("user_role", user_role),
            "response_success": response_data.get("success", True),
            "backend_metadata": response_data.get("metadata", {})
        }
        
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
