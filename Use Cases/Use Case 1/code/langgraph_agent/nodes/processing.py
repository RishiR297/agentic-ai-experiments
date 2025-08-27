"""
LangGraph Medical Agent Processing Nodes
========================================

This module contains the core node logic for the LangGraph medical assistant agent.
It implements dynamic slot-filling, LLM-driven reasoning, SQL generation, backend lookups,
context-aware user prompting, and memory/context management. All logic is designed to be
schema-agnostic and robust for production use.

Key responsibilities:
- Tool execution and SQL query handling
- Slot validation and backend lookups
- Context resolution and memory management
- LLM-based response formatting

Author: [Your Team/Name]
"""

# ========== IMPORTS ==========
import json
import logging
import re
from typing import Dict, Any
from datetime import datetime, timedelta

# Core agent imports
from ..core.state import AgentState, update_patient_context, add_to_conversation_memory
from ..core.config import AgentConfig

# LangChain message types
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Database and tools
from ..tools.database import (
    execute_query, get_service_id_and_duration, get_doctor_default_branch
)
from ..tools.mcp_context_manager import mcp_context_manager

# ========== LOGGER SETUP ==========
logger = logging.getLogger(__name__)

# ========== CONSTANTS ==========
BACKEND_FIELDS = ["StatusId", "PatientId", "BranchName", "ServiceId", "BranchId"]
USER_DEPENDENT_FIELDS = {"service_name", "patient_name", "start_time", "appointment_date"}
AUTO_RESOLVE_FIELDS = {"service_id", "branch_name", "appointment_id", "status_id", "end_time"}

# ========== UTILITY FUNCTIONS ==========

def clean_json_response(response_content: str) -> str:
    """
    Clean OpenAI response content to extract valid JSON.
    Handles responses wrapped in ```json code blocks and removes comments.
    """
    # Remove code block markers robustly
    content = response_content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    if content.startswith("```"):
        content = content[len("```"):].strip()
    if content.endswith("```"):
        content = content[:-len("```")].strip()
    
    # Remove single-line comments (// comments)
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove // comments but preserve the rest of the line
        if '//' in line:
            comment_pos = line.find('//')
            before_comment = line[:comment_pos]
            quote_count = before_comment.count('"') - before_comment.count('\\"')
            if quote_count % 2 == 0:
                line = line[:comment_pos].rstrip()
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def safe_json_parse(response_content: str, node_name: str) -> Dict[str, Any]:
    """
    Safely parse JSON response with cleaning and error handling.
    """
    try:
        cleaned_content = clean_json_response(response_content)
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {node_name}: {e}")
        logger.error(f"Response content was: {response_content}")
        
        # Try to extract JSON pattern manually
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, cleaned_content, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except:
                pass
        
        # Return a safe default
        return {
            "error": f"Failed to parse JSON in {node_name}",
            "raw_content": response_content[:200] + "..." if len(response_content) > 200 else response_content
        }


def preprocess_date_references(query: str) -> Dict[str, str]:
    """
    Preprocess date references like 'tomorrow', 'today', 'next week tuesday' etc.
    Returns a mapping of date references to actual dates.
    """
    date_mappings = {}
    current_date = datetime.now()
    
    query_lower = query.lower()

    # Handle today
    if 'today' in query_lower:
        date_mappings['today'] = current_date.strftime('%Y-%m-%d')

    # Handle tomorrow
    if 'tomorrow' in query_lower:
        tomorrow_date = current_date + timedelta(days=1)
        date_mappings['tomorrow'] = tomorrow_date.strftime('%Y-%m-%d')

    # Handle yesterday
    if 'yesterday' in query_lower:
        yesterday_date = current_date - timedelta(days=1)
        date_mappings['yesterday'] = yesterday_date.strftime('%Y-%m-%d')

    # Handle "next week [day]" more intelligently
    if 'next week' in query_lower:
        # Find next week's Monday
        days_until_next_monday = (7 - current_date.weekday()) % 7
        if days_until_next_monday == 0:  # If today is Monday
            days_until_next_monday = 7
        next_monday = current_date + timedelta(days=days_until_next_monday)
        
        # Check for specific days
        if 'tuesday' in query_lower:
            next_week_date = next_monday + timedelta(days=1)  # Monday + 1 = Tuesday
        elif 'wednesday' in query_lower:
            next_week_date = next_monday + timedelta(days=2)
        elif 'thursday' in query_lower:
            next_week_date = next_monday + timedelta(days=3)
        elif 'friday' in query_lower:
            next_week_date = next_monday + timedelta(days=4)
        elif 'saturday' in query_lower:
            next_week_date = next_monday + timedelta(days=5)
        elif 'sunday' in query_lower:
            next_week_date = next_monday + timedelta(days=6)
        elif 'monday' in query_lower:
            next_week_date = next_monday
        else:
            # Default to next Monday if no specific day mentioned
            next_week_date = next_monday
            
        date_mappings['next week'] = next_week_date.strftime('%Y-%m-%d')
        
        # Also map the specific phrase if it exists
        import re
        next_week_pattern = r'next week\s+(\w+)'
        match = re.search(next_week_pattern, query_lower)
        if match:
            full_phrase = f"next week {match.group(1)}"
            date_mappings[full_phrase] = next_week_date.strftime('%Y-%m-%d')

    # Handle this week
    if 'this week' in query_lower:
        # Find the start of this week (Monday)
        days_since_monday = current_date.weekday()
        week_start = current_date - timedelta(days=days_since_monday)
        date_mappings['this week'] = week_start.strftime('%Y-%m-%d')

    return date_mappings


def validate_sql_params(params: dict) -> str | list:
    """
    Validate required SQL parameters. Returns 'ok' if all present, else list of missing user-dependent keys.
    Auto-resolve fields are not considered missing for user prompt.
    """
    required_natural_fields = ["service_name", "patient_name", "appointment_date", "appointment_time"]
    missing = [field for field in required_natural_fields if not params.get(field)]
    return "ok" if not missing else missing

# ========== RBAC EVALUATION NODES ==========
"""
Role-Based Access Control (RBAC) Evaluation Nodes
--------------------------------------------------
These nodes enforce role-based security for appointment management operations.
- Doctors can only manage their own appointments (reschedule/cancel)
- Assistants can manage any doctor's appointments with proper doctor specification
- All users can perform general queries and book new appointments
"""

def rbac_evaluation_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Evaluate role-based access control for appointment management operations.
    
    RBAC Rules:
    - Doctors: Can only reschedule/cancel their own appointments
    - Assistants: Can reschedule/cancel any doctor's appointments (must specify target doctor)
    - All roles: Can perform queries, book new appointments, view schedules
    
    Blocks unauthorized operations and provides clear error messages.
    """
    try:
        # Get the current intended action and user context
        user_role = state.get("user_role", "").lower()
        intended_tool = state.get("selected_tool", "")
        sql_params = state.get("sql_params", {})
        
        # Management operations that require RBAC evaluation
        restricted_operations = ["appointment_rescheduling", "appointment_cancellation", "cancel_appointment"]
        
        # If not a restricted operation, allow through
        if intended_tool not in restricted_operations:
            state["rbac_evaluation"] = "approved"
            state["rbac_message"] = f"Operation '{intended_tool}' approved for {user_role}"
            return state
        
        # Extract user and target doctor information
        # For streamlit sessions, extract doctor_id from session_id
        session_id = state.get("session_id", "")
        current_doctor_id = state.get("doctor_id", "")
        target_doctor_id = sql_params.get("doctor_id", "")
        
        # Handle streamlit sessions where role is parsed as "streamlit"
        if user_role == "streamlit":
            if "doctor" in session_id:
                user_role = "doctor"
                # Extract doctor_id from session like "streamlit_doctor_14"
                if not current_doctor_id and "_" in session_id:
                    parts = session_id.split("_")
                    if len(parts) >= 3 and parts[1] == "doctor":
                        current_doctor_id = parts[2]
            elif "assistant" in session_id:
                user_role = "assistant"
                # Assistants don't have their own doctor_id
        
        # Validate required information for doctors only
        # Assistants don't need current_doctor_id since they work on behalf of others
        if user_role == "doctor" and not current_doctor_id:
            state["rbac_evaluation"] = "denied"
            state["rbac_message"] = "Access denied: Doctor identity not established"
            state["rbac_error"] = "Authentication required for appointment management operations"
            return state
        
        if not user_role:
            state["rbac_evaluation"] = "denied"
            state["rbac_message"] = "Access denied: User role not specified"
            state["rbac_error"] = "Role verification required for appointment management"
            return state
        
        # RBAC Logic
        if user_role == "doctor":
            # Doctors can only manage their own appointments
            if not target_doctor_id:
                # If no target doctor specified, assume they mean themselves
                target_doctor_id = current_doctor_id
                sql_params["doctor_id"] = current_doctor_id
                state["sql_params"] = sql_params
            
            if str(target_doctor_id) != str(current_doctor_id):
                state["rbac_evaluation"] = "denied"
                state["rbac_message"] = f"Access denied: Doctor {current_doctor_id} cannot manage Doctor {target_doctor_id}'s appointments"
                state["rbac_error"] = f"Doctors can only reschedule or cancel their own appointments. You can only manage appointments for Doctor ID {current_doctor_id}."
                return state
            
            # Doctor accessing their own appointments - approved
            state["rbac_evaluation"] = "approved"
            state["rbac_message"] = f"Doctor {current_doctor_id} authorized to manage their own appointments"
            
        elif user_role == "assistant":
            # Assistants can manage any doctor's appointments but must specify target doctor
            if not target_doctor_id:
                state["rbac_evaluation"] = "denied"
                state["rbac_message"] = "Access denied: Target doctor not specified for assistant operation"
                state["rbac_error"] = "As an assistant, you must specify which doctor's appointment you want to reschedule or cancel."
                return state
            
            # Assistant with specified doctor - approved
            state["rbac_evaluation"] = "approved"
            state["rbac_message"] = f"Assistant authorized to manage Doctor {target_doctor_id}'s appointments"
            
        else:
            # Unknown role - deny access
            state["rbac_evaluation"] = "denied" 
            state["rbac_message"] = f"Access denied: Unknown role '{user_role}'"
            state["rbac_error"] = f"Role '{user_role}' is not authorized for appointment management operations"
            return state
        
        print(f"[RBAC] {state['rbac_message']}")
        return state
        
    except Exception as e:
        print(f"[RBAC ERROR] {str(e)}")
        state["rbac_evaluation"] = "error"
        state["rbac_message"] = f"RBAC evaluation failed: {str(e)}"
        state["rbac_error"] = "Security evaluation failed. Please try again."
        return state

# ========== TOOL EXECUTION NODES ==========
"""
Tool Execution Nodes
--------------------
These nodes handle the actual execution of selected tools (database queries, bookings, etc.)
and update the state with results. They handle different tool types like appointment_booking,
schedule_query, appointment_lookup, and patient_lookup.
"""

def tool_executor_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Executes the selected tool (e.g., DB query, booking) and updates the state with results.
    """
    logger.info(f"Tool executor running for tool: {state.get('selected_tool')}")
    
    # RBAC Security Check - Automatically evaluate and enforce access control
    selected_tool = state.get("selected_tool", "")
    restricted_operations = ["appointment_rescheduling", "appointment_cancellation", "cancel_appointment"]
    
    if selected_tool in restricted_operations:
        # Automatically perform RBAC evaluation if not already done
        rbac_status = state.get("rbac_evaluation")
        
        if rbac_status != "approved":
            # Perform automatic RBAC evaluation
            user_role = state.get("user_role", "").lower()
            session_id = state.get("session_id", "")
            current_doctor_id = state.get("doctor_id", "")
            tool_parameters = state.get("tool_parameters", {})
            target_doctor_id = tool_parameters.get("doctor_id", "")
            
            # Handle streamlit sessions where role is parsed as "streamlit"
            if user_role == "streamlit" and "doctor" in session_id:
                user_role = "doctor"
                # Extract doctor_id from session like "streamlit_doctor_14"
                if not current_doctor_id and "_" in session_id:
                    parts = session_id.split("_")
                    if len(parts) >= 3 and parts[1] == "doctor":
                        current_doctor_id = parts[2]
            
            # RBAC Logic - Automatic and seamless
            if user_role == "doctor":
                # Doctors can only manage their own appointments
                if not target_doctor_id:
                    target_doctor_id = current_doctor_id
                    tool_parameters["doctor_id"] = current_doctor_id
                    state["tool_parameters"] = tool_parameters
                
                if str(target_doctor_id) == str(current_doctor_id):
                    # Doctor managing their own appointments - approved
                    state["rbac_evaluation"] = "approved"
                    logger.info(f"[RBAC AUTO] Doctor {current_doctor_id} authorized for own appointments")
                else:
                    # Doctor trying to manage other doctor's appointments - denied
                    error_msg = f"Access denied: You can only manage your own appointments (Doctor ID {current_doctor_id})"
                    logger.warning(f"[RBAC AUTO BLOCKED] Doctor {current_doctor_id} tried to access Doctor {target_doctor_id}'s appointments")
                    state.setdefault("errors", []).append(error_msg)
                    state["tool_results"] = [{"error": error_msg}]
                    state["has_errors"] = True
                    state["formatted_response"] = error_msg
                    return state
                    
            elif user_role == "assistant":
                # Assistants can manage any doctor's appointments if target doctor is specified
                if target_doctor_id:
                    state["rbac_evaluation"] = "approved"
                    logger.info(f"[RBAC AUTO] Assistant authorized for Doctor {target_doctor_id}'s appointments")
                else:
                    error_msg = "As an assistant, you must specify which doctor's appointment you want to manage"
                    logger.warning(f"[RBAC AUTO BLOCKED] Assistant tried to manage appointments without specifying doctor")
                    state.setdefault("errors", []).append(error_msg)
                    state["tool_results"] = [{"error": error_msg}]
                    state["has_errors"] = True
                    state["formatted_response"] = error_msg
                    return state
            else:
                # Unknown role - denied
                error_msg = f"Access denied: Role '{user_role}' is not authorized for appointment management"
                logger.warning(f"[RBAC AUTO BLOCKED] Unknown role '{user_role}' tried to manage appointments")
                state.setdefault("errors", []).append(error_msg)
                state["tool_results"] = [{"error": error_msg}]
                state["has_errors"] = True
                state["formatted_response"] = error_msg
                return state
    
    tool = state.get("selected_tool")
    logger.info(f"[DEBUG] tool_executor_node START: sql_query_parameters = {state.get('sql_query_parameters')} (type: {type(state.get('sql_query_parameters'))})")
    if state.get('sql_query_parameters') is None:
        logger.warning('[DEBUG] sql_query_parameters is None at tool_executor_node entry!')
    elif not isinstance(state.get('sql_query_parameters'), (list, tuple)):
        logger.warning(f'[DEBUG] sql_query_parameters is not a list/tuple: {state.get('sql_query_parameters')}')
    
    # For tools that don't use SQL, skip parameter validation
    tool = state.get("selected_tool")
    backend_complete_tools = ["schedule_query", "appointment_rescheduling", "appointment_cancellation", "appointment_query_executor"]
    if tool in backend_complete_tools:
        params = []  # No SQL parameters needed for backend lookup tools
    else:
        # Always use SQL generator's parameter list for positional SQL queries
        params = state.get("sql_query_parameters", [])
        if not isinstance(params, (list, tuple)):
            logger.error("sql_query_parameters must be a list or tuple for positional SQL placeholders.")
            state.setdefault("errors", []).append("Internal error: SQL parameters are not in list/tuple format.")
            state["tool_results"] = [{"error": "Internal error: SQL parameters are not in list/tuple format."}]
            state["has_errors"] = True
            state["formatted_response"] = "Internal error: SQL parameters are not in list/tuple format."
            return state
    results = []
    error = None
    
    # Guard: Ensure sql_query is present (except for tools that don't use SQL)
    tool = state.get("selected_tool")
    backend_complete_tools = ["schedule_query", "appointment_rescheduling", "appointment_cancellation", "appointment_query_executor"]
    if tool not in backend_complete_tools and not state.get("sql_query"):
        logger.error("No SQL query found in state. Did you forget to run the SQL generator node?")
        error_msg = "No SQL query found in state. Did you forget to run the SQL generator node?"
        state.setdefault("errors", []).append(error_msg)
        state["tool_results"] = [{"error": error_msg}]
        state["has_errors"] = True
        state["formatted_response"] = error_msg
        if "response_metadata" not in state:
            state["response_metadata"] = {}
        if "errors" not in state["response_metadata"]:
            state["response_metadata"]["errors"] = []
        state["response_metadata"]["errors"].append(error_msg)
        return state
    try:
        if tool == "appointment_booking":
            results = execute_query(state["sql_query"], params)
        elif tool == "schedule_query":
            # Enhanced schedule_query execution with intelligent slot handling
            from ..tools.database import schedule_query
            doctor_id = state.get("tool_parameters", {}).get("doctor_id")
            date = state.get("tool_parameters", {}).get("date")
            service_name = state.get("tool_parameters", {}).get("service_name")  # Optional
            suggest_slots = state.get("tool_parameters", {}).get("suggest_slots", False)
            find_earliest = state.get("tool_parameters", {}).get("find_earliest", False)
            
            # Parse user query for specific slot count requests
            original_query = state.get("current_query", "").lower()
            slot_count_limit = None
            
            # Check slot_preference first (from context resolver)
            slot_preference = state.get("resolved_references", {}).get("slot_preference")
            if slot_preference and slot_preference.startswith("next_"):
                try:
                    slot_count_limit = int(slot_preference.split("_")[1])
                    logger.info(f"Slot count from slot_preference: {slot_count_limit}")
                except:
                    pass
            elif slot_preference == "next_1" or slot_preference == "earliest":
                slot_count_limit = 1
                logger.info("Single slot request from slot_preference")
            elif slot_preference == "suggest_limited":
                slot_count_limit = 5  # Default limited suggestion count
                logger.info("Limited suggestions from slot_preference")
            
            # Fallback: Extract number from queries like "next 3 slots", "first 5 available", etc.
            if slot_count_limit is None:
                import re
                numbers = re.findall(r'\b(\d+)\s*(?:slots?|available|times?|appointments?)\b', original_query)
                if numbers:
                    try:
                        slot_count_limit = int(numbers[0])
                        logger.info(f"Detected slot count limit from query: {slot_count_limit}")
                    except:
                        pass
                
                # Special handling for "earliest" or "next" (singular) requests
                if any(word in original_query for word in ['earliest slot', 'next slot', 'first slot']):
                    slot_count_limit = 1
                    logger.info("Detected single slot request from query")
            
            if doctor_id and date:
                # Use the specialized schedule_query function when we have both doctor_id and date
                # The schedule_query function only accepts doctor_id, date, and include_availability
                include_availability = suggest_slots or find_earliest or True  # Default to True for availability
                result = schedule_query.func(doctor_id, date, include_availability)
                
                # Apply slot count limit if detected
                if slot_count_limit and result.get("success") and result.get("available_slots"):
                    original_slots = result["available_slots"]
                    limited_slots = original_slots[:slot_count_limit]
                    result["available_slots"] = limited_slots
                    result["total_available"] = len(limited_slots)
                    result["slot_limit_applied"] = slot_count_limit
                    result["total_slots_found"] = len(original_slots)
                    logger.info(f"Applied slot limit: showing {len(limited_slots)} of {len(original_slots)} slots")
                
                results = [result]  # Wrap in list for consistency
                
                # Handle earliest slot auto-booking if context is present
                booking_context = state.get("earliest_slot_booking_context")
                if (find_earliest and result.get("success") and result.get("earliest_slot") and 
                    booking_context and booking_context.get("original_intent") == "book_appointment"):
                    
                    logger.info("Earliest slot found, preparing for automatic booking")
                    earliest_slot = result["earliest_slot"]
                    
                    # Prepare booking parameters using the earliest slot
                    booking_params = {
                        "patient_name": booking_context.get("patient_name"),
                        "doctor_id": doctor_id,
                        "service_name": booking_context.get("service_name") or service_name,
                        "appointment_date": date,
                        "appointment_time": earliest_slot.get("start_time"),
                        "StartDateTime": earliest_slot.get("start")  # Full datetime
                    }
                    
                    # Store booking info for next processing phase
                    state["auto_booking_ready"] = True
                    state["auto_booking_params"] = booking_params
                    state["earliest_slot_result"] = earliest_slot
                    
                    # Enhanced result with booking preparation info
                    result["auto_booking_prepared"] = True
                    result["booking_params"] = booking_params
                    
                    logger.info(f"Auto-booking prepared for {booking_params['patient_name']} at {earliest_slot.get('start')}")
                
            elif state.get("sql_query"):
                # If we have a generated SQL query, use it directly (for cases like "next patient")
                logger.info("Using generated SQL query for schedule_query since parameters are incomplete")
                sql_params = state.get("sql_query_parameters", [])
                results = execute_query(state["sql_query"], sql_params)
            else:
                error = "Missing doctor_id or date for schedule_query"
        elif tool == "appointment_lookup":
            # Use enhanced appointment_query_executor for time-aware results
            from ..tools.database import appointment_query_executor
            
            # Extract parameters for appointment_query_executor
            tool_params = state.get("tool_parameters", {})
            doctor_id = tool_params.get("doctor_id")
            date = tool_params.get("date") or tool_params.get("appointment_date")
            patient_name = tool_params.get("patient") or tool_params.get("patient_name")
            query_type = "daily_schedule"  # Default for appointment lookups
            
            if doctor_id and date:
                # Use the enhanced function for time-aware categorization
                # Call the tool function directly using .func to bypass LangChain wrapper
                result = appointment_query_executor.func(int(doctor_id), query_type, date, patient_name)
                if result.get("success"):
                    # For time-categorized results, use the structured data
                    if result.get("time_categorized"):
                        results = [result]  # Pass the full structured result
                    else:
                        results = result.get("results", [])
                else:
                    error = result.get("error", "Unknown error in appointment query")
                    results = []
            else:
                # Fallback to raw SQL if parameters are missing
                results = execute_query(state["sql_query"], params)
        elif tool == "patient_lookup":
            results = execute_query(state["sql_query"], params)
        elif tool == "appointment_query_executor":
            # Handle next_patient and other specific appointment queries
            from ..tools.database import appointment_query_executor
            
            # Extract parameters from tool_parameters or use defaults
            tool_params = state.get("tool_parameters", {})
            doctor_id = tool_params.get("doctor_id", state.get("doctor_id"))
            query_type = tool_params.get("query_type", "next_patient")  # Default to next_patient
            date = tool_params.get("date") or tool_params.get("appointment_date")
            patient_name = tool_params.get("patient") or tool_params.get("patient_name")
            
            if doctor_id:
                # Call the appointment_query_executor tool function directly
                result = appointment_query_executor.func(int(doctor_id), query_type, date, patient_name)
                if result.get("success"):
                    results = result.get("results", [])
                else:
                    error = result.get("error", "Unknown error in appointment query executor")
                    results = []
            else:
                error = "Missing doctor_id for appointment_query_executor"
        elif tool == "appointment_rescheduling":
            # Handle appointment rescheduling with RBAC-approved access
            from ..tools.database import appointment_rescheduling
            
            # Extract parameters from tool_parameters (from backend lookup) or sql_params (from SQL generation)
            tool_params = state.get("tool_parameters", {})
            sql_params = state.get("sql_params", {})
            
            # Prefer tool_parameters for backend-complete flows
            appointment_id = tool_params.get("appointment_id") or sql_params.get("appointment_id")
            new_date = tool_params.get("appointment_date") or sql_params.get("new_date") or sql_params.get("appointment_date")
            new_time = tool_params.get("appointment_time") or sql_params.get("new_time") or sql_params.get("appointment_time")
            
            # Validate required parameters
            if not appointment_id:
                error = "Missing appointment ID for rescheduling"
            elif not new_date:
                error = "Missing new date for appointment rescheduling"
            elif not new_time:
                error = "Missing new time for appointment rescheduling"
            else:
                try:
                    # Call the appointment rescheduling function with proper LangChain format
                    reschedule_params = {
                        'appointment_id': appointment_id,
                        'new_date': new_date,
                        'new_time': new_time
                    }
                    result = appointment_rescheduling.invoke(reschedule_params)
                    if result.get("success"):
                        results = [result]  # Wrap result for consistency
                        logger.info(f"[RBAC APPROVED] Appointment {appointment_id} rescheduled successfully")
                    else:
                        error = result.get("error", "Failed to reschedule appointment")
                except Exception as e:
                    error = f"Rescheduling error: {str(e)}"
                    logger.error(f"Appointment rescheduling failed: {e}")
        
        elif tool == "appointment_cancellation" or tool == "cancel_appointment":
            # Handle appointment cancellation with RBAC-approved access
            from ..tools.database import appointment_cancellation
            
            # Extract parameters from tool_parameters (from backend lookup) or sql_params (from SQL generation)
            tool_params = state.get("tool_parameters", {})
            sql_params = state.get("sql_params", {})
            
            # Prefer tool_parameters for backend-complete flows
            appointment_id = tool_params.get("appointment_id") or sql_params.get("appointment_id")
            cancellation_reason = tool_params.get("reason", "Cancelled by user request") or sql_params.get("reason", "Cancelled by user request")
            
            # Validate required parameters
            if not appointment_id:
                error = "Missing appointment ID for cancellation"
            else:
                try:
                    # Call the appointment cancellation function with proper LangChain format
                    cancel_params = {
                        'appointment_id': appointment_id,
                        'reason': cancellation_reason
                    }
                    result = appointment_cancellation.invoke(cancel_params)
                    if result.get("success"):
                        results = [result]  # Wrap result for consistency
                        logger.info(f"[RBAC APPROVED] Appointment {appointment_id} cancelled successfully")
                    else:
                        error = result.get("error", "Failed to cancel appointment")
                except Exception as e:
                    error = f"Cancellation error: {str(e)}"
                    logger.error(f"Appointment cancellation failed: {e}")
        else:
            error = f"Unknown tool: {tool}"
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        error = str(e)
    # Check for DB errors (e.g., missing table) or empty results
    if error:
        state.setdefault("errors", []).append(error)
        state["tool_results"] = [{"error": error}]
        state["has_errors"] = True
        state["formatted_response"] = f"SQL execution failed: {error}"
        if "response_metadata" not in state:
            state["response_metadata"] = {}
        if "errors" not in state["response_metadata"]:
            state["response_metadata"]["errors"] = []
        state["response_metadata"]["errors"].append(error)
    elif isinstance(results, list) and results and any(isinstance(r, dict) and r.get("error") for r in results):
        # If results contain error dicts
        error_msgs = [r["error"] for r in results if isinstance(r, dict) and r.get("error")]
        state.setdefault("errors", []).extend(error_msgs)
        state["tool_results"] = results
        state["has_errors"] = True
        state["formatted_response"] = f"SQL execution failed: {'; '.join(error_msgs)}"
        if "response_metadata" not in state:
            state["response_metadata"] = {}
        if "errors" not in state["response_metadata"]:
            state["response_metadata"]["errors"] = []
        state["response_metadata"]["errors"].extend(error_msgs)
    else:
        # --- Rich tool_results for LLM response (especially after booking) ---
        if tool == "appointment_booking":
            # Try to fetch the just-booked appointment for confirmation
            doctor_id = state.get("tool_parameters", {}).get("doctor_id")
            patient_id = state.get("tool_parameters", {}).get("PatientId")
            if doctor_id and patient_id:
                confirm_query = """
                SELECT * FROM View_Appointments
                WHERE DoctorId = ? AND PatientId = ?
                ORDER BY StartDateTime DESC LIMIT 1
                """
                confirm_results = execute_query(confirm_query, (doctor_id, patient_id))
                if confirm_results:
                    state["tool_results"] = confirm_results
                else:
                    state["tool_results"] = results
            else:
                state["tool_results"] = results
        else:
            state["tool_results"] = results
        state["has_errors"] = False
    return state

# ========== SQL GENERATION NODES ==========
"""
SQL Generation Nodes
--------------------
These nodes use LLM to generate SQL queries based on the selected tool and parameters.
They handle query generation, parameter binding, and SQL metadata logging.
"""

def sql_generator_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Generates SQL query using LLM and executes it, logging all relevant metadata.
    """
    logger.info("SQL generator node invoked")
    # Defensive: Always ensure at least one output field is written, even on error
    # If required context is missing, set error and write to sql_metadata
    if not state.get("selected_tool"):
        logger.error("No selected_tool in state for SQL generation.")
        state.setdefault("errors", []).append("No selected_tool in state for SQL generation.")
        state["sql_metadata"] = {"error": "No selected_tool in state for SQL generation."}
        state["has_errors"] = True
        return state
    if not state.get("tool_parameters"):
        logger.error("No tool_parameters in state for SQL generation.")
        state.setdefault("errors", []).append("No tool_parameters in state for SQL generation.")
        state["sql_metadata"] = {"error": "No tool_parameters in state for SQL generation."}
        state["has_errors"] = True
        return state
    generation_context = {
        "tool_parameters": state.get("tool_parameters", {}),
        "resolved_references": state.get("resolved_references", {}),
        "query_intent": state.get("query_intent"),
        "user_role": state.get("user_role"),
        "doctor_id": state.get("doctor_id"),
        "patient_context": state.get("patient_context"),
        "current_query": state.get("current_query"),
        "current_datetime": datetime.now().isoformat(),
    }
    logger.info(f"SQL generator input tool: {state.get('selected_tool')}")
    logger.info(f"SQL generator input tool_parameters: {state.get('tool_parameters')}")
    logger.info(f"SQL generator input doctor_id: {state.get('doctor_id')}")
    logger.info(f"SQL generator input user_role: {state.get('user_role')}")
    doctor_uuid = state.get("doctor_uuid")
    doctor_id_mapped = state.get("doctor_id")
    system_prompt = (
        "You are a helpful medical assistant.\n"
        "Your ONLY task is to generate a valid SQL query for booking, lookup, or schedule actions, based on the provided context.\n"
        "IMPORTANT: For any appointment-related query (booking, lookup, schedule), you MUST use the table name 'View_Appointments'. Do NOT use 'appointments' or any other table name.\n"
        "Here is the schema for 'View_Appointments':\n"
        "CREATE TABLE View_Appointments (\n"
        "  AppointmentId INTEGER PRIMARY KEY,\n"
        "  PatientId INTEGER,\n"
        "  PatientName TEXT,\n"
        "  DoctorId INTEGER,\n"
        "  DoctorName TEXT,\n"
        "  ServiceId INTEGER,\n"
        "  ServiceName TEXT,\n"
        "  BranchName TEXT,\n"
        "  BranchId INTEGER,\n"
        "  StatusId INTEGER,\n"
        "  Status TEXT,\n"
        "  StartDateTime TEXT,\n"
        "  EndDateTime TEXT\n"
        ");\n"
        "For appointment_lookup queries:\n"
        "- If lookup_type is 'next_patient', find the next upcoming appointment for the doctor (DoctorId) after current time\n"
        "- If lookup_type is 'patient_history', find ALL appointments for the specific patient with the specific doctor (both PatientId and DoctorId filters)\n"
        "- Use ORDER BY StartDateTime ASC and LIMIT 1 for next patient\n"
        "- Use ORDER BY StartDateTime DESC for patient_history to show most recent first\n"
        "- For daily_schedule, return all appointments for the doctor on a specific date\n"
        "- For specific_appointment, use PatientId or PatientName if provided\n"
        "For booking queries, you MUST include PatientId, BranchName, and BranchId in the SQL and parameters.\n"
        "You MUST use the PatientId, DoctorId, ServiceId, BranchId, and StatusId values provided in the context. Do NOT infer or generate IDs; use only those passed in the context.\n"
        "All datetime values (StartDateTime, EndDateTime) MUST use the format 'YYYY-MM-DD HH:MM:SS' (space between date and time, not 'T').\n"
        "You MUST generate parameterized SQL queries using '?' placeholders for all values, and provide a separate array of parameters in the correct order.\n"
        "Do NOT inline values directly in the SQL.\n"
        "Respond ONLY with a valid JSON object in the following format.\n"
        "STRICT FORMATTING RULES:\n"
        "- Use double quotes (\"\") for ALL property names and string values.\n"
        "- Do NOT use single quotes anywhere.\n"
        "- Do NOT include any comments, markdown, code blocks, or explanation.\n"
        "- Do NOT include any preamble or text outside the JSON.\n"
        "- The output MUST be valid JSON and parseable by standard JSON parsers.\n"
        "- If you cannot generate a valid SQL, respond ONLY with an empty JSON object: {}\n"
        "Output format:\n"
        "{\n"
        "  \"sql_query\": \"...\",\n"
        "  \"query_parameters\": [...],\n"
        "  \"reasoning\": \"...\",\n"
        "  \"query_type\": \"...\"\n"
        "}"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate an SQL query for the following context: {json.dumps(generation_context, indent=2, default=str)}")
    ]
    response = config.llm.invoke(messages)
    logger.info(f"OpenAI response content: '{response.content}'")
    if not response.content or response.content.strip() == "":
        logger.error("Empty response from OpenAI")
        state.setdefault("errors", []).append("Empty response from OpenAI in SQL generator node.")
        state["sql_metadata"] = {"error": "Empty response from OpenAI in SQL generator node."}
        state["has_errors"] = True
        return state
    try:
        result = safe_json_parse(response.content, "SQL generator")
    except Exception as je:
        logger.error(f"JSON decode error: {je}")
        content = response.content.strip()
        if "SELECT" in content.upper():
            lines = content.split('\n')
            sql_line = None
            for line in lines:
                if "SELECT" in line.upper():
                    sql_line = line.strip()
                    break
            if sql_line:
                logger.info(f"Extracted SQL from non-JSON response: {sql_line}")
                result = {
                    "sql_query": sql_line,
                    "query_parameters": [],
                    "reasoning": "Extracted from non-JSON response",
                    "query_type": "extracted"
                }
            else:
                state.setdefault("errors", []).append(f"JSON decode error: {je}")
                state["sql_metadata"] = {"error": f"JSON decode error: {je}"}
                state["has_errors"] = True
                return state
        else:
            state.setdefault("errors", []).append(f"JSON decode error: {je}")
            state["sql_metadata"] = {"error": f"JSON decode error: {je}"}
            state["has_errors"] = True
            return state
    state["sql_query"] = result.get("sql_query")
    query_params = result.get("query_parameters", [])
    state["sql_query_parameters"] = query_params
    # Defensive: If no sql_query was produced, set error and write to sql_metadata
    if not state["sql_query"]:
        logger.error("No sql_query produced by LLM in SQL generator node.")
        state.setdefault("errors", []).append("No sql_query produced by LLM in SQL generator node.")
        state["sql_metadata"] = {"error": "No sql_query produced by LLM in SQL generator node."}
        state["has_errors"] = True
        return state
    logger.info(f"[DEBUG] sql_generator_node END: sql_query_parameters = {state['sql_query_parameters']}")
    query_type = result.get("query_type", "unknown")
    reasoning = result.get("reasoning", "No reasoning provided")
    logger.info(f"SQL generator produced query: {state['sql_query']}")
    logger.info(f"SQL generator produced parameters: {query_params}")
    logger.info(f"SQL generator query_type: {query_type}")
    # Note: tool_results will be set by tool_executor_node after actual execution
    # Don't set tool_results here as this is just SQL generation, not execution
    print("=" * 80)
    print("LLM-GENERATED SQL EVALUATION - LangGraph Medical Agent")
    print("=" * 80)
    print(f"Tool: {state.get('selected_tool', 'unknown')}")
    print(f"Query Intent: {state.get('query_intent', 'unknown')}")
    print(f"User Role: {state.get('user_role', 'unknown')}")
    print(f"Doctor ID: {state.get('doctor_id', 'N/A')} -> {doctor_id_mapped}")
    print(f"Session: {state.get('session_id', 'N/A')}")
    print(f"Original Query: {state.get('current_query', 'N/A')}")
    print(f"LLM Reasoning: {reasoning}")
    print(f"Query Type: {query_type}")
    print(f"Generated SQL: {state['sql_query']}")
    print(f"Parameters: {query_params}")
    print(f"Result Count: SQL Generated - Execution Pending")
    print(f"Context References: {state.get('resolved_references', {})}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    # Log SQL metadata (fixed indentation and block structure)
    state.setdefault("sql_metadata", {})["query_info"] = {
        "original_params": query_params,
        "mapped_params": query_params,
        "doctor_uuid_mapping": f"{doctor_uuid} -> {doctor_id_mapped}" if doctor_uuid and doctor_id_mapped else None,
        "result_count": "pending_execution",
        "generated_at": datetime.now().isoformat(),
        "tool_name": state.get("selected_tool", "unknown"),
        "query_type": query_type,
        "llm_reasoning": reasoning,
        "execution_method": "llm_generated_sql",
        "user_context": {
            "role": state.get("user_role", "unknown"),
            "doctor_id": state.get("doctor_id"),
            "session_id": state.get("session_id")
        },
        "query_intent": state["query_intent"],
        "original_query": state["current_query"],
        "tool_results": "pending_execution",
        "user_role": state["user_role"],
        "patient_context": state.get("patient_context"),
    }
    return state

# ========== SLOT VALIDATION NODES ==========
"""
Slot Validation Nodes
---------------------
These nodes validate required parameters for tools, handling both user-facing fields
(that require user input) and backend fields (that are auto-resolved). They use LLM
to determine missing fields and generate clarification prompts.
"""

def slot_validator_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Slot Validator Node (Merged)
    ---------------------------
    1. Deterministically checks for missing user-facing fields (no LLM).
    2. If missing, sets slot_validation and returns (LLM prompt node should handle clarification prompt).
    3. If all present, uses LLM to parse/normalize user input (e.g., date/time parsing).
    4. Merges normalized fields, separates backend/internal fields, sets required_lookups, slot_validation, clarification_prompt, formatted_response.
    """
    # --- Robust Slot Validation Logic ---
    BACKEND_FIELDS = ["StatusId", "PatientId", "BranchName", "ServiceId", "BranchId"]
    tool = state.get("selected_tool")
    params = state.get("tool_parameters", {})
    resolved_references = state.get("resolved_references", {})
    resolved = {**params, **resolved_references}
    
    # AUTOMATIC CONTEXT POPULATION: Auto-populate doctor_id from session if missing
    session_id = state.get("session_id", "")
    user_role = state.get("user_role", "").lower()
    current_doctor_id = state.get("doctor_id", "")
    
    # Handle streamlit sessions where role is parsed as "streamlit"
    if user_role == "streamlit" and "doctor" in session_id:
        user_role = "doctor"
        # Extract doctor_id from session like "streamlit_doctor_14"
        if not current_doctor_id and "_" in session_id:
            parts = session_id.split("_")
            if len(parts) >= 3 and parts[1] == "doctor":
                current_doctor_id = parts[2]
                state["doctor_id"] = current_doctor_id
    
    # Auto-populate doctor_id in tool parameters if missing and user is a doctor
    if user_role == "doctor" and current_doctor_id and not params.get("doctor_id"):
        params["doctor_id"] = current_doctor_id
        state["tool_parameters"] = params
        logger.info(f"[SlotValidator] Auto-populated doctor_id: {current_doctor_id}")
    
    # Use a single source of truth for required user-facing fields
    # --- LLM-Driven Slot Validation Only ---
    BACKEND_FIELDS = ["StatusId", "PatientId", "BranchName", "ServiceId", "BranchId"]
    tool = state.get("selected_tool")
    params = state.get("tool_parameters", {})
    resolved_references = state.get("resolved_references", {})
    resolved = {**params, **resolved_references}
    user_fields = config.get_tool_user_fields(tool)

    logger.info(f"[SlotValidator] Tool: {tool}")
    logger.info(f"[SlotValidator] Incoming params: {params}")
    logger.info(f"[SlotValidator] Resolved references: {resolved_references}")
    logger.info(f"[SlotValidator] Required user fields: {user_fields}")

    # CRITICAL: Handle earliest slot queries directly - no fields required
    slot_preference = resolved_references.get("slot_preference")
    query_intent = state.get("query_intent")
    
    if (slot_preference == "earliest" and query_intent == "slot_suggestion" and 
        tool == "schedule_query" and state.get("user_role") == "doctor"):
        logger.info("[SlotValidator] Earliest slot query detected - no user fields required")
        # For earliest slot queries, no additional fields needed
        state["slot_validation"] = {"status": "missing_backend", "fields": ["schedule_query_params"]}
        state["clarification_prompt"] = ""
        state["formatted_response"] = ""
        state["required_lookups"] = ["doctor_id", "date"]
        state["has_errors"] = False
        return state

    # 1. Always use LLM to determine missing user-facing fields and generate clarification prompt
    system_prompt = (
        "You are a medical assistant agent.\n"
        "Given the current tool, parameters, and context, do the following:\n"
        "1. Determine which user-facing fields are required for the current tool (do NOT include backend fields like IDs, status, etc.).\n"
        "2. Identify which required user-facing fields are missing or have placeholder/assumed values.\n"
        "   - For appointment_booking: required fields are patient_name, appointment_date, start_time, service_name\n"
        "   - For schedule_query: minimal requirements - only ask for missing info if truly needed for the specific request\n"
        "     * For doctor schedule queries ('show my schedule', 'tomorrow's appointments'), NO additional fields required - doctor_id comes from context\n"
        "     * For doctor earliest slot queries ('earliest available slot', 'earliest opening'), NO additional fields required - service_name is OPTIONAL\n"
        "     * For earliest slot detection: if no date provided, default to today - do NOT ask for date\n"
        "     * For slot suggestions: only need date if not provided\n"
        "     * Service name is OPTIONAL and should NOT be requested for general schedule viewing or doctor's earliest slot queries\n"
        "     * NEVER ask for service_name when doctor wants to view their own schedule or find earliest slots\n"
        "   - For appointment_cancellation: requires only patient_name for backend lookup to find appointment(s) to cancel\n"
        "     * NO need for appointment_date, appointment_time, or service_name - backend will find the appointment by patient name\n"
        "     * If multiple appointments exist, backend will handle selection logic\n"
        "     * Doctor_id comes from session context for RBAC validation\n"
        "   - For appointment_rescheduling: similar to cancellation, only patient_name required for finding existing appointment\n"
        "     * Backend lookup finds the current appointment, then separate validation for new date/time if needed\n"
        "   - For appointment_lookup: requirements depend on user role and query type\n"
        "     * CRITICAL: Check context.user_role first! If user_role is 'doctor' and query is about their own appointments (next patient, today's appointments, etc.), NO USER FIELDS REQUIRED - use doctor_id from context\n"
        "     * If user_role is 'patient' or query is about specific patient by name, then patient_name is required\n"
        "     * Date is optional for doctor's own appointment queries - system can find next/upcoming appointments\n"
        "   - Consider a field missing if it's not mentioned in the original user query or resolved references\n"
        "   - Detect placeholder values like '2024-12-17', '10:00', 'Consultation' when user didn't specify them\n"
        "3. If any required user-facing fields are missing, generate a natural, context-aware prompt asking ONLY for those fields.\n"
        "4. If all user-facing fields are present from the original query, parse and normalize user input for date, time, and other fields.\n"
        "5. SPECIAL: For schedule_query with earliest slot preference, be very permissive - only ask for truly missing essential info.\n"
        "6. SPECIAL: For doctor appointment_lookup queries (next patient, today's schedule, etc.), be very permissive - doctors shouldn't need to provide patient details for their own appointment queries.\n"
        "7. ALWAYS check context.user_role before determining required fields - this is critical for proper role-based validation!\n"
        "Respond in JSON with: { 'normalized': { ... }, 'required_fields': [...], 'missing_fields': [...], 'clarification_prompt': '...' }\n"
        "Do NOT ask about backend fields. Only prompt for user-facing fields that the user hasn't actually provided.\n"
        "CRITICAL: If the user only said 'Book an appointment for Alice Smith' without specifying date, time, or service, these should be considered missing fields.\n"
        "CRITICAL: For service type detection - only infer service if clear keywords are present (e.g., 'consultation', 'checkup', 'follow-up', 'surgery'). If vague like 'book appointment' without service keywords, leave service_name as missing and ask for clarification.\n"
        "CRITICAL: For earliest slot requests by doctors ('When is my earliest available slot?', 'What's my earliest opening?'), DO NOT ask for service_name OR date - default to today and show all available slots.\n"
        "CRITICAL: For doctor queries about their own appointments ('Who is my next patient?', 'What's my schedule today?', 'Show me my schedule for tomorrow'), do NOT ask for patient_name or appointment_date - these are doctor-context queries.\n"
        "CRITICAL: If query_intent is 'show_schedule' and user_role is 'doctor', NO FIELDS ARE REQUIRED - the doctor is asking to see their own appointments.\n"
        "CRITICAL: If query_intent is 'slot_suggestion' and user_role is 'doctor', service_name is OPTIONAL - doctors can query for earliest slots without specifying service type.\n"
        "CRITICAL: If slot_preference is 'earliest' in resolved_references, DO NOT ask for service_name - this is an earliest slot query where service is optional.\n"
        "CRITICAL: ALWAYS examine context.user_role to determine appropriate validation rules!"
    )
    llm_input = {
        "tool": tool,
        "parameters": resolved,
        "original_query": state.get("current_query", ""),
        "context": {
            "user_role": state.get("user_role"),
            "query_intent": state.get("query_intent"),
            "resolved_references": resolved_references,
            "slot_preference": resolved_references.get("slot_preference"),
            "patient_context": state.get("patient_context"),
            "doctor_context": state.get("doctor_context"),
            "conversation_memory": state.get("conversation_memory"),
        }
    }
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(llm_input, default=str))
    ]
    try:
        response = config.llm.invoke(messages)
        result = safe_json_parse(response.content, "slot validator")
    except Exception as e:
        logger.error(f"Slot validator LLM response parse error: {e}")
        result = {
            "normalized": {},
            "required_fields": [],
            "missing_fields": [],
            "clarification_prompt": "Sorry, I couldn't parse the required fields."
        }

    # Merge original tool_parameters with normalized values from LLM
    parsed_params = result.get("normalized", {})
    merged_params = {**params, **parsed_params}
    state["tool_parameters"] = merged_params
    all_missing = result.get("missing_fields", [])
    # Define backend/internal fields (never prompt user for these)
    missing_backend = [f for f in BACKEND_FIELDS if not state["tool_parameters"].get(f)]
    missing_natural = [f for f in all_missing if f not in BACKEND_FIELDS]

    # Always populate required_lookups with all backend fields for appointment_booking
    if state.get("selected_tool") == "appointment_booking":
        state["required_lookups"] = BACKEND_FIELDS
    elif state.get("selected_tool") == "schedule_query":
        # Schedule query always needs backend processing for parameter validation
        state["required_lookups"] = ["doctor_id", "date"]  
    elif missing_natural and missing_backend:
        state["required_lookups"] = list(set(missing_natural + missing_backend))
    elif missing_backend:
        state["required_lookups"] = missing_backend
    elif missing_natural:
        state["required_lookups"] = missing_natural
    else:
        state["required_lookups"] = []

    # Routing logic is now handled by graph condition functions, not by node state.
    if missing_natural and missing_backend:
        state["slot_validation"] = {"status": "missing", "fields": missing_natural}
        state["clarification_prompt"] = result.get("clarification_prompt", "")
        state["formatted_response"] = result.get("clarification_prompt", "")
    elif state.get("selected_tool") == "appointment_query_executor":
        # CRITICAL FIX: appointment_query_executor should go directly to tool execution
        state["slot_validation"] = {"status": "backend_complete", "fields": []}
        state["clarification_prompt"] = ""
        state["formatted_response"] = ""
    elif missing_backend or state.get("selected_tool") == "schedule_query":
        # For schedule_query, always route to backend_lookup for parameter processing
        state["slot_validation"] = {"status": "missing_backend", "fields": missing_backend or ["schedule_query_params"]}
        state["clarification_prompt"] = ""
        state["formatted_response"] = ""
    elif missing_natural:
        state["slot_validation"] = {"status": "missing", "fields": missing_natural}
        state["clarification_prompt"] = result.get("clarification_prompt", "")
        state["formatted_response"] = result.get("clarification_prompt", "")
    else:
        state["slot_validation"] = {"status": "ok", "fields": []}
        state["clarification_prompt"] = ""
        state["formatted_response"] = ""
    state["has_errors"] = False
    return state

# ========== APPOINTMENT VALIDATION NODES ==========
"""
Appointment Validation Nodes
----------------------------
These nodes perform specific appointment validations like checking for overlaps,
validating working hours, and ensuring appointments don't conflict with existing
schedules. These are deterministic validation nodes that use database queries.
"""

def appointment_overlap_check_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Check for overlapping appointments for the doctor at the requested time.
    Pure logic: deterministic DB query, no LLM.
    """
    from ..tools.database import check_appointment_overlap
    params = state.get("tool_parameters", {})
    doctor_id = params.get("DoctorID")
    appt_time = params.get("AppointmentTime")
    duration = params.get("ServiceDuration")
    # Compute end_time from appt_time and duration (duration in minutes)
    if doctor_id and appt_time and duration:
        from datetime import datetime, timedelta
        try:
            start_dt = datetime.strptime(appt_time, "%Y-%m-%d %H:%M:%S")
            end_dt = start_dt + timedelta(minutes=int(duration))
            end_time = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            overlap = check_appointment_overlap(doctor_id, appt_time, end_time)
            if overlap:
                state["has_errors"] = True
                state.setdefault("errors", []).append("Requested slot overlaps with an existing appointment.")
                state["appointment_overlap"] = True
            else:
                state["appointment_overlap"] = False
        except Exception as e:
            state["has_errors"] = True
            state.setdefault("errors", []).append(f"Error in overlap check: {e}")
            state["appointment_overlap"] = None
    else:
        state["appointment_overlap"] = None
    return state

def doctor_schedule_check_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Check if the requested appointment time is within the doctor's working hours and not during off time.
    Pure logic: deterministic DB query, no LLM.
    """
    from ..tools.database import check_doctor_working_hours, check_doctor_off_schedule
    params = state.get("tool_parameters", {})
    doctor_id = params.get("DoctorID")
    appt_time = params.get("AppointmentTime")
    if doctor_id and appt_time:
        try:
            if not check_doctor_working_hours(doctor_id, appt_time):
                state["has_errors"] = True
                state.setdefault("errors", []).append("Requested time is outside doctor's working hours.")
                state["working_hours_ok"] = False
            elif check_doctor_off_schedule(doctor_id, appt_time):
                state["has_errors"] = True
                state.setdefault("errors", []).append("Requested time is during doctor's off schedule.")
                state["working_hours_ok"] = False
            else:
                state["working_hours_ok"] = True
        except Exception as e:
            state["has_errors"] = True
            state.setdefault("errors", []).append(f"Error in schedule check: {e}")
            state["working_hours_ok"] = None
    else:
        state["working_hours_ok"] = None
    return state

# ========== RESPONSE FORMATTING NODES ==========
"""
Response Formatting Nodes
-------------------------
These nodes use LLM to format final responses based on tool results and context.
They handle both successful responses and error messages, providing natural language
responses that are context-aware and user-friendly.
"""

def response_formatter_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Formats the final response using LLM, based on tool_results and confirmation context.
Format a helpful, natural response based ONLY on the tool_results and confirmation_context. Include relevant context and suggest follow-up actions if appropriate.
Respond in JSON format:
{{
    "formatted_response": "...",
    "response_metadata": {{}},
    "suggested_followups": []
}}
"""
    # ...existing docstring removed, only valid Python code remains...
    logger.info("Formatting response using LLM")
    # Compose a context for the LLM that includes all relevant state, errors, clarifications, and tool results
    from datetime import datetime
    current_datetime = datetime.now()
    
    confirmation_context = {
        "tool_results": state.get("tool_results", []),
        "tool_parameters": state.get("tool_parameters", {}),
        "query_intent": state.get("query_intent"),
        "user_role": state.get("user_role"),
        "doctor_id": state.get("doctor_id"),
        "patient_context": state.get("patient_context"),
        "resolved_references": state.get("resolved_references", {}),
        "original_query": state.get("current_query"),
        "clarification_prompt": state.get("clarification_prompt", ""),
        "slot_validation": state.get("slot_validation", {}),
        "errors": state.get("errors", []),
        "has_errors": state.get("has_errors", False),
        "response_metadata": state.get("response_metadata", {}),
        "rbac_error": state.get("rbac_error", False),
        "error_message": state.get("error_message", ""),
        "current_date": current_datetime.strftime("%Y-%m-%d"),
        "current_datetime": current_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        "date_context": {
            "today": current_datetime.strftime("%Y-%m-%d"),
            "tomorrow": (current_datetime + timedelta(days=1)).strftime("%Y-%m-%d"),
            "yesterday": (current_datetime - timedelta(days=1)).strftime("%Y-%m-%d")
        }
    }
    # --- Context-aware prompt for logged-in doctors ---
    user_role = state.get("user_role", "user")
    doctor_id = state.get("doctor_id")
    doctor_name = state.get("tool_parameters", {}).get("DoctorName") or state.get("resolved_references", {}).get("DoctorName")
    is_doctor = user_role.lower() == "doctor" or (doctor_id and str(doctor_id) != "" and str(doctor_id) != "None")
    # Compose system prompt for second-person responses if doctor is logged in
    if is_doctor:
        system_prompt = f"""You are a helpful medical assistant acting as support staff for Dr. {doctor_name or doctor_id}.
The user interacting with you is a doctor (DoctorId: {doctor_id}, DoctorName: {doctor_name or 'Unknown'}).
Respond as their medical assistant, speaking FROM the perspective of helping manage their practice.

DATE CONTEXT: Current date is {current_datetime.strftime("%Y-%m-%d")} ({current_datetime.strftime("%A, %B %d, %Y")}).
When interpreting appointment dates in tool_results:
- {current_datetime.strftime("%Y-%m-%d")} = "today"
- {(current_datetime + timedelta(days=1)).strftime("%Y-%m-%d")} = "tomorrow" 
- {(current_datetime - timedelta(days=1)).strftime("%Y-%m-%d")} = "yesterday"
CRITICAL: Always use the correct relative date terms based on the actual appointment date in the data.

IMPORTANT: Generate dynamic, varied responses based ONLY on the actual tool_results data provided. 
DO NOT copy examples or use template phrases. Each response should be unique and contextual.

CRITICAL RESPONSE RULES:
1. Speak as the doctor's assistant helping with their schedule and patients
2. Use professional medical assistant language like "I found...", "I've located...", "I can help you..."
3. Always refer to patients by name when providing information
4. Present information clearly and professionally as support staff would
5. FOR RBAC/ACCESS ERRORS: If rbac_error is true or error_message contains access denial, explain clearly that the patient belongs to another doctor and cannot be accessed
1. For successful appointment bookings, include clear confirmation language:
   - Confirm the booking was successful
   - Include patient name, date, and time
   - Use confirmation words like "booked", "scheduled", "confirmed"

2. For appointment conflicts, clearly explain:
   - Why the booking cannot proceed
   - What existing appointment conflicts
   - Suggest alternative available times

3. For validation errors, be specific about:
   - What information is missing
   - Why the date/time is invalid
   - What the working hours are

4. **SLOT SUGGESTION QUERIES** - Be precise about what the user asked for:
   - If they ask for "next 3 slots", show exactly 3 slots (not more)
   - If they ask for "earliest slot", show only the earliest one
   - If they ask for "available slots", can show multiple but be concise
   - Format times clearly (e.g., "2:12 PM - 2:33 PM")
   - Don't include unnecessary information about existing appointments unless specifically asked
   - Focus on answering the exact question asked

5. **TIME-AWARE APPOINTMENT DISPLAY** - When showing appointments for today:
   - If tool_results contain "time_categorized": true, use the categorized data
   - Use past tense for completed appointments
   - Use present tense for current appointments  
   - Use future tense for upcoming appointments
   - CRITICAL: Base your response ONLY on the actual tool_results data
   - Use natural, varied language - avoid copying template phrases
   - If no time categorization is available, show all appointments normally

6. For schedule queries, provide a comprehensive overview with time context when available.

7. **NEXT PATIENT QUERIES** - For queries asking "Who is my next patient?" or similar:
   - When tool_results contain only 1 appointment, give a focused answer about that specific patient
   - Include the patient name, service type, and appointment time in natural language
   - Do NOT provide a full schedule summary for single-patient queries
   - Focus specifically on answering the exact question asked
   - Use varied, natural phrasing - avoid template-like responses
   - Examples of assistant responses: "I found...", "I see you have...", "Looking at your schedule...", etc.

8. **PATIENT HISTORY SUMMARIES** - For comprehensive patient information requests:
   - When lookup_type is "patient_history", provide a detailed, organized summary
   - Include appointment count, service types, visit frequency, and chronological history
   - Structure the response with clear sections: Overview, Recent Appointments, Full History
   - Highlight patterns, frequency of visits, and services received
   - Show most recent appointments first in the detailed history
   - Include dates, services, and any status information
   - Present information professionally as a medical assistant would review a patient file
   - Use section headers and organized formatting for readability

RESPONSE GENERATION GUIDELINES:
- Use ONLY the actual data from tool_results - never use placeholder examples
- Vary your language and phrasing for each response
- Be creative with how you present information while staying factual
- Ground all factual information in the actual tool outputs
- Use natural, conversational language as a medical assistant would
- NEVER copy exact phrases from prompts or examples
- ANTI-TEMPLATE RULE: Avoid formulaic responses - use creative, varied language instead
- Each response should sound unique and personalized, not like a template
- Use fresh, varied introductions for each type of query
- Speak FROM the assistant's perspective helping the doctor

If there are errors, ambiguities, or missing information, explain them clearly and politely.
If a clarification prompt is present, use it to ask the doctor for more information.
If the request was successful, summarize the results professionally as a medical assistant would.
Always respond in JSON with the following format:
{{
  "formatted_response": "...",
  "response_metadata": {{ ... }},
  "suggested_followups": []
}}"""
        prompt = (
            "Given the following context, generate a single, natural, user-facing response. "
            "CRITICAL ANALYSIS: First analyze the user's original query to understand exactly what they're asking for. "
            "PRECISION RULE: Answer ONLY what was asked - don't add extra information unless directly relevant. "
            "Examples: "
            "- 'next 3 slots' = show exactly 3 slots, no schedule summary "
            "- 'earliest slot' = show only the earliest slot "
            "- 'next patient' = show only the next patient "
            "- 'today's schedule' = show full schedule "
            "- 'summarize everything about...' = comprehensive patient history with organized sections "
            "CRITICAL: Use creative, varied language - avoid template phrases like 'Your next patient is...', 'You have a total of...', 'lined up', 'It looks like', 'busy day ahead'. "
            "Instead use natural variations like 'You'll be seeing...', 'Coming up next...', 'Your appointment is with...', 'Your schedule includes...', 'Today brings...', etc. "
            "FORBIDDEN: Do not use the phrases 'You have a total of', 'lined up', 'It looks like', 'busy day ahead' - find fresh alternatives. "
            "CRITICAL: Base your response ONLY on the actual tool_results data provided. If tool_results is empty or contains no appointments, say so - do NOT invent appointment details. "
            "If there are errors, ambiguities, or missing information, explain them clearly and politely. "
            "If a clarification prompt is present, use it to ask the doctor for more information. "
            "If the request was successful, summarize the results in a user-friendly way, always using second person (addressing the doctor as 'you'). "
            "Respond ONLY in the required JSON format.\n\n"
            f"CONTEXT:\n{json.dumps(confirmation_context, indent=2, default=str)}"
        )
    else:
        system_prompt = """You are a helpful medical assistant.
Your job is to generate a natural, context-aware response for the user based on the provided context.
If there are errors, ambiguities, or missing information, explain them clearly and politely.
If a clarification prompt is present, use it to ask the user for more information.
If the request was successful, summarize the results in a user-friendly way.
Always respond in JSON with the following format:
{
  "formatted_response": "...",
  "response_metadata": { ... },
  "suggested_followups": []
}"""
        prompt = (
            "Given the following context, generate a single, natural, user-facing response. "
            "If there are errors, ambiguities, or missing information, explain them clearly and politely. "
            "If a clarification prompt is present, use it to ask the user for more information. "
            "If the request was successful, summarize the results in a user-friendly way. "
            "Respond ONLY in the required JSON format.\n\n"
            f"CONTEXT:\n{json.dumps(confirmation_context, indent=2, default=str)}"
        )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    response = config.llm.invoke(messages)
    try:
        result = safe_json_parse(response.content, "response formatter")
        logger.info(f"Parsed result: {result}")
        # Always use LLM's formatted_response, even for errors/clarifications
        state["formatted_response"] = result.get("formatted_response", "I processed your request successfully.")
        state["has_errors"] = state.get("has_errors", False)
        state["response_metadata"] = result.get("response_metadata", {})
        state["response_metadata"].update({
            "intent": state["query_intent"],
            "tool_used": state.get("selected_tool"),
            "has_errors": state.get("has_errors", False),
            "context_resolved": bool(state.get("resolved_references"))
        })
        logger.info("Response formatted successfully")
    except Exception as e:
        logger.error(f"Response formatting error: {e}")
        state.setdefault("errors", []).append(f"Response formatting failed: {e}")
        state["has_errors"] = True
        # Fallback: ask LLM for a generic error message
        fallback_prompt = (
            "You are a helpful medical assistant. The previous attempt to generate a response failed. "
            "Please generate a polite, user-facing error message based on the following errors and context. "
            "Respond ONLY in JSON with a 'formatted_response' key.\n\n"
            f"ERRORS: {state.get('errors', [])}\nCONTEXT: {json.dumps(confirmation_context, indent=2, default=str)}"
        )
        fallback_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=fallback_prompt)
        ]
        try:
            fallback_response = config.llm.invoke(fallback_messages)
            fallback_result = safe_json_parse(fallback_response.content, "response formatter fallback")
            state["formatted_response"] = fallback_result.get("formatted_response", "Sorry, something went wrong. Please try again later.")
        except Exception as e2:
            logger.error(f"Response fallback formatting error: {e2}")
            state["formatted_response"] = "Sorry, something went wrong. Please try again later."
    return state

# ========== MULTI-STEP PLANNING NODES ==========
"""
Multi-Step Planning Nodes
-------------------------
These nodes handle complex multi-step operations that require user interaction
and progressive slot-filling. They determine missing user-dependent fields
and generate appropriate prompts to collect missing information.
"""

def multi_step_planner_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Multi-step planner for slot-filling. Uses LLM to reason about missing user-dependent fields and generate a user-facing prompt.
    Only prompts for service name, patient name, and date/time. All other fields are auto-resolved by the agent.
    """
    tool_params = state.get("tool_parameters", {})
    # Determine which user-dependent fields are missing (None, "unknown", or empty string)
    missing_fields = [field for field in USER_DEPENDENT_FIELDS if tool_params.get(field) in [None, "unknown", ""]]
    # Build context for LLM prompt
    context = {
        "patient_name": tool_params.get("patient_name"),
        "doctor_id": tool_params.get("doctor_id"),
        "service_name": tool_params.get("service_name"),
        "appointment_time": tool_params.get("start_time"),
        "appointment_date": tool_params.get("appointment_date"),
        "intent": state.get("query_intent"),
        "missing_fields": missing_fields,
        "tool_parameters": tool_params,
        "original_query": state.get("current_query"),
        "resolved_references": state.get("resolved_references", {})
    }
    # Compose a context-aware prompt that only asks for missing fields, never repeating already filled fields
    prompt = f"""
You are a medical assistant agent helping to book an appointment. The following user-dependent fields are still missing: {missing_fields}.
Here is the current context:
{json.dumps(context, indent=2, default=str)}

Generate a polite, context-aware message asking for ONLY the missing fields. If only one field is missing, ask for it directly. If multiple are missing, ask for all together. Never ask for fields that are already filled. Use available context (patient name, doctor, service, date, time) to make your question natural and specific. Respond in plain English, not JSON.
"""
    system_prompt = "You are a helpful medical assistant."
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    response = config.llm.invoke(messages)
    # Use the LLM's output as the user-facing prompt
    state["formatted_response"] = response.content.strip()
    state["required_lookups"] = missing_fields
    return state


def clean_json_response(response_content: str) -> str:
    """
    Clean OpenAI response content to extract valid JSON.
    Handles responses wrapped in ```json code blocks and removes comments.
    """
    # Remove code block markers robustly
    content = response_content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    if content.startswith("```"):
        content = content[len("```"):].strip()
    if content.endswith("```"):
        content = content[:-len("```")].strip()
    
    # Remove single-line comments (// comments)
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove // comments but preserve the rest of the line
        if '//' in line:
            comment_pos = line.find('//')
            before_comment = line[:comment_pos]
            quote_count = before_comment.count('"') - before_comment.count('\\"')
            if quote_count % 2 == 0:
                line = line[:comment_pos].rstrip()
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def preprocess_date_references(query: str) -> Dict[str, str]:
    """
    Preprocess date references like 'tomorrow', 'today', 'next week tuesday' etc.
    Returns a mapping of date references to actual dates.
    """
    date_mappings = {}
    current_date = datetime.now()
    
    query_lower = query.lower()

    # Handle today
    if 'today' in query_lower:
        date_mappings['today'] = current_date.strftime('%Y-%m-%d')

    # Handle tomorrow
    if 'tomorrow' in query_lower:
        tomorrow_date = current_date + timedelta(days=1)
        date_mappings['tomorrow'] = tomorrow_date.strftime('%Y-%m-%d')

    # Handle yesterday
    if 'yesterday' in query_lower:
        yesterday_date = current_date - timedelta(days=1)
        date_mappings['yesterday'] = yesterday_date.strftime('%Y-%m-%d')

    # Handle "next week [day]" more intelligently
    if 'next week' in query_lower:
        # Find next week's Monday
        days_until_next_monday = (7 - current_date.weekday()) % 7
        if days_until_next_monday == 0:  # If today is Monday
            days_until_next_monday = 7
        next_monday = current_date + timedelta(days=days_until_next_monday)
        
        # Check for specific days
        if 'tuesday' in query_lower:
            next_week_date = next_monday + timedelta(days=1)  # Monday + 1 = Tuesday
        elif 'wednesday' in query_lower:
            next_week_date = next_monday + timedelta(days=2)
        elif 'thursday' in query_lower:
            next_week_date = next_monday + timedelta(days=3)
        elif 'friday' in query_lower:
            next_week_date = next_monday + timedelta(days=4)
        elif 'saturday' in query_lower:
            next_week_date = next_monday + timedelta(days=5)
        elif 'sunday' in query_lower:
            next_week_date = next_monday + timedelta(days=6)
        elif 'monday' in query_lower:
            next_week_date = next_monday
        else:
            # Default to next Monday if no specific day mentioned
            next_week_date = next_monday
            
        date_mappings['next week'] = next_week_date.strftime('%Y-%m-%d')
        
        # Also map the specific phrase if it exists
        import re
        next_week_pattern = r'next week\s+(\w+)'
        match = re.search(next_week_pattern, query_lower)
        if match:
            full_phrase = f"next week {match.group(1)}"
            date_mappings[full_phrase] = next_week_date.strftime('%Y-%m-%d')

    # Handle this week
    if 'this week' in query_lower:
        # Find the start of this week (Monday)
        days_since_monday = current_date.weekday()
        week_start = current_date - timedelta(days=days_since_monday)
        date_mappings['this week'] = week_start.strftime('%Y-%m-%d')

    return date_mappings


def safe_json_parse(response_content: str, node_name: str) -> Dict[str, Any]:
    """
    Safely parse JSON response with cleaning and error handling.
    """
    try:
        cleaned_content = clean_json_response(response_content)
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {node_name}: {e}")
        logger.error(f"Response content was: {response_content}")
        
        # Try to extract JSON pattern manually
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, cleaned_content, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except:
                pass
        
        # Return a safe default
        return {
            "error": f"Failed to parse JSON in {node_name}",
            "raw_content": response_content[:200] + "..." if len(response_content) > 200 else response_content
        }

# ========== CONTEXT RESOLUTION NODES ==========
"""
Context Resolution Nodes
------------------------
These nodes analyze user queries against conversation memory to resolve references
like "next patient", "she", "tomorrow", etc. They use both LLM reasoning and
MCP (Model Context Protocol) for standardized context management.
"""

def context_resolver_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Enhanced context resolver with MCP (Model Context Protocol) integration.
    This node analyzes the current query against conversation memory and 
    existing context to resolve references like "next patient", "she", etc.
    Enhanced with MCP for standardized context management.
    """
    logger.info(f"Context resolver processing: {state['current_query']}")
    session_id = state.get("session_id", "default_session")
    
    # Parse session for automatic RBAC - extract doctor_id and user_role from session
    user_role = state.get("user_role", "")
    doctor_id = state.get("doctor_id", "")
    
    # Auto-populate from session if not already set
    if session_id.startswith("streamlit_doctor_") and not doctor_id:
        parts = session_id.split("_")
        if len(parts) >= 3 and parts[1] == "doctor":
            doctor_id = parts[2]
            user_role = "doctor"
            state["doctor_id"] = doctor_id
            state["user_role"] = user_role
            logger.info(f"[RBAC AUTO] Parsed session {session_id} -> user_role='doctor', doctor_id='{doctor_id}'")
    
    # Preprocess date references before other processing
    date_mappings = preprocess_date_references(state["current_query"])
    logger.info(f"Date mappings found: {date_mappings}")
    # First, try MCP context resolution for better reference handling
    mcp_resolved_refs = {}
    query_lower = state["current_query"].lower()
    reference_patterns = ["next patient", "her", "him", "that appointment", "this patient", "my schedule"]
    for pattern in reference_patterns:
        if pattern in query_lower:
            resolved = mcp_context_manager.resolve_reference(
                reference=pattern,
                session_id=session_id
            )
            if resolved:
                mcp_resolved_refs[pattern] = resolved
                logger.info(f"MCP resolved '{pattern}' -> {resolved.get('patient_name', 'context')}")
    # AI-driven context resolution - let LLM intelligently infer intent and timing preferences
    system_prompt = """
You are a medical assistant context resolver with enhanced patient reference tracking and conversation continuity.

CRITICAL RULES:
1. Track the CURRENT ACTIVE PATIENT from the most recent conversation context
2. When resolving pronouns ('him', 'her', 'they') always refer to the CURRENT ACTIVE PATIENT from the immediate context
3. Intelligently detect when users want the earliest/next available appointment times
4. Understand natural language timing preferences and convert them to structured data
5. SMART SERVICE DETECTION: Only infer service types when clear keywords are present
6. CONVERSATION CONTINUITY: When user provides missing information in follow-up, combine with previous incomplete requests
7. COMPREHENSIVE SUMMARIES: Detect requests for detailed patient information

COMPREHENSIVE SUMMARY DETECTION:
When users ask for detailed information about a patient, use "patient_history" intent:
- "summarize everything related to..." → patient_history
- "tell me about..." → patient_history  
- "give me details on..." → patient_history
- "what's the full picture for..." → patient_history
- "comprehensive summary of..." → patient_history
- "everything about..." → patient_history
- "all information on..." → patient_history
- Combined with patient references ("my next client", "Sarah Wilson", etc.)

IMPORTANT: When summarize/everything/details are mentioned, ALWAYS use patient_history intent, not patient_lookup.

Context Priority (highest to lowest):
1. Current query explicit mentions
2. Most recent conversation turn patient references
3. Previous conversation context for incomplete appointments
4. MCP resolved references
5. Historical context

PATIENT REFERENCE RESOLUTION:
- "him/her/they" → Current active patient from immediate context
- "next patient" → Look for upcoming scheduled patients
- "that patient" → Most recently mentioned patient
- Always prioritize the patient from the CURRENT conversation context

CONVERSATION CONTINUITY (CRITICAL):
When processing follow-up messages, check if there's an incomplete appointment request from recent conversation:
- If previous message had patient, date, time but missing service → combine with current service mention
- If previous message had patient, service but missing date/time → combine with current date/time mention
- Look for incomplete appointment details in recent conversation turns and merge them intelligently

Examples of continuity:
Previous: "Book Frank Miller for next week tuesday at 3 pm" (missing service)
Current: "book him for a consultation"
CONVERSATION CONTINUITY (CRITICAL):
When processing follow-up messages, check if there's an incomplete appointment request from recent conversation:
- If previous message had patient, date, time but missing service → combine with current service mention
- If previous message had patient, service but missing date/time → combine with current date/time mention
- Look for incomplete appointment details in recent conversation turns and merge them intelligently

SERVICE TYPE DETECTION (CRITICAL):
ONLY infer service_name when clear keywords are present:
- "consultation", "consult" → "consultation"
- "checkup", "check-up", "routine check" → "checkup"
- "follow-up", "followup" → "follow-up"
- "surgery", "operation" → "surgery"
- "exam", "examination" → "examination"
- "screening" → "screening"
- "therapy", "treatment" → "therapy"

DO NOT infer service_name for vague statements:
- "book an appointment" → service_name: null
- "schedule him" → service_name: null
- "earliest slot" → service_name: null
- "book her for tomorrow" → service_name: null

ONLY set service_name when the user explicitly mentions a service type keyword.

INTELLIGENT TIMING INFERENCE:
Detect when users want the earliest available slot through natural language understanding:
- "earliest slot/time/appointment" → User wants the first available time
- "next available" → User wants the soonest possible appointment
- "first available" → User wants the earliest time slot
- "when can I book..." → Often implies earliest availability interest
- "as soon as possible" → Clear earliest preference
- "any time tomorrow" → May indicate earliest preference

CRITICAL: Set slot_preference in resolved_references:
- "earliest" for explicit first available slot requests (booking intent)
- "suggest" ONLY when user explicitly asks for options/suggestions (inquiry intent)
- "next_1" for single slot requests ("next slot", "earliest slot")
- "next_N" for specific quantity requests ("next 3 slots", "first 5 available") where N is the number
- "suggest_limited" for bounded suggestions ("show me a few slots", "some options")
- null for specific time requests OR when missing information should be requested

QUANTITY DETECTION EXAMPLES:
- "next 3 slots" → slot_preference: "next_3"
- "first 5 available" → slot_preference: "next_5" 
- "next slot" → slot_preference: "next_1"
- "earliest slot" → slot_preference: "earliest" (same as next_1)
- "what are my next available slots" → slot_preference: "suggest"
- "show me some options" → slot_preference: "suggest_limited"

IMPORTANT: Do NOT auto-default to "suggest" for incomplete bookings. If information is missing, set slot_preference to null so the system can ask for the missing details.

You MUST include slot_preference in your resolved_references when detected.

Query: "What appointments do I have today?"
Output: {"query_intent": "show_schedule", "resolved_references": {"appointment_date": "2025-08-24", "patient_name": null, "service_name": null, "slot_preference": null}}

Query: "Show my schedule for next Monday"
Output: {"query_intent": "show_schedule", "resolved_references": {"appointment_date": "2025-08-25", "patient_name": null, "service_name": null, "slot_preference": null}}

Query: "book him for my earliest slot tomorrow"
Output: {"query_intent": "book_appointment", "resolved_references": {"patient_name": "[from context]", "appointment_date": "2025-08-19", "slot_preference": "earliest", "service_name": null}}

Query: "When is my earliest available slot?"
Output: {"query_intent": "slot_suggestion", "resolved_references": {"appointment_date": null, "patient_name": null, "service_name": null, "slot_preference": "earliest"}}

Query: "Summarize everything related to my next client"
Output: {"query_intent": "patient_history", "resolved_references": {"patient_name": "Sarah Wilson", "appointment_date": null, "service_name": null, "slot_preference": null}}

Query: "What's my earliest opening today?"
Output: {"query_intent": "slot_suggestion", "resolved_references": {"appointment_date": "2025-08-25", "patient_name": null, "service_name": null, "slot_preference": "earliest"}}

Query: "what are my next 3 available slots"
Output: {"query_intent": "slot_suggestion", "resolved_references": {"appointment_date": "2025-08-25", "patient_name": null, "service_name": null, "slot_preference": "next_3"}}

Query: "show me the first 5 available appointments"
Output: {"query_intent": "slot_suggestion", "resolved_references": {"appointment_date": "2025-08-25", "patient_name": null, "service_name": null, "slot_preference": "next_5"}}

Query: "what's my next slot"
Output: {"query_intent": "slot_suggestion", "resolved_references": {"appointment_date": "2025-08-25", "patient_name": null, "service_name": null, "slot_preference": "next_1"}}

Query: "Who is my next patient?"
Output: {"query_intent": "next_patient", "resolved_references": {"patient_name": null, "appointment_date": null, "service_name": null, "slot_preference": null}}

Query: "What's my next appointment?"
Output: {"query_intent": "next_patient", "resolved_references": {"patient_name": null, "appointment_date": null, "service_name": null, "slot_preference": null}}

Output Schema (JSON only, no markdown):
{
  "query_intent": "book_appointment|reschedule_appointment|cancel_appointment|slot_suggestion|patient_lookup|patient_history|show_schedule|appointment_lookup|next_patient|...",
  "resolved_references": {
    "patient_name": "...",
    "appointment_date": "YYYY-MM-DD", 
    "appointment_time": "HH:MM",
    "service_name": "...|null",
    "slot_preference": "earliest|suggest|suggest_limited|next_1|next_2|next_3|next_4|next_5|next_10|null"
  },
  "context_updates": {
    "active_patients": ["patient_name"],
    "scheduling_preferences": {...}
  }
}
""".strip()
    mcp_context_summary = mcp_context_manager.get_context_summary(session_id)
    
    # Extract recent appointment-related context for continuity
    recent_appointment_context = None
    recent_messages = [msg.content for msg in state["messages"][-5:] if hasattr(msg, 'content')]
    
    # Look for incomplete appointment requests in recent messages
    for message in reversed(recent_messages[:-1]):  # Exclude current message
        if message and any(word in message.lower() for word in ['book', 'schedule', 'appointment']):
            # Try to extract appointment details from previous message
            import re
            date_match = re.search(r'\b(tomorrow|today|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})\b', message.lower())
            time_match = re.search(r'\b(\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm))\b', message.lower())
            patient_match = re.search(r'\b(?:for|book|schedule)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b', message)
            service_match = re.search(r'\b(consultation|checkup|check-up|follow-up|surgery|exam|examination|screening|therapy|treatment)\b', message.lower())
            
            if patient_match or date_match or time_match:
                recent_appointment_context = {
                    "message": message,
                    "patient": patient_match.group(1) if patient_match else None,
                    "date_text": date_match.group(1) if date_match else None,
                    "time_text": time_match.group(1) if time_match else None,
                    "service": service_match.group(1) if service_match else None
                }
                break
    
    # Look for patient names in recent messages (reverse order for most recent)
    recent_patient_name = None
    for message in reversed(recent_messages):
        if message:
            # Simple pattern matching for patient names (could be enhanced)
            import re
            name_patterns = [
                r'\b(?:for|book|schedule)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:for|appointment)\b',
                r'\bpatient\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
            ]
            for pattern in name_patterns:
                match = re.search(pattern, message)
                if match:
                    recent_patient_name = match.group(1)
                    break
            if recent_patient_name:
                break
    
    context_info = {
        "current_query": state["current_query"],
        "user_role": state["user_role"],
        "doctor_id": state.get("doctor_id"),
        "patient_context": state.get("patient_context"),
        "doctor_context": state.get("doctor_context"),
        "conversation_memory": state["conversation_memory"],
        "recent_messages": recent_messages,
        "most_recent_patient": recent_patient_name,  # Add this for context priority
        "recent_appointment_context": recent_appointment_context,  # Add incomplete appointment details
        "mcp_context_summary": mcp_context_summary,
        "mcp_resolved_references": mcp_resolved_refs,
        "date_mappings": date_mappings,
        "current_date": datetime.now().strftime('%Y-%m-%d')
    }
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(context_info, indent=2, default=str))
        ]
        response = config.llm.invoke(messages)
        logger.info(f"Context resolver OpenAI response content: '{response.content}'")
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from OpenAI in context resolver")
            raise ValueError("Empty response from OpenAI")
        try:
            result = safe_json_parse(response.content, "context resolver")
        except Exception as je:
            logger.error(f"Context resolution error: {je}")
            # Use fallback logic if JSON parsing fails
            raise je
        # Update state with resolved information
        state["query_intent"] = result.get("query_intent")
        # Booking-intent fallback heuristic (traceable override)
        if state["query_intent"] == "time_specific_lookup":
            booking_verbs = ["book", "schedule", "add", "make", "create", "set", "reserve"]
            query_words = re.findall(r'\b\w+\b', state["current_query"].lower())
            if any(verb in query_words for verb in booking_verbs):
                # Preserve original intent for debugging/multi-intent
                state["query_intent_original"] = state["query_intent"]
                state["query_intent"] = "book_appointment"
                logger.warning("Overriding intent: time_specific_lookup → book_appointment (based on verb match)")
        # Final resolved references:
        # - LLM output takes precedence
        # - MCP fills gaps
        # - Date mappings are injected into the same object
        # Booking-intent fallback heuristic (traceable override)
        if state["query_intent"] == "time_specific_lookup":
            booking_verbs = ["book", "schedule", "add", "make", "create", "set", "reserve"]
            query_words = re.findall(r'\b\w+\b', state["current_query"].lower())
            if any(verb in query_words for verb in booking_verbs):
                # Preserve original intent for debugging/multi-intent
                state["query_intent_original"] = state["query_intent"]
                state["query_intent"] = "book_appointment"
                logger.warning("Overriding intent: time_specific_lookup → book_appointment (based on verb match)")
        # AI-driven slot preference inference from LLM response
        llm_resolved = result.get("resolved_references", {})
        final_resolved = {**mcp_resolved_refs, **date_mappings, **llm_resolved}
        
        # Intelligent slot preference inference based on query and LLM understanding
        query_lower = state["current_query"].lower()
        intent = state.get("query_intent", "").lower()
        
        # Let the LLM's natural understanding guide slot preference detection
        # Only override if LLM didn't already detect a preference
        llm_slot_preference = final_resolved.get("slot_preference")
        
        if "book" in intent or "appointment" in intent:
            # For booking intents, check if time is missing or if earliest-indicating phrases exist
            if not final_resolved.get("appointment_time") and not final_resolved.get("time"):
                # No specific time mentioned - check for earliest preference indicators
                earliest_indicators = [
                    "earliest", "next available", "first available", "as soon as possible",
                    "first appointment", "earliest slot", "earliest time", "soonest",
                    "next slot", "first slot"
                ]
                if any(indicator in query_lower for indicator in earliest_indicators):
                    final_resolved["slot_preference"] = "earliest"
                    logger.info(f"AI-detected earliest slot preference from: '{state['current_query']}'")
                elif not llm_slot_preference:
                    # Booking without time and no earliest indicators - leave as decided by LLM
                    # Don't auto-default to suggest, let the system ask for missing info
                    pass
            else:
                # Specific time mentioned
                if not llm_slot_preference:
                    final_resolved["slot_preference"] = None
        elif intent != "show_schedule" and any(word in intent for word in ["available", "slots", "schedule", "check"]) and any(word in query_lower for word in ["available", "options", "suggest", "show me"]):
            # Only suggest slots if user explicitly asks for options/suggestions AND it's not a show_schedule intent
            if not llm_slot_preference:
                final_resolved["slot_preference"] = "suggest"
        elif not llm_slot_preference:
            # Only set to None if LLM didn't detect any preference
            final_resolved["slot_preference"] = None
        
        state["resolved_references"] = final_resolved
        context_updates = result.get("context_updates", {})
        if context_updates.get("patient_context"):
            state = update_patient_context(state, **context_updates["patient_context"])
        logger.info(f"Context resolved - Intent: {state['query_intent']}, References: {state['resolved_references']}")
        
        # Post-processing: Detect comprehensive summary requests that were misclassified
        query_lower = state["current_query"].lower()
        summary_indicators = ["summarize", "everything", "all information", "comprehensive", "details about", "tell me about"]
        if (state['query_intent'] == "patient_lookup" and 
            any(indicator in query_lower for indicator in summary_indicators) and
            state.get('resolved_references', {}).get('patient_name')):
            logger.info(f"POST-PROCESSING: Redirecting patient_lookup to patient_history for comprehensive request")
            state['query_intent'] = "patient_history"
        
        # Post-processing: Detect next patient queries that were misclassified
        next_patient_indicators = ["next patient", "next appointment", "who is next", "who's next", "my next patient", "my next appointment"]
        if (state['query_intent'] == "appointment_lookup" and 
            any(indicator in query_lower for indicator in next_patient_indicators)):
            logger.info(f"POST-PROCESSING: Redirecting appointment_lookup to next_patient for 'next patient' request")
            state['query_intent'] = "next_patient"
            # Clear specific patient references since we want the actual next patient
            if state.get('resolved_references'):
                state['resolved_references']['patient_name'] = None
                state['resolved_references']['appointment_date'] = None
    except Exception as e:
        logger.error(f"Context resolution error: {e}")
        state["errors"].append(f"Context resolution failed: {e}")
        state["has_errors"] = True
        query_lower = state["current_query"].lower()
        if any(word in query_lower for word in ["next", "upcoming"]):
            state["query_intent"] = "next_patient"
        elif any(word in query_lower for word in ["summarize", "everything", "history", "past", "details", "comprehensive", "tell me about", "all information"]):
            state["query_intent"] = "patient_history"
        elif any(word in query_lower for word in ["schedule", "calendar"]):
            state["query_intent"] = "schedule"
        else:
            state["query_intent"] = "general_query"
        if mcp_resolved_refs:
            state["resolved_references"] = mcp_resolved_refs
            logger.info(f"Using MCP fallback references: {mcp_resolved_refs}")
    return state

# ========== TOOL SELECTION NODES ==========
"""
Tool Selection Nodes
--------------------
These nodes use LLM to select the appropriate tool based on query intent,
user role permissions, and available context. They handle tool selection
logic and parameter preparation for the selected tools.
"""

def tool_selector_node(state: AgentState, config: AgentConfig) -> AgentState:
    logger.info(f"Tool selector processing intent: {state['query_intent']}")
    
    system_prompt = config.get_system_prompt("tool_selector")
    
    # Get allowed tools for user role
    allowed_tools = config.get_role_permissions(state["user_role"])
    
    selection_context = {
        "query_intent": state["query_intent"],
        "resolved_references": state["resolved_references"],
        "user_role": state["user_role"],
        "doctor_id": state.get("doctor_id"),
        "allowed_tools": allowed_tools,
        "patient_context": state.get("patient_context"),
        "doctor_context": state.get("doctor_context")
    }
    
    # Override intent for doctor availability queries that were misclassified as "show_schedule"
    original_query = state.get("current_query", "").lower()
    resolved_refs = state.get("resolved_references", {})
    
    # Check if this is actually a doctor availability query (looking for available slots)
    if (state["query_intent"] == "show_schedule" and 
        state["user_role"] == "doctor" and
        resolved_refs.get("appointment_date") and
        any(keyword in original_query for keyword in ["schedule", "available", "slots", "free", "open"])):
        
        logger.info(f"Overriding intent from 'show_schedule' to 'schedule' for doctor availability query: {original_query}")
        selection_context["query_intent"] = "schedule"
        state["query_intent_override"] = "schedule"  # Track the override for debugging
    
    prompt = f"""
{system_prompt}

Selection context:
{json.dumps(selection_context, indent=2, default=str)}

TOOL SELECTION GUIDELINES:
- For "time_specific_lookup" intent: Use "schedule_query" with specific time filtering
- For "next_patient" intent: Use "appointment_query_executor" with query_type="next_patient" to find chronologically next appointment
- For "patient_lookup" intent: Use "appointment_query_executor" to find patient appointments
- For "show_schedule" intent: ALWAYS use "appointment_query_executor" to show existing scheduled appointments
- For "schedule" intent (when looking for available slots): Use "schedule_query" for finding available slots
- For "book_appointment" intent: Use "appointment_booking" for creating new appointments
- For "patient_history" intent: Use "appointment_query_executor" with appropriate query_type

CRITICAL RULE FOR DOCTOR AVAILABILITY QUERIES:
If the query is asking about a doctor's availability, available slots, or schedule for finding appointment times:
- Use "schedule_query" tool to find available time slots
- Examples: "Dr. Smith's schedule for February 22nd", "What slots does Dr. Johnson have available tomorrow", "Show me available times for Dr. Brown on Monday"

CRITICAL RULE FOR VIEWING EXISTING APPOINTMENTS:
If query_intent is "show_schedule" for viewing already booked appointments:
- Use "appointment_query_executor" tool to show existing scheduled appointments
- Examples: "Show me my appointments for today", "What appointments do I have tomorrow"

TIME-SPECIFIC HANDLING:
- If resolved_references contains specific times (e.g., "2 PM", "14:00"), this indicates a time-specific lookup
- Generate parameters that will filter by the specific time mentioned
- For time-specific queries, include both date and time constraints in parameters

IMPORTANT DISTINCTION:
- "show_schedule" = View existing booked appointments (use appointment_lookup)
- "schedule" = Find available appointment slots (use schedule_query)
- Doctor availability queries should use "schedule_query" regardless of how they're phrased

Select the best tool and parameters. Respond in JSON format:
{{
    "selected_tool": "tool_name",
    "tool_parameters": {{}},
    "reasoning": "Explain tool choice and parameter generation, especially for time-specific queries"
}}
"""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = config.llm.invoke(messages)
        
        # Debug the response content
        logger.info(f"Tool selector OpenAI response content: '{response.content}'")
        
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from OpenAI in tool selector")
            raise ValueError("Empty response from OpenAI")
        
        try:
            result = safe_json_parse(response.content, "tool selector")
        except Exception as je:
            logger.error(f"Tool selection error: {je}")
            # Use fallback logic if JSON parsing fails
            raise je
        
        state["selected_tool"] = result.get("selected_tool")
        state["tool_parameters"] = result.get("tool_parameters", {})
        
        # CRITICAL FIX: Override LLM tool selection for specific intents to ensure correct behavior
        if state["query_intent"] == "next_patient":
            logger.info(f"Overriding tool selection for next_patient intent: {state['selected_tool']} -> appointment_query_executor")
            state["selected_tool"] = "appointment_query_executor"
            state["tool_parameters"] = {
                "query_type": "next_patient", 
                "doctor_id": state.get("doctor_id", ""),
                "lookup_type": "next_patient"
            }
        elif state["query_intent"] == "patient_history":
            logger.info(f"Overriding tool selection for patient_history intent: {state['selected_tool']} -> appointment_lookup")
            state["selected_tool"] = "appointment_lookup"
            # Backend lookup will populate the patient_id and lookup_type
        elif state["query_intent"] == "reschedule_appointment":
            logger.info(f"Overriding tool selection for reschedule intent: {state['selected_tool']} -> appointment_rescheduling")
            state["selected_tool"] = "appointment_rescheduling"
            # RBAC will be enforced in the tool executor before actual execution
        elif state["query_intent"] == "cancel_appointment":
            logger.info(f"Overriding tool selection for cancel intent: {state['selected_tool']} -> appointment_cancellation")
            state["selected_tool"] = "appointment_cancellation"
            # RBAC will be enforced in the tool executor before actual execution
        elif state["query_intent"] == "book_appointment":
            # CRITICAL FIX: Always use appointment_booking tool for booking requests
            logger.info(f"Overriding tool selection for book_appointment intent: {state['selected_tool']} -> appointment_booking")
            state["selected_tool"] = "appointment_booking"
            # Keep the existing tool_parameters but ensure it includes required fields from resolved_references
            resolved_refs = state.get("resolved_references", {})
            tool_params = state.get("tool_parameters", {})
            
            # Merge resolved references into tool parameters
            if resolved_refs.get("patient_name"):
                tool_params["patient_name"] = resolved_refs["patient_name"]
            if resolved_refs.get("appointment_date"):
                tool_params["appointment_date"] = resolved_refs["appointment_date"]
            if resolved_refs.get("appointment_time"):
                tool_params["appointment_time"] = resolved_refs["appointment_time"]
            if resolved_refs.get("service_name"):
                tool_params["service_name"] = resolved_refs["service_name"]
            
            tool_params["doctor_id"] = state.get("doctor_id", "")
            state["tool_parameters"] = tool_params
        
        logger.info(f"Tool selected: {state['selected_tool']} with params: {state['tool_parameters']}")
        
    except Exception as e:
        logger.error(f"Tool selection error: {e}")
        state["errors"].append(f"Tool selection failed: {e}")
        state["has_errors"] = True
        # Fallback tool selection
        if state["query_intent"] == "next_patient":
            state["selected_tool"] = "appointment_lookup"
            state["tool_parameters"] = {"doctor_id": state.get("doctor_id")}
        elif state["query_intent"] == "patient_history":
            state["selected_tool"] = "appointment_lookup"
            state["tool_parameters"] = {"doctor_id": state.get("doctor_id")}
            # Extract patient name from resolved references if available
            resolved_refs = state.get("resolved_references", {})
            if resolved_refs.get("patient_name"):
                state["tool_parameters"]["patient_name"] = resolved_refs["patient_name"]
        else:
            state["selected_tool"] = "schedule_query"
            state["tool_parameters"] = {"doctor_id": state.get("doctor_id")}
    
    return state


def validate_sql_params(params: dict) -> str | list:
    """
    Validate required SQL parameters. Returns 'ok' if all present, else list of missing user-dependent keys.
    Auto-resolve fields are not considered missing for user prompt.
    """
    required_natural_fields = ["service_name", "patient_name", "appointment_date", "appointment_time"]
    missing = [field for field in required_natural_fields if not params.get(field)]
    return "ok" if not missing else missing

# ========== PRODUCTION VALIDATION FUNCTIONS ==========
"""
Production Validation Functions
-------------------------------
These functions provide production-ready validation logic for appointments,
including conflict checking, working hours validation, service availability,
and appointment timing validation. Used by the LLM-driven validation system.
"""

def validate_booking_conflicts(start_datetime: str, doctor_id: str, duration_minutes: int = 21) -> dict:
    """
    Check for appointment conflicts in the database.
    Returns validation result with conflict details if any.
    """
    try:
        from datetime import datetime, timedelta
        from ..tools.database import execute_query
        
        # Parse the start datetime
        start_dt = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        end_datetime = end_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Check for overlapping appointments
        conflict_query = """
        SELECT PatientName, StartDateTime, EndDateTime, ServiceName
        FROM View_Appointments 
        WHERE DoctorId = ? 
        AND Status = 'Scheduled'
        AND (
            (StartDateTime < ? AND EndDateTime > ?) OR
            (StartDateTime < ? AND EndDateTime > ?) OR
            (StartDateTime >= ? AND StartDateTime < ?)
        )
        """
        
        conflicts = execute_query(conflict_query, [
            doctor_id, 
            start_datetime, start_datetime,  # New appointment starts during existing
            end_datetime, end_datetime,      # New appointment ends during existing  
            start_datetime, end_datetime     # New appointment encompasses existing
        ])
        
        if conflicts:
            return {
                "valid": False,
                "error_type": "booking_conflict",
                "conflicts": conflicts,
                "message": f"Time slot {start_dt.strftime('%I:%M %p')} is already booked"
            }
        
        return {"valid": True}
        
    except Exception as e:
        logger.error(f"Booking conflict validation error: {e}")
        return {
            "valid": False, 
            "error_type": "validation_error",
            "message": f"Could not validate booking conflicts: {e}"
        }

def validate_working_hours(start_datetime: str, doctor_id: str) -> dict:
    """
    Validate appointment time against doctor's working hours and clinic schedule.
    Returns validation result with schedule details if invalid.
    """
    try:
        from datetime import datetime
        from ..tools.database import execute_query
        
        start_dt = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
        # Convert Python weekday (0=Monday) to database WeekDay (1=Monday)  
        weekday_num = start_dt.weekday() + 1  # Python: 0-6, DB: 1-7
        appointment_time = start_dt.time()
        
        # Check doctor's schedule for the specific day
        schedule_query = """
        SELECT FromTime, ToTime, IsActive
        FROM COR_DoctorSchedule
        WHERE DoctorId = ? AND WeekDay = ? AND IsActive = 1
        """
        
        schedule = execute_query(schedule_query, [doctor_id, weekday_num])
        
        if not schedule:
            weekday_name = start_dt.strftime('%A')
            return {
                "valid": False,
                "error_type": "no_schedule",
                "weekday": weekday_name,
                "weekday_num": weekday_num,
                "message": f"No schedule found for {weekday_name}"
            }
        
        doctor_schedule = schedule[0]
        
        # Parse working hours (remove microseconds if present)
        from_time_str = doctor_schedule['FromTime'].split('.')[0]
        to_time_str = doctor_schedule['ToTime'].split('.')[0]
        
        from_time = datetime.strptime(from_time_str, '%H:%M:%S').time()
        to_time = datetime.strptime(to_time_str, '%H:%M:%S').time()
        
        if not (from_time <= appointment_time <= to_time):
            return {
                "valid": False,
                "error_type": "outside_hours",
                "working_hours": {
                    "start": from_time.strftime('%I:%M %p'),
                    "end": to_time.strftime('%I:%M %p')
                },
                "requested_time": appointment_time.strftime('%I:%M %p'),
                "weekday": start_dt.strftime('%A'),
                "message": f"Appointment time {appointment_time.strftime('%I:%M %p')} is outside working hours"
            }
        
        # Check for doctor's off-schedule (vacation, sick days, etc.)
        off_schedule_query = """
        SELECT Date, Reason, IsOff
        FROM COR_DoctorOffSchedule
        WHERE DoctorId = ? 
        AND DATE(Date) = ?
        AND IsOff = 1
        AND IsActive = 1
        """
        
        appointment_date = start_dt.strftime('%Y-%m-%d')
        off_schedule = execute_query(off_schedule_query, [doctor_id, appointment_date])
        
        if off_schedule:
            reason = off_schedule[0].get('Reason') or 'unavailable'
            return {
                "valid": False,
                "error_type": "doctor_off_schedule",
                "off_schedule": off_schedule[0],
                "reason": reason,
                "message": f"Doctor is {reason} on {start_dt.strftime('%B %d, %Y')}"
            }
        
        return {"valid": True}
        
    except Exception as e:
        logger.error(f"Working hours validation error: {e}")
        return {
            "valid": False,
            "error_type": "validation_error", 
            "message": f"Could not validate working hours: {e}"
        }

def validate_appointment_time(start_datetime: str) -> dict:
    """
    Validate appointment is not in the past and is on a reasonable future date.
    Returns validation result with timing details if invalid.
    """
    try:
        from datetime import datetime, timedelta
        
        start_dt = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        
        # Check if appointment is in the past
        if start_dt <= now:
            return {
                "valid": False,
                "error_type": "past_appointment",
                "requested_time": start_dt,
                "current_time": now,
                "message": f"Cannot book appointments in the past"
            }
        
        # Check if appointment is too far in the future (e.g., more than 1 year)
        max_future = now + timedelta(days=365)
        if start_dt > max_future:
            return {
                "valid": False,
                "error_type": "too_far_future",
                "requested_time": start_dt,
                "max_date": max_future,
                "message": f"Cannot book appointments more than 1 year in advance"
            }
        
        return {"valid": True}
        
    except Exception as e:
        logger.error(f"Appointment time validation error: {e}")
        return {
            "valid": False,
            "error_type": "validation_error",
            "message": f"Could not validate appointment time: {e}"
        }

def validate_service_availability(service_name: str, doctor_id: str) -> dict:
    """
    Validate that the requested service is available and recognized.
    Returns validation result with service suggestions if invalid.
    """
    try:
        from ..tools.database import execute_query
        
        # Get all available services from existing appointments
        services_query = """
        SELECT DISTINCT ServiceName 
        FROM View_Appointments 
        WHERE ServiceName IS NOT NULL AND ServiceName != ''
        ORDER BY ServiceName
        """
        available_services = execute_query(services_query, [])
        
        if not available_services:
            return {
                "valid": False,
                "error_type": "no_services",
                "message": "No services are currently available"
            }
        
        # Normalize service name for comparison
        service_normalized = service_name.lower().strip()
        available_service_names = [s['ServiceName'].strip() for s in available_services]
        
        # Check exact match (case-insensitive)
        for service_data in available_services:
            service_db_name = service_data['ServiceName'].strip()
            if service_db_name.lower() == service_normalized:
                return {
                    "valid": True,
                    "service_name": service_db_name,
                    "normalized_service": service_db_name
                }
        
        # Check partial matches for suggestions
        partial_matches = []
        for service_data in available_services:
            service_db_name = service_data['ServiceName'].strip()
            service_lower = service_db_name.lower()
            if (service_normalized in service_lower or 
                service_lower in service_normalized or
                any(word in service_lower for word in service_normalized.split() if len(word) > 2)):
                partial_matches.append(service_db_name)
        
        return {
            "valid": False,
            "error_type": "service_not_found",
            "requested_service": service_name,
            "available_services": available_service_names,
            "suggested_services": partial_matches,
            "message": f"Service '{service_name}' not found"
        }
        
    except Exception as e:
        logger.error(f"Service validation error: {e}")
        return {
            "valid": False,
            "error_type": "validation_error",
            "message": f"Could not validate service: {e}"
        }

def generate_validation_error_response(validation_errors: list, state: AgentState, config: AgentConfig) -> str:
    """
    Use LLM to generate user-friendly error messages and suggestions for validation failures.
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        system_prompt = """You are a helpful medical appointment assistant. 
Generate user-friendly error messages and helpful suggestions when appointment bookings fail validation.
Be empathetic, clear, and provide actionable alternatives when possible.
Always address the doctor directly using "you" and maintain a professional tone."""
        
        context = {
            "validation_errors": validation_errors,
            "doctor_id": state.get("doctor_id"),
            "tool_parameters": state.get("tool_parameters", {}),
            "resolved_references": state.get("resolved_references", {})
        }
        
        prompt = f"""
The following appointment booking validation errors occurred:

{json.dumps(validation_errors, indent=2, default=str)}

Context:
{json.dumps(context, indent=2, default=str)}

Please generate a helpful response that:
1. Explains what went wrong in simple terms
2. Suggests specific alternatives or next steps
3. Maintains a professional, helpful tone
4. Addresses the doctor directly

Respond in JSON format:
{{
  "formatted_response": "Your helpful message here",
  "suggested_followups": ["suggestion 1", "suggestion 2"],
  "response_metadata": {{"validation_errors": true}}
}}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = config.llm.invoke(messages)
        result = safe_json_parse(response.content, "validation error response generator")
        
        return result.get("formatted_response", "Appointment booking validation failed. Please try again.")
        
    except Exception as e:
        logger.error(f"Error generating validation response: {e}")
        return "There was an issue with your appointment request. Please check the details and try again."

def run_appointment_validations(params: dict, state: AgentState, config: AgentConfig) -> dict:
    """
    Run all appointment validations and return consolidated results.
    """
    validation_errors = []
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    start_datetime = params.get("StartDateTime")
    doctor_id = params.get("doctor_id")
    service_name = params.get("service_name")
    
    if not start_datetime or not doctor_id:
        validation_results["valid"] = False
        validation_results["errors"].append({
            "error_type": "missing_required_data",
            "message": "Missing required appointment data for validation"
        })
        return validation_results
    
    # 1. Validate appointment timing (past/future)
    time_validation = validate_appointment_time(start_datetime)
    if not time_validation["valid"]:
        validation_results["valid"] = False
        validation_results["errors"].append(time_validation)
    
    # 2. Validate working hours and doctor availability
    hours_validation = validate_working_hours(start_datetime, doctor_id)
    if not hours_validation["valid"]:
        validation_results["valid"] = False
        validation_results["errors"].append(hours_validation)
    
    # 3. Validate booking conflicts
    conflict_validation = validate_booking_conflicts(start_datetime, doctor_id)
    if not conflict_validation["valid"]:
        validation_results["valid"] = False
        validation_results["errors"].append(conflict_validation)
    
    # 4. Validate service availability (if service provided)
    if service_name:
        service_validation = validate_service_availability(service_name, doctor_id)
        if not service_validation["valid"]:
            # Service validation is more flexible - can suggest alternatives
            if service_validation.get("suggested_services"):
                validation_results["warnings"].append(service_validation)
            else:
                validation_results["valid"] = False
                validation_results["errors"].append(service_validation)
    
    return validation_results

# ========== BACKEND LOOKUP NODES ==========
"""
Backend Lookup Nodes
--------------------
These nodes resolve backend/internal fields (IDs, branch info, status) using
database lookups and context. They handle data normalization, appointment
validation integration, and prepare data for SQL generation.
"""

def backend_lookup_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Backend Lookup Node with Production Validations
    -----------------------------------------------
    Resolves backend/internal fields (IDs, branch, status) using available context and lookup tools.
    Implements production-ready validations for booking appointments.
    This node is never exposed to the user and does not prompt for input.
    """
    params = state.get("tool_parameters", {})
    logger.info(f"Lookup received params: {params}")
    resolved = {**params, **state.get("resolved_references", {})}
    
    # Validate dates early to catch invalid dates like "February 30th"
    def validate_date_string(date_str, field_name):
        """Validate that a date string is a valid date"""
        if not date_str:
            return True  # Empty dates are handled elsewhere
        
        try:
            from datetime import datetime
            # Try to parse the date
            if len(date_str) == 10 and date_str.count('-') == 2:  # YYYY-MM-DD format
                datetime.strptime(date_str, "%Y-%m-%d")
            elif 'T' in date_str:  # ISO format
                datetime.fromisoformat(date_str)
            else:
                # Try various common formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
                    try:
                        datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Unable to parse date format: {date_str}")
            return True
        except ValueError as e:
            logger.error(f"Invalid date in {field_name}: {date_str} - {e}")
            error_msg = f"Invalid date '{date_str}' in {field_name}. Please provide a valid date."
            state.setdefault("errors", []).append(error_msg)
            state["slot_validation"] = {"status": "error", "fields": [field_name]}
            state["has_errors"] = True
            state["formatted_response"] = error_msg
            return False
    
    # Check all date fields in params and resolved references
    date_fields = ["date", "appointment_date", "appointment_datetime"]
    for field in date_fields:
        if field in params and not validate_date_string(params[field], field):
            return state
        if field in resolved and not validate_date_string(resolved[field], field):
            return state
    
    backend_fields = ["StatusId", "PatientId", "BranchName", "ServiceId", "BranchId"]
    debug_lookups = {}
    # Defensive: Always write to at least one field, even on error
    tool = state.get("selected_tool")
    if not tool:
        logger.error("No selected_tool in state for backend lookup.")
        state.setdefault("errors", []).append("No selected_tool in state for backend lookup.")
        state["slot_validation"] = {"status": "error", "fields": ["selected_tool"]}
        state["has_errors"] = True
        logger.debug(f"[backend_lookup_node] Early return: missing selected_tool. State keys: {list(state.keys())}")
        return state
    if not params:
        # For some tools, empty params is acceptable - they use context instead
        if tool in ["appointment_lookup", "calendar_summary", "doctor_availability"]:
            logger.info(f"[backend_lookup_node] Tool {tool} has empty params, using context-based lookup")
            params = {}  # Initialize empty params dict
        else:
            logger.error("No tool_parameters in state for backend lookup.")
            state.setdefault("errors", []).append("No tool_parameters in state for backend lookup.")
            state["slot_validation"] = {"status": "error", "fields": ["tool_parameters"]}
            state["has_errors"] = True
            logger.debug(f"[backend_lookup_node] Early return: missing tool_parameters. State keys: {list(state.keys())}")
            return state
    logger.debug(f"[backend_lookup_node] Tool: {tool}")
    if tool == "appointment_booking":
        # --- Datetime Handling ---
        # 1. Handle appointment_datetime in ISO format (2025-04-14T10:45:00)
        if "appointment_datetime" in params and params["appointment_datetime"]:
            dt_str = params["appointment_datetime"]
            # Detect ISO format with 'T'
            if "T" in dt_str:
                try:
                    from datetime import datetime
                    dt_obj = datetime.fromisoformat(dt_str)
                    formatted = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    params["StartDateTime"] = formatted
                    resolved["StartDateTime"] = formatted
                    logger.info(f"Reformatted appointment_datetime from ISO to SQL format: {formatted}")
                except Exception as e:
                    logger.warning(f"Could not parse appointment_datetime: {e}")
            else:
                # If already in correct format, just copy
                params["StartDateTime"] = dt_str
                resolved["StartDateTime"] = dt_str
        # 2. Handle separate date/time fields
        elif params.get("date") and params.get("time"):
            try:
                from datetime import datetime
                dt = datetime.strptime(f"{params['date']} {params['time']}", "%Y-%m-%d %H:%M")
                formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
                params["StartDateTime"] = formatted
                resolved["StartDateTime"] = formatted
                logger.info(f"Combined date and time to StartDateTime: {formatted}")
            except Exception as e:
                logger.warning(f"Could not parse date/time: {e}")
        
        # 3. Handle earliest slot booking (find_earliest=True)
        elif params.get("find_earliest") and params.get("doctor_id") and params.get("appointment_date"):
            try:
                from datetime import datetime
                from ..tools.database import schedule_query
                # Get available slots for the date
                result = schedule_query.invoke({
                    "doctor_id": params["doctor_id"],
                    "date": params["appointment_date"],
                    "service_name": params.get("service_name"),
                    "suggest_slots": False,
                    "find_earliest": True
                })
                
                if result.get("success") and result.get("available_slots"):
                    # Take the first (earliest) slot
                    earliest_slot_str = result["available_slots"][0]  # Format: "10:00 - 10:21"
                    earliest_time = earliest_slot_str.split(" - ")[0]  # Get "10:00"
                    
                    # Combine date and earliest time
                    dt = datetime.strptime(f"{params['appointment_date']} {earliest_time}", "%Y-%m-%d %H:%M")
                    formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
                    params["StartDateTime"] = formatted
                    resolved["StartDateTime"] = formatted
                    logger.info(f"Found earliest slot: {earliest_time} -> StartDateTime: {formatted}")
                else:
                    logger.warning(f"Could not find earliest slot for doctor {params['doctor_id']} on {params['appointment_date']}")
            except Exception as e:
                logger.warning(f"Could not resolve earliest slot: {e}")
        
        # Remove appointment_date and appointment_time keys if present
        for k in ["appointment_date", "appointment_time"]:
            if k in params:
                del params[k]

        # --- PatientId Lookup (robust: get_or_create_patient_id) ---
        if "patient_name" in params and params["patient_name"]:
            try:
                from ..tools.database import get_or_create_patient_id
                patient_id = get_or_create_patient_id(params["patient_name"])
                params["PatientId"] = patient_id
                resolved["PatientId"] = patient_id
                debug_lookups["PatientId"] = patient_id
                logger.info(f"Resolved PatientId for {params['patient_name']}: {patient_id}")
            except Exception as e:
                import traceback
                logger.error(f"Exception in get_or_create_patient_id for {params['patient_name']}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                params["PatientId"] = None
                resolved["PatientId"] = None
                logger.warning(f"Could not resolve PatientId for {params['patient_name']}: {e}")
        else:
            params["PatientId"] = None
            resolved["PatientId"] = None
            logger.warning("Missing patient_name for PatientId lookup.")

        # --- ServiceId Lookup ---
        service_info = None
        if "service_name" in params and params["service_name"]:
            service_info = get_service_id_and_duration(params["service_name"])
            if service_info and "service_id" in service_info:
                params["ServiceId"] = service_info["service_id"]
                resolved["ServiceId"] = service_info["service_id"]
                debug_lookups["ServiceId"] = service_info["service_id"]
                
                # IMPORTANT: Use canonical service name from database if available
                if "service_name" in service_info:
                    params["service_name"] = service_info["service_name"]  # Replace user input with canonical form
                    logger.info(f"Normalized service name to canonical form: {service_info['service_name']}")
                
                logger.info(f"Resolved ServiceId for {params['service_name']}: {service_info['service_id']}")
            else:
                params["ServiceId"] = None
                resolved["ServiceId"] = None
                logger.warning(f"Could not resolve ServiceId for {params['service_name']}")
        else:
            params["ServiceId"] = None
            resolved["ServiceId"] = None
            logger.warning("Missing service_name for ServiceId lookup.")

        # --- BranchId and BranchName Lookup ---
        branch_info = None
        if "doctor_id" in params and params["doctor_id"]:
            branch_info = get_doctor_default_branch(params["doctor_id"])
            if branch_info:
                params["BranchName"] = branch_info[0]
                params["BranchId"] = branch_info[1]
                resolved["BranchName"] = branch_info[0]
                resolved["BranchId"] = branch_info[1]
                debug_lookups["BranchName"] = branch_info[0]
                debug_lookups["BranchId"] = branch_info[1]
                logger.info(f"Resolved Branch info for doctor_id={params['doctor_id']}: {branch_info}")
            else:
                params["BranchName"] = None
                params["BranchId"] = None
                resolved["BranchName"] = None
                resolved["BranchId"] = None
                logger.warning(f"Could not resolve Branch info for doctor_id={params['doctor_id']}")
        else:
            params["BranchName"] = None
            params["BranchId"] = None
            resolved["BranchName"] = None
            resolved["BranchId"] = None
            logger.warning("Missing doctor_id for Branch lookup.")

        # --- DoctorName Lookup (always resolve from doctor_id using View_Appointments) ---
        doctor_name = None
        if params.get("doctor_id"):
            try:
                from ..tools.database import execute_query
                results = execute_query("SELECT DoctorName FROM View_Appointments WHERE DoctorId = ? AND DoctorName IS NOT NULL AND DoctorName != '' LIMIT 1", (params["doctor_id"],))
                if results and results[0].get("DoctorName"):
                    doctor_name = results[0]["DoctorName"]
                    params["DoctorName"] = doctor_name
                    resolved["DoctorName"] = doctor_name
                    debug_lookups["DoctorName"] = doctor_name
                    logger.info(f"Resolved DoctorName for doctor_id={params['doctor_id']}: {doctor_name}")
                else:
                    logger.warning(f"Could not resolve DoctorName for doctor_id={params['doctor_id']}, no matching entry in View_Appointments.")
            except Exception as e:
                logger.warning(f"Could not resolve DoctorName for doctor_id={params.get('doctor_id')}: {e}")
            # Only set Unknown Doctor if not resolved
            if not doctor_name:
                params["DoctorName"] = "Unknown Doctor"
                resolved["DoctorName"] = "Unknown Doctor"
                debug_lookups["DoctorName"] = "Unknown Doctor"
        else:
            params["DoctorName"] = "Unknown Doctor"
            resolved["DoctorName"] = "Unknown Doctor"
            debug_lookups["DoctorName"] = "Unknown Doctor"
            logger.warning("Missing doctor_id for DoctorName lookup, set as 'Unknown Doctor'")

        # --- StatusId (default to 1) ---
        params["StatusId"] = 1
        resolved["StatusId"] = 1
        debug_lookups["StatusId"] = 1

        # --- EndDateTime Calculation (enforce no .0 at end) ---
        if params.get("StartDateTime") and service_info and service_info.get("duration"):
            try:
                start_dt = datetime.strptime(params["StartDateTime"].split(".")[0], "%Y-%m-%d %H:%M:%S")
                end_dt = start_dt + timedelta(minutes=int(service_info["duration"]))
                end_formatted = end_dt.strftime("%Y-%m-%d %H:%M:%S")
                params["EndDateTime"] = end_formatted
                resolved["EndDateTime"] = end_formatted
                logger.info(f"Calculated EndDateTime: {params['EndDateTime']}")
            except Exception as e:
                params["EndDateTime"] = None
                resolved["EndDateTime"] = None
                logger.warning(f"Could not parse StartDateTime for end time calculation: {e}")
        else:
            params["EndDateTime"] = None
            resolved["EndDateTime"] = None

        # ========================================
        # PRODUCTION APPOINTMENT VALIDATIONS
        # ========================================
        
        # Run comprehensive validations for appointment booking
        if params.get("StartDateTime") and params.get("doctor_id"):
            logger.info("🧠 Starting LLM-driven appointment validation planning")
            
            # Import and use LLM validation planner
            from .llm_validation_planner import llm_validation_planner_node
            
            # Let LLM plan and execute validations
            validation_state = llm_validation_planner_node({
                **state,
                'tool_parameters': params
            }, config)
            
            validation_results = validation_state.get('validation_results', {})
            
            if not validation_results.get("valid", True):
                logger.warning(f"❌ LLM-planned validation failed: {len(validation_results.get('errors', []))} errors")
                
                # Generate user-friendly error response using LLM
                error_response = generate_validation_error_response(
                    validation_results.get("errors", []), state, config
                )
                
                # Set validation failure state
                state["slot_validation"] = {
                    "status": "validation_failed", 
                    "fields": ["appointment_constraints"],
                    "validation_errors": validation_results.get("errors", [])
                }
                state["has_errors"] = True
                state["formatted_response"] = error_response
                state.setdefault("errors", []).extend([err.get("message", "Validation error") for err in validation_results.get("errors", [])])
                
                # Add validation metadata
                if "response_metadata" not in state:
                    state["response_metadata"] = {}
                state["response_metadata"]["validation_errors"] = validation_results.get("errors", [])
                state["response_metadata"]["validation_failed"] = True
                state["response_metadata"]["llm_planned_validation"] = validation_state.get('llm_planned_validation', False)
                state["response_metadata"]["validation_plan"] = validation_state.get('validation_plan')
                
                logger.info(f"🚫 Blocking appointment booking due to LLM validation failures")
                return state
            
            elif validation_results.get("warnings"):
                logger.info(f"⚠️ LLM validation completed with {len(validation_results['warnings'])} warnings")
                
                # Handle warnings (e.g., service suggestions) but don't block booking
                if "response_metadata" not in state:
                    state["response_metadata"] = {}
                state["response_metadata"]["validation_warnings"] = validation_results["warnings"]
                state["response_metadata"]["llm_planned_validation"] = validation_state.get('llm_planned_validation', False)
                
                # For service warnings, update service information if suggestions available
                for warning in validation_results["warnings"]:
                    if warning.get("error_type") == "service_not_found" and warning.get("suggested_services"):
                        logger.info(f"💡 Service suggestions available: {warning['suggested_services']}")
                        state["response_metadata"]["suggested_services"] = warning["suggested_services"]
                        
            else:
                logger.info("✅ All LLM-planned appointment validations passed")
                # Add metadata for successful LLM validation
                if "response_metadata" not in state:
                    state["response_metadata"] = {}
                state["response_metadata"]["llm_planned_validation"] = validation_state.get('llm_planned_validation', False)
                state["response_metadata"]["validation_plan"] = validation_state.get('validation_plan')
                
        else:
            logger.warning("⚠️ Skipping validations - missing StartDateTime or doctor_id")

        # --- Handle "earliest slot" requests ---
        needs_earliest_slot = resolved.get("needs_earliest_slot", False)
        if needs_earliest_slot and params.get("doctor_id") and params.get("date"):
            try:
                from ..tools.database import get_earliest_available_slot
                
                # Get service duration for slot calculation
                service_duration = 21  # default
                if service_info and service_info.get("duration"):
                    service_duration = int(service_info["duration"])
                
                earliest_slot = get_earliest_available_slot(
                    doctor_id=int(params["doctor_id"]), 
                    date=params["date"],
                    service_duration_minutes=service_duration
                )
                
                if earliest_slot:
                    # Update time parameters with earliest available slot
                    params["time"] = earliest_slot["start_time"] 
                    params["start_time"] = earliest_slot["start_time"]
                    params["StartDateTime"] = earliest_slot["start_datetime"]
                    params["EndDateTime"] = earliest_slot["end_datetime"]
                    resolved["time"] = earliest_slot["start_time"]
                    resolved["start_time"] = earliest_slot["start_time"]
                    resolved["StartDateTime"] = earliest_slot["start_datetime"]
                    resolved["EndDateTime"] = earliest_slot["end_datetime"]
                    
                    logger.info(f"Found earliest available slot: {earliest_slot['start_time']} on {params['date']}")
                else:
                    # No slots available
                    error_msg = f"No available slots found for doctor {params['doctor_id']} on {params['date']}"
                    logger.warning(error_msg)
                    state.setdefault("errors", []).append(error_msg)
                    state["slot_validation"] = {"status": "no_slots_available", "fields": ["time"]}
                    state["has_errors"] = True
                    return state
                    
            except Exception as e:
                logger.warning(f"Error finding earliest slot: {e}")
                # Continue with original logic if slot finding fails

        # --- Log all resolved and missing fields ---
        still_missing = [f for f in ["PatientId", "ServiceId", "BranchId", "BranchName"] if not params.get(f)]
        logger.info(f"Backend lookup completed. Resolved: {debug_lookups}, Still missing: {still_missing}")
        
        # --- Normalize data fields to match database formatting ---
        if not still_missing:  # Only normalize if all required fields are present
            try:
                from ..tools.database import normalize_appointment_data
                logger.info("Normalizing appointment data to match database format...")
                normalized_params = normalize_appointment_data(params)
                state["tool_parameters"] = normalized_params
                logger.info(f"Data normalization completed. Sample changes: ServiceName={params.get('service_name')} → {normalized_params.get('ServiceName')}")
            except Exception as e:
                logger.warning(f"Data normalization failed: {e}, proceeding with original data")
                state["tool_parameters"] = params
        else:
            state["tool_parameters"] = params
        
        if still_missing:
            logger.warning(f"Backend lookup could not resolve: {still_missing}")
            state["slot_validation"] = {"status": "missing_backend", "fields": still_missing}
            # Error handling for missing backend fields
            err_msg = f"Backend lookup could not resolve: {', '.join(still_missing)}"
            state.setdefault("errors", []).append(err_msg)
            if "response_metadata" not in state:
                state["response_metadata"] = {}
            if "errors" not in state["response_metadata"]:
                state["response_metadata"]["errors"] = []
            state["response_metadata"]["errors"].append(err_msg)
            state["has_errors"] = True
        else:
            state["slot_validation"] = {"status": "ok", "fields": []}
            state["has_errors"] = False

        # --- Update state ---
        # Note: tool_parameters already updated above with normalized data
        state["resolved_references"] = resolved
        state["required_lookups"] = []
        backend_logs = {
            "tool": state.get("selected_tool"),
            "resolved_lookups": debug_lookups,
            "still_missing": still_missing,
            "final_params": state["tool_parameters"]  # Use the normalized parameters
        }
        state.setdefault("response_metadata", {})["backend_logs"] = backend_logs
        state.setdefault("mcp_context", {})["backend_logs"] = backend_logs
        logger.info(f"Backend lookup node result: params={params}, debug={backend_logs}")
        print(f"[DEBUG] backend_lookup_node slot_validation: {state.get('slot_validation')}")
        logger.debug(f"Final tool parameters for SQL generation: {json.dumps(params, indent=2)}")
        logger.debug(f"[backend_lookup_node] Returning for appointment_booking. State keys: {list(state.keys())}")
        logger.debug(f"[backend_lookup_node] slot_validation: {state.get('slot_validation')}, errors: {state.get('errors')}")
        return state
    
    elif tool == "appointment_lookup":
        # --- Appointment Lookup: Handle doctor and patient appointment queries ---
        user_role = state.get("user_role", "user")
        doctor_id = state.get("doctor_id") or params.get("doctor_id")
        patient_name = params.get("patient_name")
        appointment_date = params.get("appointment_date") or params.get("date")
        query_intent = state.get("query_intent")
        
        logger.info(f"[appointment_lookup] Processing: user_role={user_role}, doctor_id={doctor_id}, patient_name={patient_name}, date={appointment_date}")
        
        # For doctor queries about their own appointments, use doctor_id from context
        if user_role.lower() == "doctor" and doctor_id:
            params["doctor_id"] = doctor_id
            # Doctor queries typically don't need additional fields
            if query_intent in ["next_patient", "patient_lookup"]:
                # Looking for next patient appointment
                params["lookup_type"] = "next_patient"
            elif query_intent in ["schedule", "daily_schedule"]:
                # Looking for today's schedule
                params["lookup_type"] = "daily_schedule"
                if not appointment_date:
                    from datetime import datetime
                    params["appointment_date"] = datetime.now().strftime("%Y-%m-%d")
            else:
                params["lookup_type"] = "general_appointment"
                
            state["tool_parameters"] = params
            state["slot_validation"] = {"status": "backend_complete", "fields": []}
            state["has_errors"] = False
            logger.info(f"Backend lookup completed for appointment_lookup (doctor): {params}")
            return state
            
        # For patient-specific lookups, ensure we have necessary information
        elif patient_name or appointment_date:
            # Patient or staff looking up specific appointments
            if patient_name:
                # Look up patient ID
                try:
                    from ..tools.database import get_or_create_patient_id
                    patient_id = get_or_create_patient_id(patient_name)
                    if patient_id:
                        params["PatientId"] = patient_id
                        params["patient_id"] = patient_id
                        logger.info(f"Resolved PatientId for {patient_name}: {patient_id}")
                    else:
                        logger.warning(f"Could not find patient: {patient_name}")
                        state.setdefault("errors", []).append(f"Patient '{patient_name}' not found")
                        state["slot_validation"] = {"status": "error", "fields": ["patient_name"]}
                        state["has_errors"] = True
                        return state
                except Exception as e:
                    logger.error(f"Error looking up patient {patient_name}: {e}")
                    state.setdefault("errors", []).append(f"Error looking up patient: {e}")
                    state["slot_validation"] = {"status": "error", "fields": ["patient_name"]}
                    state["has_errors"] = True
                    return state
            
            params["lookup_type"] = "specific_appointment"
            state["tool_parameters"] = params
            state["slot_validation"] = {"status": "backend_complete", "fields": []}
            state["has_errors"] = False
            logger.info(f"Backend lookup completed for appointment_lookup (patient): {params}")
            return state
        
        else:
            # Missing required information
            missing_fields = []
            if user_role.lower() != "doctor":
                if not patient_name:
                    missing_fields.append("patient_name")
                if not appointment_date:
                    missing_fields.append("appointment_date")
            
            if missing_fields:
                logger.error(f"Backend lookup for appointment_lookup missing fields: {missing_fields}")
                state.setdefault("errors", []).append(f"Backend lookup for appointment_lookup missing fields: {', '.join(missing_fields)}")
                state["slot_validation"] = {"status": "missing_backend", "fields": missing_fields}
                state["has_errors"] = True
                return state
        
        # Fallback - should not reach here
        state["tool_parameters"] = params
        state["slot_validation"] = {"status": "backend_complete", "fields": []}
        state["has_errors"] = False
        return state
    
    elif tool == "schedule_query":
        # --- Schedule Query: Handle slot suggestions and earliest slot finding ---
        doctor_id = params.get("doctor_id")
        
        # Enhanced date resolution from multiple sources
        date = (params.get("date") or 
                params.get("appointment_date") or 
                state.get("resolved_references", {}).get("appointment_date") or
                state.get("resolved_references", {}).get("date"))
        
        # Parse datetime if needed
        if date and "T" in str(date):
            # Extract date part from datetime string like "2025-08-25T15:00:00"
            date = str(date).split("T")[0]
            
        find_earliest = params.get("find_earliest", False)
        
        # More intelligent date resolution from query context
        if not date:
            # Try to extract from common date mappings in resolved_references
            resolved_refs = state.get("resolved_references", {})
            for key, value in resolved_refs.items():
                if key in ["tomorrow", "today", "next week", "next week tuesday"] and value:
                    date = value
                    logger.info(f"Resolved date from '{key}': {date}")
                    break
        
        # Validate required fields are present
        missing_fields = []
        if not doctor_id:
            missing_fields.append("doctor_id")
        if not date:
            missing_fields.append("date")
        
        if missing_fields:
            logger.error(f"Backend lookup for schedule_query missing fields: {missing_fields}")
            state.setdefault("errors", []).append(f"Backend lookup for schedule_query missing fields: {', '.join(missing_fields)}")
            state["slot_validation"] = {"status": "missing_backend", "fields": missing_fields}
            state["has_errors"] = True
        else:
            # Ensure standardized parameter names
            # Keep doctor_id as string (e.g., "DR001", "DR002")
            params["doctor_id"] = doctor_id
            params["date"] = date
            
            # If this is an earliest slot request for booking, prepare for potential auto-booking
            slot_preference = state.get("resolved_references", {}).get("slot_preference")
            if slot_preference == "earliest" and state.get("query_intent") == "book_appointment":
                logger.info(f"Preparing for earliest slot auto-booking: doctor_id={doctor_id}, date={date}")
                params["find_earliest"] = True
                # Store booking context for post-schedule processing
                state["earliest_slot_booking_context"] = {
                    "patient_name": state.get("resolved_references", {}).get("patient_name"),
                    "service_name": state.get("resolved_references", {}).get("service_name"),
                    "original_intent": "book_appointment"
                }
            
            state["tool_parameters"] = params
            state["slot_validation"] = {"status": "backend_complete", "fields": []}
            state["has_errors"] = False
            logger.info(f"Backend lookup completed for schedule_query: doctor_id={doctor_id}, date={date}, find_earliest={params.get('find_earliest', False)}")
        
        return state
    
    elif tool == "doctor_availability":
        # --- Doctor Availability: Check doctor schedule and availability ---
        doctor_id = params.get("doctor_id") or state.get("doctor_id")
        date = params.get("date") or params.get("appointment_date")
        
        # If no doctor specified, this might be a general availability query
        if not doctor_id:
            # Try to extract from query or context
            query = state.get("current_query", "").lower()
            if "dr " in query or "doctor " in query:
                # Could extract doctor name from query, but for now set as general query
                params["lookup_type"] = "general_availability"
            else:
                params["lookup_type"] = "general_availability"
        else:
            params["doctor_id"] = doctor_id
            params["lookup_type"] = "specific_doctor_availability"
        
        if date:
            params["date"] = date
        
        state["tool_parameters"] = params
        state["slot_validation"] = {"status": "backend_complete", "fields": []}
        state["has_errors"] = False
        logger.info(f"Backend lookup completed for doctor_availability: {params}")
        return state
    
    elif tool == "calendar_summary":
        # --- Calendar Summary: Summarize appointments for doctor or date range ---
        user_role = state.get("user_role", "user")
        doctor_id = state.get("doctor_id") or params.get("doctor_id")
        date = params.get("date") or params.get("appointment_date")
        
        if user_role.lower() == "doctor" and doctor_id:
            params["doctor_id"] = doctor_id
            params["lookup_type"] = "doctor_calendar"
        
        if not date:
            # Default to today for calendar summary
            from datetime import datetime
            params["date"] = datetime.now().strftime("%Y-%m-%d")
        
        state["tool_parameters"] = params
        state["slot_validation"] = {"status": "backend_complete", "fields": []}
        state["has_errors"] = False
        logger.info(f"Backend lookup completed for calendar_summary: {params}")
        return state
    
    elif tool == "patient_history":
        # --- Patient History: Look up patient medical/appointment history ---
        patient_name = params.get("patient_name")
        
        if patient_name:
            # Look up patient ID
            try:
                from ..tools.database import get_or_create_patient_id
                patient_id = get_or_create_patient_id(patient_name)
                if patient_id:
                    params["PatientId"] = patient_id
                    params["patient_id"] = patient_id
                    params["lookup_type"] = "patient_history"
                    logger.info(f"Resolved PatientId for history lookup {patient_name}: {patient_id}")
                else:
                    logger.warning(f"Could not find patient for history: {patient_name}")
                    state.setdefault("errors", []).append(f"Patient '{patient_name}' not found")
                    state["slot_validation"] = {"status": "error", "fields": ["patient_name"]}
                    state["has_errors"] = True
                    return state
            except Exception as e:
                logger.error(f"Error looking up patient history for {patient_name}: {e}")
                state.setdefault("errors", []).append(f"Error looking up patient: {e}")
                state["slot_validation"] = {"status": "error", "fields": ["patient_name"]}
                state["has_errors"] = True
                return state
        else:
            # Missing patient name
            logger.error("Backend lookup for patient_history missing patient_name")
            state.setdefault("errors", []).append("Backend lookup for patient_history missing patient_name")
            state["slot_validation"] = {"status": "missing_backend", "fields": ["patient_name"]}
            state["has_errors"] = True
            return state
        
        state["tool_parameters"] = params
        state["slot_validation"] = {"status": "backend_complete", "fields": []}
        state["has_errors"] = False
        logger.info(f"Backend lookup completed for patient_history: {params}")
        return state
    
    elif tool == "appointment_rescheduling":
        # --- Appointment Rescheduling: Find existing appointment to reschedule ---
        patient_name = params.get("patient_name")
        doctor_id = params.get("doctor_id") or state.get("doctor_id")
        appointment_date = params.get("appointment_date")
        service_name = params.get("service_name")
        
        if not patient_name:
            logger.error("Backend lookup for appointment_rescheduling missing patient_name")
            state.setdefault("errors", []).append("Patient name is required for rescheduling")
            state["slot_validation"] = {"status": "missing_backend", "fields": ["patient_name"]}
            state["has_errors"] = True
            return state
        
        # Find the existing appointment to reschedule
        try:
            from ..tools.database import find_appointment_for_rescheduling
            from datetime import datetime
            
            # Look for the appointment by patient name, doctor, and service
            # For rescheduling, we want to find existing appointment, not search by new date
            # Try first without date restriction to find any upcoming appointment
            appointment_info = find_appointment_for_rescheduling(
                patient_name=patient_name,
                doctor_id=doctor_id,
                service_name=service_name,
                current_date=None  # Don't restrict by date - find any existing appointment
            )
            
            # If no appointment found and we have a specific service, try without service filter
            if not appointment_info and service_name:
                appointment_info = find_appointment_for_rescheduling(
                    patient_name=patient_name,
                    doctor_id=doctor_id,
                    service_name=None,
                    current_date=None
                )
            
            # If still no appointment, try looking at today's appointments
            if not appointment_info:
                today = datetime.now().strftime('%Y-%m-%d')
                appointment_info = find_appointment_for_rescheduling(
                    patient_name=patient_name,
                    doctor_id=doctor_id,
                    service_name=service_name,
                    current_date=today
                )
            
            if appointment_info:
                # Check for RBAC error
                if appointment_info.get("error") == "access_denied":
                    logger.warning(f"RBAC violation: {appointment_info['message']}")
                    return {
                        "tool_parameters": params,
                        "status": "rbac_denied",
                        "error_message": appointment_info["message"],
                        "rbac_error": True
                    }
                
                params["appointment_id"] = appointment_info["id"]
                params["current_appointment"] = appointment_info
                logger.info(f"Found appointment to reschedule: ID {appointment_info['id']} for {patient_name}")
            else:
                logger.warning(f"Could not find appointment to reschedule for {patient_name}")
                # Don't fail here - let the tool function handle the lookup with more flexible criteria
        except Exception as e:
            logger.warning(f"Error finding appointment for rescheduling: {e}")
            # Don't fail - let the tool function handle it
        
        state["tool_parameters"] = params
        state["slot_validation"] = {"status": "backend_complete", "fields": []}
        state["has_errors"] = False
        logger.info(f"Backend lookup completed for appointment_rescheduling: {params}")
        return state
    
    elif tool == "appointment_cancellation":
        # --- Appointment Cancellation: Find existing appointment to cancel ---
        patient_name = params.get("patient_name")
        doctor_id = params.get("doctor_id") or state.get("doctor_id")
        appointment_date = params.get("appointment_date")
        service_name = params.get("service_name")
        
        if not patient_name:
            logger.error("Backend lookup for appointment_cancellation missing patient_name")
            state.setdefault("errors", []).append("Patient name is required for cancellation")
            state["slot_validation"] = {"status": "missing_backend", "fields": ["patient_name"]}
            state["has_errors"] = True
            return state
        
        # Find the existing appointment to cancel
        try:
            from ..tools.database import find_appointment_for_cancellation
            # Look for the appointment by patient name, doctor, and optionally service/date
            appointment_info = find_appointment_for_cancellation(
                patient_name=patient_name,
                doctor_id=doctor_id,
                service_name=service_name,
                current_date=appointment_date
            )
            
            if appointment_info:
                # Check for RBAC error
                if appointment_info.get("error") == "access_denied":
                    logger.warning(f"RBAC violation: {appointment_info['message']}")
                    return {
                        "tool_parameters": params,
                        "status": "rbac_denied",
                        "error_message": appointment_info["message"],
                        "rbac_error": True
                    }
                
                params["appointment_id"] = appointment_info["id"]
                params["current_appointment"] = appointment_info
                logger.info(f"Found appointment to cancel: ID {appointment_info['id']} for {patient_name}")
            else:
                logger.warning(f"Could not find appointment to cancel for {patient_name}")
                # Don't fail here - let the tool function handle the lookup with more flexible criteria
        except Exception as e:
            logger.warning(f"Error finding appointment for cancellation: {e}")
            # Don't fail - let the tool function handle it
        
        state["tool_parameters"] = params
        state["slot_validation"] = {"status": "backend_complete", "fields": []}
        state["has_errors"] = False
        logger.info(f"Backend lookup completed for appointment_cancellation: {params}")
        return state
    
    # --- Other tools: Only perform lookups if implemented, else set explicit error ---
    logger.error(f"[backend_lookup_node] No backend lookup implemented for tool: {tool}")
    state.setdefault("errors", []).append(f"No backend lookup implemented for tool: {tool}")
    state["slot_validation"] = {"status": "error", "fields": [tool]}
    state["has_errors"] = True
    logger.debug(f"[backend_lookup_node] No backend lookup for tool: {tool}. State keys: {list(state.keys())}")
    logger.debug(f"[backend_lookup_node] slot_validation: {state.get('slot_validation')}, errors: {state.get('errors')}")
    return state

# ========== MEMORY MANAGEMENT NODES ==========
"""
Memory Management Nodes
-----------------------
These nodes update conversation memory and context for future reference using
standardized MCP (Model Context Protocol). They handle context preservation,
reference tracking, and conversation continuity across multiple interactions.
"""

def memory_manager_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Memory Manager Node
    ------------------
    Updates conversation memory and context for future reference using standardized MCP (Model Context Protocol).
    This node is critical for context preservation and reference resolution.
    """
    # =========================
    # Experimental / Under Development Blocks
    # =========================
    # If you see large commented-out or half-integrated blocks below, they are likely
    # experimental, legacy, or under development. Do not remove unless confirmed dead code.
    # =========================
    # Enhanced memory manager with MCP context storage.
    """
    Enhanced memory manager with MCP context storage.
    """
    logger.info("Memory manager updating conversation state with MCP integration")
    session_id = state.get("session_id", "default_session")
    try:
        # Add current interaction to traditional memory
        state = add_to_conversation_memory(
            state,
            query=state["current_query"],
            result=state["tool_results"],
            flow_step=f"{state['query_intent']}:{state['selected_tool']}"
        )
        # Enhanced MCP context storage based on query intent
        # Store patient/appointment context for both next_patient and time_specific_lookup
        if state["query_intent"] in ["next_patient", "time_specific_lookup"] and state["tool_results"]:
            appointment = state["tool_results"][0] if state["tool_results"] else None
            if appointment and isinstance(appointment, dict):
                state["conversation_memory"]["implicit_references"]["current_patient"] = {
                    "name": appointment.get("PatientName"),
                    "id": appointment.get("PatientID"),
                    "appointment_id": appointment.get("AppointmentID"),
                    "start_time": appointment.get("StartDateTime"),
                    "timestamp": datetime.now().isoformat()
                }
                patient_context_id = mcp_context_manager.add_patient_context(
                    patient_name=appointment.get("PatientName", "Unknown"),
                    patient_id=str(appointment.get("PatientID", "")),
                    appointment_details={
                        "appointment_id": appointment.get("AppointmentID"),
                        "start_time": appointment.get("StartDateTime"),
                        "end_time": appointment.get("EndDateTime"),
                        "appointment_type": appointment.get("AppointmentType"),
                        "status": appointment.get("Status")
                    },
                    session_id=session_id
                )
                appointment_context_id = mcp_context_manager.add_appointment_context(
                    query_intent=state["query_intent"],
                    appointments=state["tool_results"],
                    session_id=session_id
                )
                logger.info(f"Added MCP contexts: patient={patient_context_id}, appointment={appointment_context_id}")
                state = update_patient_context(
                    state,
                    patient_id=appointment.get("PatientID"),
                    patient_name=appointment.get("PatientName"),
                    appointment_id=appointment.get("AppointmentID"),
                    appointment_date=appointment.get("StartDateTime")
                )
                
        if state["query_intent"] == "schedule" and state["tool_results"]:
            schedule_context_id = mcp_context_manager.add_schedule_context(
                schedule_data=state["tool_results"],
                date=datetime.now().strftime("%Y-%m-%d"),
                session_id=session_id
            )
            logger.info(f"Added MCP schedule context: {schedule_context_id}")
        if state["query_intent"] == "patient_history" and state["tool_results"]:
            if state["resolved_references"].get("patient_name"):
                patient_name = state["resolved_references"]["patient_name"]
                patient_context_id = mcp_context_manager.add_patient_context(
                    patient_name=patient_name,
                    patient_id=state["resolved_references"].get("patient_id", ""),
                    appointment_details={
                        "history_query": True,
                        "results_count": len(state["tool_results"]),
                        "query_time": datetime.now().isoformat()
                    },
                    session_id=session_id
                )
                logger.info(f"Added MCP patient history context: {patient_context_id}")
        # Update doctor context with recent activity
        if state.get("doctor_context") and state["tool_results"]:
            state["doctor_context"]["last_queried_date"] = datetime.now().strftime("%Y-%m-%d")
            if state["query_intent"] in ["schedule", "next_patient"]:
                state["doctor_context"]["current_appointments"] = state["tool_results"][:5]
        # Add response to message history for LangGraph
        if state.get("formatted_response"):
            state["messages"].append(AIMessage(content=state["formatted_response"]))
        # Log MCP context summary for debugging
        mcp_summary = mcp_context_manager.get_context_summary(session_id)
        logger.info(f"MCP context summary: {mcp_summary['total_items']} items, types: {mcp_summary['context_types']}")
        logger.info("Memory updated successfully with MCP integration")
    except Exception as e:
        logger.error(f"Memory management error: {e}")
        state.setdefault("errors", []).append(f"Memory update failed: {e}")
    logger.info(f"Slot validator updated tool_parameters: {state['tool_parameters']}")
    return state