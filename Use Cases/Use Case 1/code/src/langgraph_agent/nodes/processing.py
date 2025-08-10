"""
LangGraph Medical Agent Processing Nodes
========================================
This module contains the core node logic for the LangGraph medical assistant agent.
It implements dynamic slot-filling, LLM-driven reasoning, SQL generation, backend lookups,
context-aware user prompting, and memory/context management. All logic is designed to be
schema-agnostic and robust for production use. Do not remove or rewrite core functions unless absolutely necessary.

Key responsibilities:
- Tool execution and SQL query handling
- Slot validation and backend lookups
- Context resolution and memory management
- LLM-based response formatting

Author: [Your Team/Name]
"""

# ========== Standard Library Imports ==========
import json
import logging
import re
from typing import Dict, Any
from datetime import datetime, timedelta

# ========== LangGraph Agent Core & Utilities ==========
from langgraph_agent.core.base import AgentState, AgentConfig
from langgraph_agent.core.state import update_patient_context, add_to_conversation_memory

# ========== LangChain & External Libraries ==========
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ========== Database & Context Manager Tools ==========
from langgraph_agent.tools.database import (
    execute_query, resolve_doctor_uuid_to_id, resolve_doctor_name_from_uuid,
    get_next_appointment, get_patient_history, get_doctor_schedule, lookup_patient_id, 
    get_service_id_and_duration, get_doctor_default_branch
)

from langgraph_agent.tools.mcp_context_manager import mcp_context_manager

# Core agent types
from langgraph_agent.core.base import AgentState, AgentConfig
# LangChain message types
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph_agent.tools.mcp_context_manager import mcp_context_manager


logger = logging.getLogger(__name__)

# --- Tool Executor Node ---
def tool_executor_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Executes the selected tool (e.g., DB query, booking) and updates the state with results.
    """
    logger.info(f"Tool executor running for tool: {state.get('selected_tool')}")
    tool = state.get("selected_tool")
    logger.info(f"[DEBUG] tool_executor_node START: sql_query_parameters = {state.get('sql_query_parameters')} (type: {type(state.get('sql_query_parameters'))})")
    if state.get('sql_query_parameters') is None:
        logger.warning('[DEBUG] sql_query_parameters is None at tool_executor_node entry!')
    elif not isinstance(state.get('sql_query_parameters'), (list, tuple)):
        logger.warning(f'[DEBUG] sql_query_parameters is not a list/tuple: {state.get('sql_query_parameters')}')
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
    # Guard: Ensure sql_query is present
    if not state.get("sql_query"):
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
            results = execute_query(state["sql_query"], params)
        elif tool == "appointment_lookup":
            results = execute_query(state["sql_query"], params)
        elif tool == "patient_lookup":
            results = execute_query(state["sql_query"], params)
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

# --- SQL Generator Node ---
def sql_generator_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Generates SQL query using LLM and executes it, logging all relevant metadata.
    """
    logger.info("SQL generator node invoked")
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
        raise ValueError("Empty response from OpenAI")
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
                raise je
        else:
            raise je
    state["sql_query"] = result.get("sql_query")
    query_params = result.get("query_parameters", [])
    state["sql_query_parameters"] = query_params
    logger.info(f"[DEBUG] sql_generator_node END: sql_query_parameters = {state['sql_query_parameters']}")
    query_type = result.get("query_type", "unknown")
    reasoning = result.get("reasoning", "No reasoning provided")
    logger.info(f"SQL generator produced query: {state['sql_query']}")
    logger.info(f"SQL generator produced parameters: {query_params}")
    logger.info(f"SQL generator query_type: {query_type}")
    if result.get("success"):
        if result.get("results"):
            state["tool_results"] = result["results"]
        else:
            state["tool_results"] = [{"rowcount": result.get("rowcount", 0), "success": True}]
    else:
        state["tool_results"] = []
        state.setdefault("errors", []).append(f"Appointment query executor error: {result.get('error')}")
        state["has_errors"] = True
    print("=" * 80)
    print("🗄️  LLM-GENERATED SQL EVALUATION - LangGraph Medical Agent")
    print("=" * 80)
    print(f"📊 Tool: {state.get('selected_tool', 'unknown')}")
    print(f"🎯 Query Intent: {state.get('query_intent', 'unknown')}")
    print(f"🔐 User Role: {state.get('user_role', 'unknown')}")
    print(f"👤 Doctor ID: {state.get('doctor_id', 'N/A')} -> {doctor_id_mapped}")
    print(f"🆔 Session: {state.get('session_id', 'N/A')}")
    print(f"📝 Original Query: {state.get('current_query', 'N/A')}")
    print(f"🧠 LLM Reasoning: {reasoning}")
    print(f"Query Type: {query_type}")
    print(f"🗄️  Generated SQL: {state['sql_query']}")
    print(f"📊 Parameters: {query_params}")
    print(f"📈 Result Count: {len(state['tool_results']) if state['tool_results'] else 0} rows")
    print(f"🎯 Context References: {state.get('resolved_references', {})}")
    print(f"⏰ Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    # Log SQL metadata (fixed indentation and block structure)
    state.setdefault("sql_metadata", {})["query_info"] = {
        "original_params": query_params,
        "mapped_params": query_params,
        "doctor_uuid_mapping": f"{doctor_uuid} -> {doctor_id_mapped}" if doctor_uuid and doctor_id_mapped else None,
        "result_count": len(state["tool_results"]) if state["tool_results"] else 0,
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
        "tool_results": state["tool_results"],
        "user_role": state["user_role"],
        "patient_context": state.get("patient_context"),
    }
    return state


def slot_validator_node(state, config):
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

    # 1. Always use LLM to determine missing user-facing fields and generate clarification prompt
    system_prompt = (
        "You are a medical assistant agent.\n"
        "Given the current tool, parameters, and context, do the following:\n"
        "1. Determine which user-facing fields are required for the current tool (do NOT include backend fields like IDs, status, etc.).\n"
        "2. Identify which required user-facing fields are missing.\n"
        "3. If any required user-facing fields are missing, generate a natural, context-aware prompt asking ONLY for those fields, using available context.\n"
        "4. If all user-facing fields are present, parse and normalize user input for date, time, and other fields.\n"
        "Respond in JSON with: { 'normalized': { ... }, 'required_fields': [...], 'missing_fields': [...], 'clarification_prompt': '...' }\n"
        "Do NOT ask about backend fields. Only prompt for user-facing fields relevant to the tool."
    )
    llm_input = {
        "tool": tool,
        "parameters": resolved,
        "context": {
            "query_intent": state.get("query_intent"),
            "resolved_references": resolved_references,
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
    elif missing_backend:
        state["slot_validation"] = {"status": "missing_backend", "fields": missing_backend}
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


def appointment_overlap_check_node(state, config):
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

def doctor_schedule_check_node(state, config):
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


# --- Response Formatter Node ---
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
    }
    # --- Context-aware prompt for logged-in doctors ---
    user_role = state.get("user_role", "user")
    doctor_id = state.get("doctor_id")
    doctor_name = state.get("tool_parameters", {}).get("DoctorName") or state.get("resolved_references", {}).get("DoctorName")
    is_doctor = user_role.lower() == "doctor" or (doctor_id and str(doctor_id) != "" and str(doctor_id) != "None")
    # Compose system prompt for second-person responses if doctor is logged in
    if is_doctor:
        system_prompt = (
            "You are a helpful medical assistant.\n"
            "The user interacting with you is a doctor (DoctorId: {doctor_id}, DoctorName: {doctor_name}).\n"
            "Generate all responses in the second person, addressing the doctor directly (e.g., 'You have 3 appointments today', 'Your next patient is...').\n"
            "If there are errors, ambiguities, or missing information, explain them clearly and politely.\n"
            "If a clarification prompt is present, use it to ask the doctor for more information.\n"
            "If the request was successful, summarize the results in a user-friendly way, always using second person.\n"
            "Always respond in JSON with the following format:\n"
            "{\n  \"formatted_response\": \"...\",\n  \"response_metadata\": { ... },\n  \"suggested_followups\": []\n}"
        ).format(doctor_id=doctor_id, doctor_name=doctor_name)
        prompt = (
            "Given the following context, generate a single, natural, user-facing response. "
            "If there are errors, ambiguities, or missing information, explain them clearly and politely. "
            "If a clarification prompt is present, use it to ask the doctor for more information. "
            "If the request was successful, summarize the results in a user-friendly way, always using second person (addressing the doctor as 'you'). "
            "Respond ONLY in the required JSON format.\n\n"
            f"CONTEXT:\n{json.dumps(confirmation_context, indent=2, default=str)}"
        )
    else:
        system_prompt = (
            "You are a helpful medical assistant.\n"
            "Your job is to generate a natural, context-aware response for the user based on the provided context.\n"
            "If there are errors, ambiguities, or missing information, explain them clearly and politely.\n"
            "If a clarification prompt is present, use it to ask the user for more information.\n"
            "If the request was successful, summarize the results in a user-friendly way.\n"
            "Always respond in JSON with the following format:\n"
            "{\n  \"formatted_response\": \"...\",\n  \"response_metadata\": { ... },\n  \"suggested_followups\": []\n}"
        )
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

# Only prompt for user-dependent fields; auto-resolve the rest
USER_DEPENDENT_FIELDS = {"service_name", "patient_name", "start_time", "appointment_date"}
AUTO_RESOLVE_FIELDS = {"service_id", "branch_name", "appointment_id", "status_id", "end_time"}
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
            if quote_count % 2 == 0:
                line = line[:comment_pos].rstrip()
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def preprocess_date_references(query: str) -> Dict[str, str]:
    """
    Preprocess date references like 'tomorrow', 'today', 'next week' etc.
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

    # Handle next week
    if 'next week' in query_lower:
        next_week_date = current_date + timedelta(days=7)
        date_mappings['next week'] = next_week_date.strftime('%Y-%m-%d')

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


def context_resolver_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Enhanced context resolver with MCP (Model Context Protocol) integration.
    This node analyzes the current query against conversation memory and 
    existing context to resolve references like "next patient", "she", etc.
    Enhanced with MCP for standardized context management.
    """
    logger.info(f"Context resolver processing: {state['current_query']}")
    session_id = state.get("session_id", "default_session")
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
    # Refined system prompt: all rules and output schema in SystemMessage
    system_prompt = """
You are a medical assistant context resolver.
- Resolve vague references (e.g. 'him', 'her', 'next patient') using MCP context, recent messages, and date mappings.
- Respond with ONLY a valid JSON object (no markdown, no explanation, no preamble).
- If any value cannot be resolved, set it to null.
Output Schema:
{
  \"query_intent\": "...",
  \"resolved_references\": { ... },
  \"context_updates\": { ... }
}
""".strip()
    # Refined system prompt: all rules and output schema in SystemMessage
    system_prompt = """
You are a medical assistant context resolver.
- Resolve vague references (e.g. 'him', 'her', 'next patient') using MCP context, recent messages, and date mappings.
- Respond with ONLY a valid JSON object (no markdown, no explanation, no preamble).
- If any value cannot be resolved, set it to null.
Output Schema:
{
  \"query_intent\": "...",
  \"resolved_references\": { ... },
  \"context_updates\": { ... }
}
""".strip()
    mcp_context_summary = mcp_context_manager.get_context_summary(session_id)
    context_info = {
        "current_query": state["current_query"],
        "user_role": state["user_role"],
        "doctor_id": state.get("doctor_id"),
        "patient_context": state.get("patient_context"),
        "doctor_context": state.get("doctor_context"),
        "conversation_memory": state["conversation_memory"],
        "recent_messages": [msg.content for msg in state["messages"][-5:] if hasattr(msg, 'content')],
        "recent_messages": [msg.content for msg in state["messages"][-5:] if hasattr(msg, 'content')],
        "mcp_context_summary": mcp_context_summary,
        "mcp_resolved_references": mcp_resolved_refs,
        "date_mappings": date_mappings,
        "current_date": datetime.now().strftime('%Y-%m-%d')
    }
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(context_info, indent=2, default=str))
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
        # Final resolved references:
        # - LLM output takes precedence
        # - MCP fills gaps
        # - Date mappings are injected into the same object
        llm_resolved = result.get("resolved_references", {})
        final_resolved = {**mcp_resolved_refs, **date_mappings, **llm_resolved}
        final_resolved = {**mcp_resolved_refs, **date_mappings, **llm_resolved}
        state["resolved_references"] = final_resolved
        context_updates = result.get("context_updates", {})
        if context_updates.get("patient_context"):
            state = update_patient_context(state, **context_updates["patient_context"])
        logger.info(f"Context resolved - Intent: {state['query_intent']}, References: {state['resolved_references']}")
    except Exception as e:
        logger.error(f"Context resolution error: {e}")
        state["errors"].append(f"Context resolution failed: {e}")
        state["has_errors"] = True
        query_lower = state["current_query"].lower()
        if any(word in query_lower for word in ["next", "upcoming"]):
            state["query_intent"] = "next_patient"
        elif any(word in query_lower for word in ["history", "past"]):
            state["query_intent"] = "patient_history"
        elif any(word in query_lower for word in ["schedule", "calendar"]):
            state["query_intent"] = "schedule"
        else:
            state["query_intent"] = "general_query"
        if mcp_resolved_refs:
            state["resolved_references"] = mcp_resolved_refs
            logger.info(f"Using MCP fallback references: {mcp_resolved_refs}")
    return state


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
    
    prompt = f"""
{system_prompt}

Selection context:
{json.dumps(selection_context, indent=2, default=str)}

TOOL SELECTION GUIDELINES:
- For "time_specific_lookup" intent: Use "schedule_query" with specific time filtering
- For "next_patient" intent: Use "appointment_lookup" to find chronologically next appointment
- For "schedule" intent: Use "schedule_query" for general schedule viewing
- For "patient_history" intent: Use "patient_lookup" or "history_query"

TIME-SPECIFIC HANDLING:
- If resolved_references contains specific times (e.g., "2 PM", "14:00"), this indicates a time-specific lookup
- Generate parameters that will filter by the specific time mentioned
- For time-specific queries, include both date and time constraints in parameters

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
        
        logger.info(f"Tool selected: {state['selected_tool']} with params: {state['tool_parameters']}")
        
    except Exception as e:
        logger.error(f"Tool selection error: {e}")
        state["errors"].append(f"Tool selection failed: {e}")
        state["has_errors"] = True
        # Fallback tool selection
        if state["query_intent"] == "next_patient":
            state["selected_tool"] = "appointment_lookup"
            state["tool_parameters"] = {"type": "next_appointment", "doctor_id": state.get("doctor_id")}
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

# def slot_validator_node(state: AgentState, config: AgentConfig) -> AgentState:
#     """
#     Slot Validator Node
#     ------------------
#     Validates required parameters before SQL execution. Prompts ONLY for natural fields (service name, patient name, date/time).
#     All IDs and backend fields are resolved internally (set as None for now, with comments for future DB lookups).
#     Logs backend field resolution in response_metadata and mcp_context.
#     """

#     logger.info(f"Slot validator received tool_parameters: {state.get('tool_parameters')}")
#     logger.info(f"Slot validator checking parameters for tool: {state.get('selected_tool')}")
#     params = state.get("tool_parameters", {})
#     resolved = {**params, **state.get("resolved_references", {})}

#     # Use LLM to determine required fields, missing fields, and parse natural language dates
#     system_prompt = (
#         "You are a medical assistant agent."
#         " Given the current tool, parameters, and context, do the following:"
#         " 1. Parse any natural language dates/times (e.g., 'next Monday', 'tomorrow') and fill them in as ISO date/time if possible."
#         " 2. Determine which fields are required for the current tool (do NOT include backend fields like service_id, patient_id, etc.)."
#         " 3. Identify which required fields are missing."
#         " 4. If any required fields are missing, generate a natural prompt asking ONLY for those fields, using available context."
#         " Respond in JSON with:"
#         " {"
#         "   'parsed_parameters': { ... },"
#         "   'required_fields': [ ... ],"
#         "   'missing_fields': [ ... ],"
#         "   'clarification_prompt': '...'"
#         " }"
#         " Do NOT ask about backend fields. Only prompt for user-facing fields relevant to the tool."
#     )
#     llm_input = {
#         "tool": state.get("selected_tool"),
#         "parameters": resolved,
#         "context": {
#             "query_intent": state.get("query_intent"),
#             "resolved_references": state.get("resolved_references", {}),
#             "patient_context": state.get("patient_context"),
#             "doctor_context": state.get("doctor_context"),
#             "conversation_memory": state.get("conversation_memory"),
#         }
#     }
#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=json.dumps(llm_input, default=str))
#     ]
#     response = config.llm.invoke(messages)
#     try:
#         result = safe_json_parse(response.content, "slot validator")
#     except Exception as e:
#         logger.error(f"Slot validator LLM response parse error: {e}")
#         result = {
#             "parsed_parameters": params,
#             "required_fields": [],
#             "missing_fields": [],
#             "clarification_prompt": "Sorry, I couldn't parse the required fields."
#         }


#     # Merge original tool_parameters with parsed_parameters from LLM
#     parsed_params = result.get("parsed_parameters", {})
#     # Always start from original tool_parameters and update with parsed_params
#     merged_params = {**params, **parsed_params}
#     state["tool_parameters"] = merged_params
#     all_missing = result.get("missing_fields", [])
#     # Define backend/internal fields (never prompt user for these)
#     backend_fields = ["StatusId", "PatientId", "BranchName", "ServiceId", "BranchId"]
#     missing_backend = [f for f in backend_fields if not state["tool_parameters"].get(f)]
#     missing_natural = [f for f in all_missing if f not in backend_fields]

#     # Always populate required_lookups with all backend fields for appointment_booking
#     if state.get("selected_tool") == "appointment_booking":
#         state["required_lookups"] = backend_fields
#     elif missing_natural and missing_backend:
#         state["required_lookups"] = list(set(missing_natural + missing_backend))
#     elif missing_backend:
#         state["required_lookups"] = missing_backend
#     elif missing_natural:
#         state["required_lookups"] = missing_natural
#     else:
#         state["required_lookups"] = []

#     # Routing logic is now handled by graph condition functions, not by node state.
#     if missing_natural and missing_backend:
#         state["slot_validation"] = {"status": "missing", "fields": missing_natural}
#         state["clarification_prompt"] = result.get("clarification_prompt", "")
#         state["formatted_response"] = result.get("clarification_prompt", "")
#     elif missing_backend:
#         state["slot_validation"] = {"status": "missing_backend", "fields": missing_backend}
#         state["clarification_prompt"] = ""
#         state["formatted_response"] = ""
#     elif missing_natural:
#         state["slot_validation"] = {"status": "missing", "fields": missing_natural}
#         state["clarification_prompt"] = result.get("clarification_prompt", "")
#         state["formatted_response"] = result.get("clarification_prompt", "")
#     else:
#         state["slot_validation"] = {"status": "ok", "fields": []}
#         state["clarification_prompt"] = ""
#         state["formatted_response"] = ""
#     return state

def backend_lookup_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Backend Lookup Node
    ------------------
    Resolves backend/internal fields (IDs, branch, status) using available context and lookup tools.
    This node is never exposed to the user and does not prompt for input.
    """
    params = state.get("tool_parameters", {})
    logger.info(f"🔍 Lookup received params: {params}")
    resolved = {**params, **state.get("resolved_references", {})}
    backend_fields = ["StatusId", "PatientId", "BranchName", "ServiceId", "BranchId"]
    debug_lookups = {}
    if state.get("selected_tool") == "appointment_booking":
        # --- Datetime Handling ---
        # 1. Handle appointment_datetime in ISO format (2025-04-14T10:45:00)
        if "appointment_datetime" in params and params["appointment_datetime"]:
            dt_str = params["appointment_datetime"]
            # Detect ISO format with 'T'
            if "T" in dt_str:
                try:
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
                dt = datetime.strptime(f"{params['date']} {params['time']}", "%Y-%m-%d %H:%M")
                formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
                params["StartDateTime"] = formatted
                resolved["StartDateTime"] = formatted
                logger.info(f"Combined date and time to StartDateTime: {formatted}")
            except Exception as e:
                logger.warning(f"Could not parse date/time: {e}")
        # Remove appointment_date and appointment_time keys if present
        for k in ["appointment_date", "appointment_time"]:
            if k in params:
                del params[k]

        # --- PatientId Lookup (robust: get_or_create_patient_id) ---
        if "patient_name" in params and params["patient_name"]:
            try:
                from langgraph_agent.tools.database import get_or_create_patient_id
                patient_id = get_or_create_patient_id(params["patient_name"])
                params["PatientId"] = patient_id
                resolved["PatientId"] = patient_id
                debug_lookups["PatientId"] = patient_id
                logger.info(f"Resolved PatientId for {params['patient_name']}: {patient_id}")
            except Exception as e:
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
                from langgraph_agent.tools.database import execute_query
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

        # --- Log all resolved and missing fields ---
        still_missing = [f for f in ["PatientId", "ServiceId", "BranchId", "BranchName"] if not params.get(f)]
        logger.info(f"Backend lookup completed. Resolved: {debug_lookups}, Still missing: {still_missing}")
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
        state["tool_parameters"] = params
        state["resolved_references"] = resolved
        state["required_lookups"] = []
        backend_logs = {
            "tool": state.get("selected_tool"),
            "resolved_lookups": debug_lookups,
            "still_missing": still_missing,
            "final_params": params
        }
        state.setdefault("response_metadata", {})["backend_logs"] = backend_logs
        state.setdefault("mcp_context", {})["backend_logs"] = backend_logs
        logger.info(f"Backend lookup node result: params={params}, debug={backend_logs}")
        print(f"[DEBUG] backend_lookup_node slot_validation: {state.get('slot_validation')}")
        logger.debug(f"Final tool parameters for SQL generation: {json.dumps(params, indent=2)}")
        return state
    # ...existing code for other tools...

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
        # Store patient/appointment context for both next_patient and time_specific_lookup
        if state["query_intent"] in ["next_patient", "time_specific_lookup"] and state["tool_results"]:
            appointment = state["tool_results"][0] if state["tool_results"] else None
            if appointment and isinstance(appointment, dict):
                state["conversation_memory"]["implicit_references"]["current_patient"] = {
                    "name": appointment.get("PatientName"),
                    "id": appointment.get("PatientID"),
                    "appointment_id": appointment.get("AppointmentID"),
                    "name": appointment.get("PatientName"),
                    "id": appointment.get("PatientID"),
                    "appointment_id": appointment.get("AppointmentID"),
                    "timestamp": datetime.now().isoformat()
                }
                patient_context_id = mcp_context_manager.add_patient_context(
                    patient_name=appointment.get("PatientName", "Unknown"),
                    patient_id=str(appointment.get("PatientID", "")),
                    patient_name=appointment.get("PatientName", "Unknown"),
                    patient_id=str(appointment.get("PatientID", "")),
                    appointment_details={
                        "appointment_id": appointment.get("AppointmentID"),
                        "start_time": appointment.get("StartDateTime"),
                        "end_time": appointment.get("EndDateTime"),
                        "appointment_type": appointment.get("AppointmentType"),
                        "status": appointment.get("Status")
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
                    patient_id=appointment.get("PatientID"),
                    patient_name=appointment.get("PatientName"),
                    appointment_id=appointment.get("AppointmentID"),
                    appointment_date=appointment.get("StartDateTime")
                )
        if state["query_intent"] == "schedule" and state["tool_results"]:
        if state["query_intent"] == "schedule" and state["tool_results"]:
            schedule_context_id = mcp_context_manager.add_schedule_context(
                schedule_data=state["tool_results"],
                date=datetime.now().strftime("%Y-%m-%d"),
                session_id=session_id
            )
            logger.info(f"Added MCP schedule context: {schedule_context_id}")
        if state["query_intent"] == "patient_history" and state["tool_results"]:
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