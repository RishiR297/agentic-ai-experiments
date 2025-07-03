# ==============================================
# File: nodes.py
# Purpose: Define LangGraph nodes that process the agent's internal state
# ==============================================

# ===============================================
# IMPORTS
# ===============================================

# Standard library imports
import re
import os
import json
import difflib
from typing import Optional
from datetime import datetime, timedelta

# Third-party imports
from dotenv import load_dotenv
from pydantic import BaseModel
from dateutil import parser as date_parser

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

# Local imports
from agent.state import AgentState
from tool_server import MCP_TOOL_REGISTRY, MCP_FUNCTION_LOOKUP
from tools.doctor import (
    get_branch_id_for_doctor, 
    get_services_for_doctor, 
    is_service_valid_for_doctor, 
    suggest_doctor_for_service
)
from agent.tools.mcp_client import call_mcp_tool
from agent.tools.appointment import (
    clean_date_line,
    parse_slot_line,
    detect_selected_slot_with_llm
)
from utils.llm_extraction import extract_fields_from_user_input

# ===============================================
# CONFIGURATION & SETUP
# ===============================================

# Initialize environment
load_dotenv()
print("Loaded nodes.py at runtime")

# Safe conversion to avoid serialization errors
SAFE_TOOL_REGISTRY = [
    tool if isinstance(tool, dict) else tool.dict() for tool in MCP_TOOL_REGISTRY
]

# Sensitive tools that require special access
SENSITIVE_TOOLS = {"get_next_client_info", "summarize_appointments"}

# Azure OpenAI LLM setup
llm_with_tools = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview",
    model_kwargs={"tools": SAFE_TOOL_REGISTRY}
)

# LLM without tools for prompting tasks like slot-filling
llm_basic = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview",
)

# MCP-based planner prompt
MCP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that decides the next action."),
    ("human", "{user_input}")
])

# Tool lookup for MCP functions
TOOL_LOOKUP = MCP_FUNCTION_LOOKUP

# Debug: Print registered tools
print("Tools registered:")
for i, tool in enumerate(MCP_TOOL_REGISTRY):
    print(f"Tool #{i + 1}: type={type(tool)}")
    print(json.dumps(tool, indent=2) if isinstance(tool, dict) else tool)

for tool in MCP_TOOL_REGISTRY:
    print(f"- {tool['function']['name']}")

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def detect_selected_slot(state: dict) -> dict:
    """
    Detects if user input refers to a specific available slot using LLM inference.
    Returns a dict with start_time and end_time if match is found.
    """
    user_input = state.get("user_input", "").lower()
    slot_lines = state.get("available_slot_lines", [])
    
    return detect_selected_slot_with_llm(user_input, slot_lines, llm_basic)


def extract_slots(user_input: str) -> dict:
    """Extract structured fields from user input using LLM."""
    return extract_fields_from_user_input(user_input, llm_basic)

# ===============================================
# STATE MANAGEMENT FUNCTIONS
# ===============================================

def check_missing_fields(state: dict) -> list:
    """Check which required fields are missing for booking."""
    # Check if we have a proposed booking or intermediate steps for booking
    proposed_booking = state.get("proposed_booking")
    tool_name = None
    
    if proposed_booking:
        tool_name = proposed_booking.get("tool_name")
    else:
        tool_name = state.get("intermediate_steps", [{}])[0].get("tool_name") if state.get("intermediate_steps") else None
    
    if tool_name != "book_appointment_tool":
        return []

    missing_fields = []
    
    # Check both state and proposed_booking args
    for field in AgentState.REQUIRED_FIELDS:
        state_val = state.get(field)
        booking_val = None
        
        if proposed_booking and "args" in proposed_booking:
            booking_val = proposed_booking["args"].get(field)
        
        # Field is missing if both state and booking args are None/empty
        if not state_val and not booking_val:
            missing_fields.append(field)
    
    print(f"[DEBUG] Missing fields check: {missing_fields}")
    return missing_fields

def fill_missing_booking_fields(state: dict) -> None:
    """Ensure proposed_booking contains up-to-date fields from state."""
    if "proposed_booking" not in state:
        return

    args = state["proposed_booking"].get("args", {})
    updated = False

    for key in ["patient_name", "service_name", "doctor_name", "branch_id", "start_time", "end_time"]:
        state_val = state.get(key)
        if state_val and (args.get(key) is None or args.get(key) == ""):
            args[key] = state_val
            updated = True

    # Special handling for service_name to ensure it doesn't get lost
    if state.get("service_name") and args.get("service_name") is None:
        args["service_name"] = state["service_name"]
        updated = True

    if updated:
        print("[DEBUG] Rehydrated proposed_booking args:", args)
    else:
        print("[DEBUG] No changes to proposed_booking args")

# ===============================================
# MAIN NODE FUNCTIONS
# ===============================================

def welcome_node(state: dict) -> dict:
    """Initial welcome message for new conversations."""
    if not state.get("chat_history"):
        return {
            **state,
            "final_answer": "👋 Hi! How can I help you today?",
            "next": "answer"
        }
    return state

def ask_for_missing_fields_node(state: dict) -> dict:
    """Ask user for missing required fields."""
    missing_fields = state.get("missing_fields", [])
    
    if not missing_fields:
        return state

    # Only fill missing fields, don't trigger other logic
    fill_missing_booking_fields(state)
    
    prompts = {
        "doctor_name": "the doctor's name",
        "patient_name": "your name",
        "branch_id": "the branch ID",
        "service_name": "the service needed",
        "start_time": "the preferred start time",
        "end_time": "the preferred end time"
    }

    questions = [f"- Please provide {prompts.get(field, field)}." for field in missing_fields]
    message = "To proceed with booking, I need the following:\n" + "\n".join(questions)

    # Clear any slot suggestions to prevent looping
    state.pop("available_slot_lines", None)
    state["final_answer"] = message
    return state

def planner_node(state: dict) -> dict:
    """Main planning node that processes user input and determines next actions."""
    fill_missing_booking_fields(state)
    
    user_input = state.get("user_input")
    if not user_input:
        raise ValueError("Missing 'user_input' in agent state")
    user_input_lower = user_input.lower()

    # ----------------------------------------
    # Step 1: Extract structured fields
    # ----------------------------------------
    extracted_fields = {}
    should_extract = not state.get("awaiting_confirmation") and not state.get("available_slot_lines")

    if should_extract:
        try:
            extracted_fields = extract_slots(user_input)
            # Merge all extracted fields into state
            for key, value in extracted_fields.items():
                if value is not None:
                    state[key] = value

            print("[DEBUG] Updated state after slot extraction:", {
                k: state.get(k) for k in ["patient_name", "service_name", "doctor_name", "weekday"]
            })

            fill_missing_booking_fields(state)
        except Exception as e:
            print(f"extract_slots failed: {e}")

    # If we're awaiting confirmation or have slot selection, also extract fields for missing info
    if state.get("awaiting_confirmation") or state.get("available_slot_lines") or state.get("missing_fields"):
        try:
            extracted_fields = extract_slots(user_input)
            # Only merge non-conflicting fields
            for key, value in extracted_fields.items():
                if value is not None and key in ["patient_name", "service_name"] and not state.get(key):
                    state[key] = value
                    print(f"[DEBUG] Added missing field {key}: {value}")
            
            fill_missing_booking_fields(state)
            
            # If we have a proposed booking and just added missing info, check if we can proceed
            if state.get("proposed_booking") and state.get("awaiting_confirmation"):
                missing = check_missing_fields(state)
                if not missing:
                    # All fields are now available, proceed to booking
                    state["intermediate_steps"] = [state.pop("proposed_booking")]
                    state["awaiting_confirmation"] = False
                    state.pop("missing_fields", None)  # Clear missing fields
                    return {**state, "next": "tool"}
            
            # If we were collecting missing fields and now have them, check what to do next
            if state.get("missing_fields"):
                missing = check_missing_fields(state)
                if not missing:
                    # All required fields collected, ready to proceed
                    if state.get("proposed_booking"):
                        state["awaiting_confirmation"] = True
                        state.pop("missing_fields", None)
                        # Generate confirmation message
                        doctor = state.get("doctor_name", "the doctor")
                        start_time = state.get("start_time")
                        readable_time = datetime.fromisoformat(start_time).strftime("%A, %b %d at %H:%M") if start_time else "[unknown time]"
                        state["final_answer"] = (
                            f"Great! Now I have all the information needed.\n"
                            f"Confirming your appointment on {readable_time} with {doctor}.\n"
                            f"Would you like to proceed with this booking? Please reply with 'yes' to confirm or 'no' to cancel."
                        )
                        return {**state, "next": "answer"}
                else:
                    # Still missing some fields, continue collecting
                    state["missing_fields"] = missing
                    return {**state, "next": "ask_missing_info"}
                    
        except Exception as e:
            print(f"extract_slots for missing fields failed: {e}")

    # Autofill branch_id if missing but doctor known
    if state.get("doctor_name") and not state.get("branch_id"):
        inferred_branch_id = get_branch_id_for_doctor(state["doctor_name"])
        print(f"[DEBUG autofill] Inferred branch ID for {state['doctor_name']}: {inferred_branch_id}")
        if inferred_branch_id:
            state["branch_id"] = inferred_branch_id

    # Handle user asking for services
    if any(kw in user_input.lower() for kw in ["what services", "available services", "offer", "provide"]):
        doctor = state.get("doctor_name")
        if doctor:
            services = get_services_for_doctor(doctor)
            if services:
                formatted = ", ".join(services)
                clean_doc_name = doctor.replace("Dr.", "").strip()
                state["final_answer"] = f"Dr. {clean_doc_name} offers the following services: {formatted}.\nPlease choose one to proceed."
                # Clear any pending steps or tool results if needed
                state.pop("intermediate_steps", None)
                state["tool_results"] = []
                return {**state, "next": "answer"}

    # Handle user asking if doctor is available "today"
    if any(kw in user_input.lower() for kw in ["available today", "today", "free today", "open today"]):
        doctor = state.get("doctor_name")
        if doctor:
            # Get today's weekday (0=Monday, 6=Sunday)
            today_weekday = datetime.now().weekday()
            
            print(f"[DEBUG today check] Doctor: {doctor}, Today's weekday: {today_weekday}")
            
            # Set the weekday and trigger slot suggestion
            state["weekday"] = today_weekday
            state["intermediate_steps"] = [{
                "tool_name": "suggest_appointment_slots",
                "args": {
                    "doctor_name": doctor,
                    "weekday": today_weekday
                }
            }]
            return {**state, "next": "tool"}
    
    # Validate service name
    if state.get("doctor_name") and state.get("service_name"):
        doctor = state["doctor_name"]
        service = state["service_name"]
        if not is_service_valid_for_doctor(doctor, service):
            alternatives = suggest_doctor_for_service(service)
            if alternatives:
                state["final_answer"] = (
                    f"Dr. {doctor.replace('Dr.', '').strip()} does not offer '{service}'.\n"
                    f"However, {alternatives[0]} does. Would you like to book with them?"
                )
            else:
                state["final_answer"] = (
                    f"Sorry, '{service}' is not available with any of our doctors at the moment."
                )
            return {**state, "next": "answer"}
        
    # Clear previous outputs
    state.pop("final_answer", None)
    state["tool_results"] = []

    # ----------------------------------------
    # Step 2: Handle booking confirmation
    # ----------------------------------------
    if state.get("awaiting_confirmation") and state.get("proposed_booking"):
        user_input_lower = user_input.lower()
        confirmed = any(word in user_input_lower for word in ["yes", "yeah", "yep", "confirm", "go ahead", "sure"])

        if confirmed:
            if missing := check_missing_fields(state):
                state["missing_fields"] = missing
                return {**state, "next": "ask_missing_info"}
            else:
                state["intermediate_steps"] = [state.pop("proposed_booking")]
                state["awaiting_confirmation"] = False
                return {**state, "next": "tool"}
        else:
            # Check if user is providing missing info instead of confirming
            if any(field in extracted_fields for field in ["patient_name", "service_name"]):
                # User is providing missing info, don't cancel - let it flow to missing field collection
                pass
            else:
                # Not a confirmation and not providing info, assume cancel
                state["awaiting_confirmation"] = False
                state.pop("proposed_booking", None)
                state.pop("start_time", None)
                state.pop("end_time", None)
                state["final_answer"] = "Okay, booking cancelled. Let me know if you'd like to try a different slot."
                state.pop("available_slot_lines", None)
                return {**state, "next": "answer"}

    # ----------------------------------------
    # Step 3: Detect slot selection
    # ----------------------------------------
    if not state.get("awaiting_confirmation"):
        selected = detect_selected_slot(state)
        if selected:
            # User selected a slot -> prepare confirmation
            start_time = selected.get("start_time")
            end_time = selected.get("end_time")
            doctor = state.get("doctor_name", "the doctor")
            patient = state.get("patient_name")
            service = state.get("service_name")

            # Autofill missing branch ID now based on doctor
            if not state.get("branch_id") and doctor:
                inferred_branch_id = get_branch_id_for_doctor(doctor)
                print(f"[DEBUG autofill] Inferred branch ID for {doctor}: {inferred_branch_id}")
                if inferred_branch_id:
                    state["branch_id"] = inferred_branch_id

            state["awaiting_confirmation"] = True
            state["start_time"] = start_time
            state["end_time"] = end_time
            
            # Preserve existing patient_name and service_name if they exist
            if patient:
                state["patient_name"] = patient
            if service:
                state["service_name"] = service

            print("[DEBUG confirmation step] State patient_name:", state.get("patient_name"))
            print("[DEBUG confirmation step] State service_name:", state.get("service_name"))
            print("[DEBUG] Proposed booking args:", {
                "doctor_name": doctor,
                "start_time": start_time,
                "end_time": end_time,
                "patient_name": state.get("patient_name"),
                "branch_id": state.get("branch_id"),
                "service_name": state.get("service_name"),
            })

            state["proposed_booking"] = {
                "tool_name": "book_appointment_tool",
                "args": {
                    "doctor_name": doctor,
                    "start_time": start_time,
                    "end_time": end_time,
                    "patient_name": state.get("patient_name"),
                    "branch_id": state.get("branch_id"),
                    "service_name": state.get("service_name"),
                },
             }

            # Check for missing fields immediately after slot selection
            missing_fields = check_missing_fields(state)
            print(f"[DEBUG] Missing fields after slot selection: {missing_fields}")
            
            if missing_fields:
                # Don't show confirmation, go directly to collecting missing info
                state["missing_fields"] = missing_fields
                state.pop("available_slot_lines", None)
                return {**state, "next": "ask_missing_info"}

            state.pop("available_slot_lines", None)
            readable_time = datetime.fromisoformat(start_time).strftime("%A, %b %d at %H:%M") if start_time else "[unknown time]"
            state["final_answer"] = (
                f"You selected the slot on {readable_time} with {doctor}.\n"
                f"Would you like to confirm this booking? Please reply with 'yes' to proceed or 'no' to cancel."
            )

            return {
                **state,
                "chat_history": state.get("chat_history", [])[-6:] + [HumanMessage(content=user_input)],
                "next": "answer",
            }

    # ----------------------------------------
    # Step 4: Fallback to LLM planner
    # ----------------------------------------
    # Only use LLM planner if we don't have an ongoing booking process
    # and we're not collecting missing info
    if (not state.get("awaiting_confirmation") and 
        not state.get("start_time") and 
        not state.get("proposed_booking") and
        not state.get("missing_fields")):
        
        # Check if we should suggest slots (but only if not collecting missing info)
        if "doctor_name" in state and "weekday" in state:
            state["intermediate_steps"] = [{
                "tool_name": "suggest_appointment_slots",
                "args": {
                    "doctor_name": state["doctor_name"],
                    "weekday": state["weekday"]
                }
            }]
            return {**state, "next": "tool"}
        
        # If no specific slot request and no missing fields, proceed with LLM
        system_prompt = SystemMessage(
            content="You are a helpful assistant. Use the available tools to assist the user."
        )
        chat_history = state.get("chat_history", [])[-6:]
        messages: list[BaseMessage] = [system_prompt] + chat_history

        response = llm_with_tools.invoke(messages)

        if not getattr(response, "tool_calls", None):
            chat_history.append(response)
            return {
                **state,
                "final_answer": response.content,
                "chat_history": chat_history,
                "next": "answer"
            }
    else:
        # We have ongoing booking process or missing fields - don't trigger LLM planner
        if state.get("missing_fields"):
            return {**state, "next": "ask_missing_info"}
        else:
            # Some other ongoing process, just return state
            return state

    # ----------------------------------------
    # Step 5: Parse tool calls
    # ----------------------------------------
    if hasattr(response, "tool_calls") and response.tool_calls:
        clean_steps = []

        for tool_call in response.tool_calls:
            tool_name = getattr(tool_call, "name", None) if not isinstance(tool_call, dict) else tool_call.get("name")
            raw_args = getattr(tool_call, "args", {}) if not isinstance(tool_call, dict) else tool_call.get("args", {})

            if hasattr(raw_args, "model_dump"):
                raw_args = raw_args.model_dump()
            elif not isinstance(raw_args, dict):
                raw_args = dict(raw_args)

            # Merge only expected arguments
            allowed_keys = set(raw_args.keys())
            filtered_fields = {k: v for k, v in extracted_fields.items() if k in allowed_keys}
            final_args = {**raw_args, **filtered_fields}

            # Fix outdated 'after'
            if "after" in final_args:
                try:
                    parsed = datetime.fromisoformat(final_args["after"].split("T")[0])
                    if parsed.date() < datetime.now().date():
                        final_args["after"] = datetime.now().isoformat()
                except:
                    final_args["after"] = datetime.now().isoformat()

            # Weekday mismatch warning
            if "weekday" in raw_args and "weekday" in extracted_fields:
                if raw_args["weekday"] != extracted_fields["weekday"]:
                    print(f"[⚠️ Weekday Mismatch] Planner: {raw_args['weekday']} vs Extracted: {extracted_fields['weekday']}")

            print("Tool call:", tool_name)
            print("Raw LLM args:", raw_args)
            print("Extracted fields:", extracted_fields)
            print("Final merged args:", final_args)

            clean_steps.append({
                "tool_name": tool_name,
                "args": final_args
            })
        
        # Before return in tool call block:
        state["chat_history"] = chat_history

        return {
            **state,
            "intermediate_steps": clean_steps,
            "chat_history": chat_history
        }

    # ----------------------------------------
    # Step 6: No tool calls fallback
    # ----------------------------------------
    print("No tool calls returned by LLM.")
    return {
        **state,
        "intermediate_steps": [],
        "chat_history": chat_history,
        "final_answer": "I couldn't process that. Could you please rephrase or provide more details?"
    }

def route_node(state: dict) -> dict:
    """Router node to determine next step in the flow."""
    steps = state.get("intermediate_steps", [])
    if not steps:
        state["next"] = "answer"
        return state

    fill_missing_booking_fields(state)  # Move this up BEFORE check_missing_fields
    missing = check_missing_fields(state)

    if missing:
        state["next"] = "ask_missing_info"
        state["missing_fields"] = missing
    else:
        state["next"] = "tool"

    return state

def respond_naturally_node(state: dict) -> dict:
    """Generate natural language responses based on tool results."""
    steps = state.get("intermediate_steps", [])
    if not steps:
        return state

    tool_name = steps[0]["tool_name"]
    args = steps[0]["args"]
    doctor = args.get("doctor_name", state.get("doctor_name", "the doctor")).replace("Dr.", "").strip()
    service = args.get("service_name", "the requested service")
    weekday = state.get("weekday")
    message = ""
    
    if state.get("awaiting_confirmation"):
        # Prevent overwriting planner node's confirmation message
        return state

    if tool_name == "suggest_appointment_slots":
        tool_results = state.get("tool_results", [])
        slots_text = tool_results[0] if tool_results else ""
        lines = slots_text.splitlines()
        available_slots = []

        weekday_name = None
        is_today_request = False
        
        if weekday is not None:
            weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]
            # Check if this is a "today" request
            current_weekday = datetime.now().weekday()
            is_today_request = weekday == current_weekday

        for line in lines:
            try:
                cleaned = clean_date_line(line)
                date_part = cleaned.split(":")[0].strip()
                date_with_year = f"{date_part} {datetime.now().year}"
                parsed_date = datetime.strptime(date_with_year, "%A, %b %d %Y")

                if weekday is None or parsed_date.weekday() == weekday:
                    slot_info = parse_slot_line(line)
                    if slot_info:
                        available_slots.append({
                            "start_time": slot_info["start_time"],
                            "end_time": slot_info["end_time"],
                            "display": line
                        })
            except Exception:
                available_slots.append({
                    "start_time": None,
                    "end_time": None,
                    "display": line
                })

        if available_slots:
            if is_today_request:
                message = (
                    f"Yes, Dr. {doctor} is available today ({weekday_name}). "
                    f"Here are the available time slots:\n\n"
                    + "\n".join([slot["display"] for slot in available_slots])
                    + "\n\nWhich slot would you like to book?"
                )
            else:
                message = (
                    f"Yes, Dr. {doctor} is available"
                    + (f" on {weekday_name}" if weekday_name else "")
                    + ". Here are the time slots:\n\n"
                    + "\n".join([slot["display"] for slot in available_slots])
                    + "\n\nWhich slot would you like to book?"
                )
            state["available_slot_lines"] = available_slots
        else:
            if is_today_request:
                message = (
                    f"Dr. {doctor} is not available today ({weekday_name}). "
                    "But here are some upcoming available slots:\n\n"
                    + slots_text
                )
            elif weekday_name:
                message = (
                    f"Dr. {doctor} is not available on {weekday_name}. "
                    "But here are some nearby available slots:\n\n"
                    + slots_text
                )
            else:
                message = f"Let me check available time slots for Dr. {doctor}.\n\n" + slots_text

        state["tool_results"] = []  # Clear to prevent re-display in final serializer

    elif tool_name == "book_appointment_tool":
        message = f"Booking your {service} with Dr. {doctor}. One moment..."

    elif tool_name == "appointments":
        message = f"Fetching current appointments for Dr. {doctor}."

    else:
        message = f"I'm working on your request using {tool_name}. Hang tight!"

    state["final_answer"] = message

    # Fuzzy matching logic to detect possible weekday typos in user input
    original_input = state.get("user_input", "").lower()

    if weekday is not None:
        weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]
        weekday_name_lower = weekday_name.lower()
        weekday_name_list = [d.lower() for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]

        # Split user input into words and check if any are close matches to any weekday
        matches = []
        for day in weekday_name_list:
            for w in original_input.split():
                close = difflib.get_close_matches(w, [day], cutoff=0.7)
                if close:
                    matches.append(day)

        # If a weekday other than the extracted one is matched fuzzily, warn the user
        if matches and weekday_name_lower not in matches:
            state["final_answer"] = (
                f"I interpreted that as '{weekday_name}'. Let me know if you meant a different day.\n\n"
                + state["final_answer"]
            )

    return state

def call_tool_node(state: dict) -> dict:
    """Execute the requested tools and return results."""
    tool_calls = state.get("intermediate_steps", [])
    if not tool_calls:
        raise ValueError("No tool_calls found in state.")

    results = []
    booking_confirmation = None
    appointments_output = None
    role = state.get("user_role", "patient")  # default to patient

    for call in tool_calls:
        tool_name = call["tool_name"]
        arguments = dict(call["args"])

        # Inject weekday if relevant
        if tool_name == "suggest_appointment_slots" and "weekday" in state:
            arguments["weekday"] = state["weekday"]

        # Restrict access to certain tools
        if tool_name in SENSITIVE_TOOLS and role != "doctor":
            results.append(f"Access denied to `{tool_name}`. Doctor access required.")
            continue

        try:
            result = call_mcp_tool(tool_name, arguments)
            print(f"Tool '{tool_name}' result:", result)
            results.append(result)

            if tool_name == "book_appointment_tool":
                print("📥 Incoming booking payload:", arguments)
                booking_confirmation = result
            elif tool_name == "get_appointments":
                appointments_output = result

        except Exception as e:
            import traceback
            print(f"[ERROR] Tool '{tool_name}' failed:", e)
            traceback.print_exc()

            user_friendly_msg = (
                f"❌ Something went wrong while trying to use `{tool_name}`. "
                f"Please try again later or modify your request."
            )
            results.append(user_friendly_msg)

    return {
        **state,
        "tool_results": results,
        "booking_confirmation": booking_confirmation,
        "appointments_output": appointments_output,
        "next": "answer"
    }

def generate_final_answer(state: dict) -> dict:
    """Final answer serializer."""
    natural = state.get("final_answer", "")
    tool_outputs = state.get("tool_results", [])

    if tool_outputs:
        safe_results = []
        for item in tool_outputs:
            if isinstance(item, BaseModel):
                safe_results.append(item.model_dump())
            elif not isinstance(item, (str, int, float, dict, list)):
                safe_results.append(str(item))
            else:
                safe_results.append(item)

        tool_str = "\n".join([
            json.dumps(r, indent=2) if isinstance(r, (dict, list)) else str(r)
            for r in safe_results
        ])
        # Combine both
        final_output = f"{natural}\n\n{tool_str}"
    else:
        final_output = natural or "I couldn't process that."
        print("Final output generated:", final_output)
    
    return {
        **state,
        "final_answer": final_output
    }
