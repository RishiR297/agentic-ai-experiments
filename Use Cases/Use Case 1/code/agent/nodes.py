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
SENSITIVE_TOOLS = {"get_next_client_info", "summarize_calendar_today"}

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

def authenticate_doctor(doctor_name: str, doctor_id: str) -> dict:
    """
    Authenticate doctor credentials against the database.
    Returns dict with success status and doctor info.
    Accepts either full GUID or just the first section before the hyphen.
    """
    try:
        from utils.db import get_db_connection
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print(f"[DEBUG] Authenticating: name='{doctor_name}', id='{doctor_id}'")
        
        # Query to verify doctor exists with given name and ID
        # Support both full GUID and first section of GUID
        # Clean doctor name by removing "Dr." prefix if present
        clean_name = doctor_name.replace("Dr.", "").strip()
        
        cursor.execute("""
            SELECT UserId, Firstname, Lastname, SpecialtyId, DisplayName, IsActive
            FROM COR_Doctor 
            WHERE (LOWER(Firstname || ' ' || Lastname) LIKE LOWER(?) 
                   OR LOWER(DisplayName) LIKE LOWER(?)
                   OR LOWER(Firstname) LIKE LOWER(?)
                   OR LOWER(Lastname) LIKE LOWER(?)) 
            AND (LOWER(UserId) = LOWER(?) OR LOWER(UserId) LIKE LOWER(?))
        """, (f"%{clean_name}%", f"%{clean_name}%", f"%{clean_name}%", f"%{clean_name}%", doctor_id, f"{doctor_id}-%"))
        
        result = cursor.fetchone()
        print(f"[DEBUG] Query result: {result}")
        conn.close()
        
        if result:
            user_id, firstname, lastname, specialty_id, display_name, is_active = result
            full_name = display_name if display_name else f"{firstname} {lastname}"
            
            # Warn if doctor is not active but still authenticate for testing
            status_msg = ""
            if not is_active:
                status_msg = " (Note: Doctor account is inactive)"
                
            return {
                "authenticated": True,
                "user_id": user_id,
                "name": full_name,
                "firstname": firstname,
                "lastname": lastname,
                "specialty_id": specialty_id,
                "is_active": bool(is_active),
                "status_message": status_msg
            }
        else:
            return {"authenticated": False, "error": "Invalid doctor credentials - name or ID not found"}
            
    except Exception as e:
        print(f"Authentication error: {e}")
        import traceback
        traceback.print_exc()
        return {"authenticated": False, "error": "Authentication system error"}

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
    # Show welcome message based on user role, not chat history
    if state.get("user_role"):
        if state["user_role"] == "doctor":
            doctor_name = state.get("doctor_authenticated_name", "Doctor")
            return {
                **state,
                "final_answer": f"👋 Welcome back, Dr. {doctor_name}! I can help you check your appointments, patient schedules, and daily summaries. How can I assist you today?",
                "next": "answer"
            }
        else:
            return {
                **state,
                "final_answer": "👋 Welcome back! I'm here to help you book appointments.\n\n🔹 Do you know which doctor you'd like to see? (Just tell me their name)\n🔹 Or would you like me to suggest doctors based on a service you need?\n\nHow can I assist you today?",
                "next": "answer"
            }
    else:
        # No role established - ask for identification
        return {
            **state,
            "final_answer": "👋 Welcome! Please identify yourself:\n\n🔹 Type 'patient' if you'd like to book an appointment\n🔹 Type 'doctor' followed by your name and ID if you're a doctor (e.g., 'doctor Dr. Smith ID 123ABC' or 'doctor Antonella ID 11712738')\n\nHow can I help you today?",
            "awaiting_role_identification": True,
            "next": "answer"
        }
    
    return state
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
    
    user_input = state.get("user_input", "")
    
    # Handle initialization case (empty user input)
    if not user_input.strip():
        print("[DEBUG] Empty user input - routing to welcome")
        return {**state, "next": "welcome"}
    
    user_input_lower = user_input.lower()

    # ----------------------------------------
    # Step 0: Check if user already has a role
    # ----------------------------------------
    user_role = state.get("user_role")
    
    # If user already has a role, skip role identification
    if user_role and not state.get("awaiting_role_identification"):
        print(f"[DEBUG] User already has role: {user_role}")
        # Continue to normal processing
        pass
    # ----------------------------------------
    # Step 0.1: Handle role identification for new users ONLY
    # ----------------------------------------
    elif not user_role:  # Only handle role identification if no role is set
        if "patient" in user_input_lower:
            state["user_role"] = "patient"
            state.pop("awaiting_role_identification", None)
            state["awaiting_doctor_or_service"] = True
            state["final_answer"] = "Great! I'm here to help you book appointments.\n\n🔹 Do you know which doctor you'd like to see? (Just tell me their name)\n🔹 Or would you like me to suggest doctors based on a service you need? (e.g., 'I need a checkup' or 'I need dental care')\n\nWhat can I help you with?"
            return {**state, "next": "answer"}
        
        elif "doctor" in user_input_lower:
            # Extract doctor name and ID from input
            # Expected format: "doctor Dr. Smith ID123" or "doctor John Smith ID456"
            import re
            
            # Try to extract doctor name and ID
            id_match = re.search(r'ID\s+([A-Fa-f0-9\-]+)', user_input, re.IGNORECASE)
            if not id_match:
                state["final_answer"] = "Please provide your doctor ID in the format: 'doctor Dr. YourName ID yourID' (e.g., 'doctor Antonella ID 11712738')"
                return {**state, "next": "answer"}
            
            doctor_id = id_match.group(1)
            
            # Extract doctor name (everything between "doctor" and "ID")
            name_match = re.search(r'doctor\s+(.+?)\s+ID', user_input, re.IGNORECASE)
            if not name_match:
                state["final_answer"] = "Please provide your name in the format: 'doctor Dr. YourName ID yourID' (e.g., 'doctor Antonella ID 11712738')"
                return {**state, "next": "answer"}
            
            doctor_name = name_match.group(1).strip()
            
            # Authenticate doctor
            auth_result = authenticate_doctor(doctor_name, doctor_id)
            
            if auth_result["authenticated"]:
                state["user_role"] = "doctor"
                state["doctor_authenticated_name"] = auth_result["name"]
                state["doctor_user_id"] = auth_result["user_id"]
                state["doctor_specialty_id"] = auth_result["specialty_id"]
                state.pop("awaiting_role_identification", None)
                
                state["final_answer"] = (
                    f"✅ Authentication successful! Welcome Dr. {auth_result['name']}.\n\n"
                    f"I can help you with:\n"
                    f"🔹 Check your upcoming appointments\n"
                    f"🔹 View today's schedule\n"
                    f"🔹 Get next patient information\n"
                    f"🔹 Summarize your daily calendar\n\n"
                    f"What would you like to do?"
                )
                return {**state, "next": "answer"}
            else:
                state["final_answer"] = f"❌ Authentication failed: {auth_result.get('error', 'Invalid credentials')}. Please try again with the correct format: 'doctor Dr. YourName ID yourID' (e.g., 'doctor Antonella ID 11712738')"
                return {**state, "next": "answer"}
        
        else:
            # If we're awaiting role identification but no role keywords found
            if state.get("awaiting_role_identification"):
                state["final_answer"] = "Please specify if you are a 'patient' or a 'doctor' (with your name and ID)."
                return {**state, "next": "answer"}
            else:
                # No role set yet, prompt for identification
                state["awaiting_role_identification"] = True
                state["final_answer"] = "Please identify yourself:\n\n🔹 Type 'patient' if you'd like to book an appointment.\n🔹 Type 'doctor' followed by your name and ID if you're a doctor (e.g., 'doctor Dr. Smith ID 123ABC' or 'doctor Antonella ID 11712738').\n\nHow can I assist you today?"
                return {**state, "next": "answer"}

    # ----------------------------------------
    # Step 0.2: Handle patient doctor/service selection
    # ----------------------------------------
    if state.get("awaiting_doctor_or_service") and state.get("user_role") == "patient":
        # Check if user mentioned a doctor name
        extracted_fields = extract_slots(user_input)
        doctor_name = extracted_fields.get("doctor_name")
        
        if doctor_name:
            # User specified a doctor name
            state["doctor_name"] = doctor_name
            state.pop("awaiting_doctor_or_service", None)
            state["awaiting_service_selection"] = True
            
            # Autofill branch_id
            if not state.get("branch_id"):
                inferred_branch_id = get_branch_id_for_doctor(doctor_name)
                if inferred_branch_id:
                    state["branch_id"] = inferred_branch_id
            
            state["final_answer"] = f"Perfect! You'd like to see {doctor_name}.\n\nWhat service do you need? Here are the available services:\n\n🔹 **CONSULTATION** - General consultations and assessments\n🔹 **FACIAL** - Facial treatments and skin care\n🔹 **BOTOX** - Anti-aging and wrinkle treatments\n🔹 **BBL** - Laser skin treatments\n🔹 **SCULPTRA** - Facial sculpting treatments\n🔹 **EVOLVE X** - Body contouring\n🔹 **PRP** - Platelet-rich plasma therapy\n🔹 **DNA TEST** - Genetic testing services\n\nWhich service would you like to book?"
            return {**state, "next": "answer"}
        
        # Check if user is asking for service-based suggestions
        service_keywords = ["need", "want", "looking for", "service", "checkup", "consultation", "treatment", "care", "appointment for"]
        if any(keyword in user_input_lower for keyword in service_keywords):
            # User is asking for service-based suggestions
            # Extract potential service name
            service_name = extracted_fields.get("service_name")
            
            # Map common terms to actual services in database
            service_mapping = {
                "checkup": "CONSULTATION",
                "general checkup": "CONSULTATION", 
                "consultation": "CONSULTATION",
                "facial": "FACIAL",
                "botox": "BOTOX",
                "beauty": "FACIAL",
                "skin care": "FACIAL",
                "anti-aging": "BOTOX",
                "wrinkles": "BOTOX",
                "skin treatment": "BBL",
                "laser": "BBL",
                "sculpting": "SCULPTRA",
                "body contouring": "EVOLVE X",
                "prp": "PRP",
                "platelet": "PRP",
                "dna": "DNA TEST",
                "genetic": "DNA TEST"
            }
            
            # Try to map user input to actual service
            mapped_service = None
            if service_name:
                # Direct match first
                mapped_service = service_name.upper()
                # Try mapping if not found
                if not mapped_service or mapped_service not in ["FACIAL", "ULTHERAPY", "BBL", "BOTOX", "DNA TEST", "CONSULTATION", "SCULPTRA", "RADIESSE", "EVOLVE X", "PRP"]:
                    for key, value in service_mapping.items():
                        if key.lower() in service_name.lower():
                            mapped_service = value
                            break
            else:
                # Try to find service keywords in user input
                for key, value in service_mapping.items():
                    if key.lower() in user_input_lower:
                        mapped_service = value
                        break
            
            if mapped_service:
                # Try to find doctors who provide this service
                doctors_with_service = suggest_doctor_for_service(mapped_service)
                if doctors_with_service:
                    state.pop("awaiting_doctor_or_service", None)
                    state["awaiting_doctor_selection"] = True
                    state["suggested_service"] = mapped_service
                    state["final_answer"] = f"For '{mapped_service.lower()}', I can suggest these doctors:\n\n" + "\n".join([f"🔹 {doc}" for doc in doctors_with_service[:5]]) + f"\n\nWhich doctor would you prefer?"
                    return {**state, "next": "answer"}
            
            # If no service match found, show available services
            state["final_answer"] = (
                "I'd be happy to suggest doctors based on the service you need. Here are the services we offer:\n\n"
                "🔹 **CONSULTATION** - General consultations and assessments\n"
                "🔹 **FACIAL** - Facial treatments and skin care\n" 
                "🔹 **BOTOX** - Anti-aging and wrinkle treatments\n"
                "🔹 **BBL** - Laser skin treatments\n"
                "🔹 **SCULPTRA** - Facial sculpting treatments\n"
                "🔹 **EVOLVE X** - Body contouring\n"
                "🔹 **PRP** - Platelet-rich plasma therapy\n"
                "🔹 **DNA TEST** - Genetic testing services\n\n"
                "Which service interests you? Or if you know a specific doctor's name, just tell me!"
            )
            return {**state, "next": "answer"}
        
        # User didn't specify doctor or clear service request
        state["final_answer"] = "I can help you in two ways:\n\n🔹 If you know a doctor's name, just tell me (e.g., 'Dr. Smith')\n🔹 If you need help choosing, tell me what kind of appointment you need (e.g., 'I need a checkup')\n\nWhat would you prefer?"
        return {**state, "next": "answer"}
    
    # ----------------------------------------
    # Step 0.3: Handle doctor selection from suggested list  
    # ----------------------------------------
    if state.get("awaiting_doctor_selection") and state.get("user_role") == "patient":
        extracted_fields = extract_slots(user_input)
        doctor_name = extracted_fields.get("doctor_name")
        
        if doctor_name:
            state["doctor_name"] = doctor_name
            state["service_name"] = state.get("suggested_service")  # Use the suggested service
            state.pop("awaiting_doctor_selection", None)
            state.pop("suggested_service", None)
            state["awaiting_appointment_time"] = True
            
            # Autofill branch_id
            if not state.get("branch_id"):
                inferred_branch_id = get_branch_id_for_doctor(doctor_name)
                if inferred_branch_id:
                    state["branch_id"] = inferred_branch_id
            
            state["final_answer"] = f"Excellent choice! You'd like to see {doctor_name}.\n\nWhen would you like to schedule your appointment? You can say:\n🔹 'Today' or 'Tomorrow'\n🔹 A specific day like 'Monday' or 'Next Tuesday'\n🔹 'This week' to see available slots"
            return {**state, "next": "answer"}
        else:
            state["final_answer"] = "Please choose one of the suggested doctors by mentioning their name, or tell me if you'd like to see different options."
            return {**state, "next": "answer"}
    
    # ----------------------------------------
    # Step 0.3b: Handle service selection for the chosen doctor
    # ----------------------------------------
    if state.get("awaiting_service_selection") and state.get("user_role") == "patient":
        # Extract service name from user input
        extracted_fields = extract_slots(user_input)
        service_name = extracted_fields.get("service_name")
        
        # Map common terms to actual services in database
        service_mapping = {
            "checkup": "CONSULTATION",
            "general checkup": "CONSULTATION", 
            "consultation": "CONSULTATION",
            "facial": "FACIAL",
            "botox": "BOTOX",
            "beauty": "FACIAL",
            "skin care": "FACIAL",
            "anti-aging": "BOTOX",
            "wrinkles": "BOTOX",
            "skin treatment": "BBL",
            "laser": "BBL",
            "sculpting": "SCULPTRA",
            "body contouring": "EVOLVE X",
            "prp": "PRP",
            "platelet": "PRP",
            "dna": "DNA TEST",
            "genetic": "DNA TEST"
        }
        
        # Try to map user input to actual service
        mapped_service = None
        if service_name:
            # Direct match first
            mapped_service = service_name.upper()
            # Try mapping if not found in valid services
            if mapped_service not in ["FACIAL", "ULTHERAPY", "BBL", "BOTOX", "DNA TEST", "CONSULTATION", "SCULPTRA", "RADIESSE", "EVOLVE X", "PRP"]:
                for key, value in service_mapping.items():
                    if key.lower() in service_name.lower():
                        mapped_service = value
                        break
        else:
            # Try to find service keywords in user input
            for key, value in service_mapping.items():
                if key.lower() in user_input_lower:
                    mapped_service = value
                    break
        
        if mapped_service and mapped_service in ["FACIAL", "ULTHERAPY", "BBL", "BOTOX", "DNA TEST", "CONSULTATION", "SCULPTRA", "RADIESSE", "EVOLVE X", "PRP"]:
            # Valid service selected
            state["service_name"] = mapped_service
            state.pop("awaiting_service_selection", None)
            state["awaiting_appointment_time"] = True
            
            state["final_answer"] = f"Great! You'd like to book a {mapped_service.lower()} appointment with {state.get('doctor_name')}.\n\nWhen would you like to schedule your appointment? You can say:\n🔹 'Today' or 'Tomorrow'\n🔹 A specific day like 'Monday' or 'Next Tuesday'\n🔹 'This week' to see available slots"
            return {**state, "next": "answer"}
        else:
            # Invalid or no service selected
            state["final_answer"] = "Please select one of the available services:\n\n🔹 **CONSULTATION** - General consultations and assessments\n🔹 **FACIAL** - Facial treatments and skin care\n🔹 **BOTOX** - Anti-aging and wrinkle treatments\n🔹 **BBL** - Laser skin treatments\n🔹 **SCULPTRA** - Facial sculpting treatments\n🔹 **EVOLVE X** - Body contouring\n🔹 **PRP** - Platelet-rich plasma therapy\n🔹 **DNA TEST** - Genetic testing services\n\nWhich service would you like to book?"
            return {**state, "next": "answer"}

    # ----------------------------------------
    # Step 0.4: Handle appointment time selection
    # ----------------------------------------
    if state.get("awaiting_appointment_time") and state.get("user_role") == "patient":
        # Handle "today", "tomorrow", specific days, etc.
        if "today" in user_input_lower:
            today_weekday = datetime.now().weekday()
            state["weekday"] = today_weekday
            state.pop("awaiting_appointment_time", None)
            state["intermediate_steps"] = [{
                "tool_name": "suggest_appointment_slots",
                "args": {
                    "doctor_name": state.get("doctor_name"),
                    "weekday": today_weekday,
                    "service_name": state.get("service_name", "CONSULTATION")
                }
            }]
            return {**state, "next": "tool"}
        
        elif "tomorrow" in user_input_lower:
            tomorrow_weekday = (datetime.now().weekday() + 1) % 7
            state["weekday"] = tomorrow_weekday
            state.pop("awaiting_appointment_time", None)
            state["intermediate_steps"] = [{
                "tool_name": "suggest_appointment_slots",
                "args": {
                    "doctor_name": state.get("doctor_name"),
                    "weekday": tomorrow_weekday,
                    "service_name": state.get("service_name", "CONSULTATION")
                }
            }]
            return {**state, "next": "tool"}
        
        elif "this week" in user_input_lower:
            state.pop("awaiting_appointment_time", None)
            state["intermediate_steps"] = [{
                "tool_name": "suggest_appointment_slots", 
                "args": {
                    "doctor_name": state.get("doctor_name"),
                    "service_name": state.get("service_name", "CONSULTATION")
                }
            }]
            return {**state, "next": "tool"}
        
        else:
            # Try to extract specific weekday
            extracted_fields = extract_slots(user_input)
            weekday = extracted_fields.get("weekday")
            
            if weekday is not None:
                state["weekday"] = weekday
                state.pop("awaiting_appointment_time", None)
                state["intermediate_steps"] = [{
                    "tool_name": "suggest_appointment_slots",
                    "args": {
                        "doctor_name": state.get("doctor_name"),
                        "weekday": weekday
                    }
                }]
                return {**state, "next": "tool"}
            else:
                state["final_answer"] = "When would you like your appointment? You can say:\n🔹 'Today' or 'Tomorrow'\n🔹 A specific day like 'Monday', 'Tuesday', etc.\n🔹 'This week' to see all available slots"
                return {**state, "next": "answer"}

    # ----------------------------------------
    # Step 1: Extract structured fields (for existing flow)
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

    # ----------------------------------------
    # Doctor-specific queries
    # ----------------------------------------
    if state.get("user_role") == "doctor":
        doctor_name = state.get("doctor_authenticated_name", "")
        
        # Handle doctor asking for their own appointments with improved time awareness
        if any(kw in user_input_lower for kw in ["my appointments", "my schedule", "my patients", "appointments", "upcoming appointments", "check appointments"]):
            # Determine time filter based on user input
            after_date = None
            limit = 10  # Default limit
            
            if any(kw in user_input_lower for kw in ["today", "today's"]):
                after_date = datetime.now().strftime('%Y-%m-%d')
                limit = 20  # More for daily view
            elif any(kw in user_input_lower for kw in ["this week", "week", "upcoming"]):
                after_date = datetime.now().isoformat()
                limit = 15  # Week view
            elif any(kw in user_input_lower for kw in ["next week"]):
                next_week = datetime.now() + timedelta(weeks=1)
                after_date = next_week.isoformat()
                limit = 15
            else:
                # Default: upcoming appointments from now, limited to next week
                after_date = datetime.now().isoformat()
                limit = 10
            
            # Prepare the tool arguments
            tool_args = {
                "doctor_name": f"Dr. {doctor_name.split()[-1]}" if doctor_name else "",
                "limit": limit
            }
            
            if after_date:
                tool_args["after"] = after_date
            
            state["intermediate_steps"] = [{
                "tool_name": "get_appointments",
                "args": tool_args
            }]
            return {**state, "next": "tool"}
        
        # Handle doctor asking for next patient
        if any(kw in user_input_lower for kw in ["next patient", "next client", "who's next"]):
            state["intermediate_steps"] = [{
                "tool_name": "get_next_client_info",
                "args": {
                    "doctor_name": f"Dr. {doctor_name.split()[-1]}" if doctor_name else ""
                }
            }]
            return {**state, "next": "tool"}
        
        # Handle doctor asking for daily summary
        if any(kw in user_input_lower for kw in ["summarize today", "daily summary", "today's summary", "calendar summary"]):
            state["intermediate_steps"] = [{
                "tool_name": "summarize_calendar_today",
                "args": {
                    "doctor_name": f"Dr. {doctor_name.split()[-1]}" if doctor_name else ""
                }
            }]
            return {**state, "next": "tool"}

    # Handle user asking if doctor is available "today" (patient flow)
    if state.get("user_role") == "patient" and any(kw in user_input.lower() for kw in ["available today", "today", "free today", "open today"]):
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
    
    # Validate service name (only when service is first provided, not during slot selection)
    if (state.get("doctor_name") and state.get("service_name") and 
        not state.get("awaiting_slot_selection") and 
        not state.get("awaiting_confirmation") and 
        not state.get("proposed_booking")):
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
        
        weekday_name = None
        is_today_request = False
        
        if weekday is not None:
            weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]
            # Check if this is a "today" request
            current_weekday = datetime.now().weekday()
            is_today_request = weekday == current_weekday

        # Simply display the slots text as-is since it's already formatted
        if slots_text and "🎯 Available appointment slots:" in slots_text:
            # Extract the appointment slot lines for selection
            lines = slots_text.splitlines()
            available_slots = []
            
            for line in lines:
                if line.strip() and "📅" in line:
                    # Extract individual slots from the line
                    # Format: 📅 Tuesday, Jul 08: 10:00-10:15, 10:15-10:30, 10:30-10:45 (15 min slots)
                    try:
                        # Remove emoji and extract date part
                        cleaned_line = line.replace("📅", "").strip()
                        if ":" in cleaned_line:
                            date_part, time_part = cleaned_line.split(":", 1)
                            date_part = date_part.strip()
                            
                            # Extract individual time slots before the parenthetical
                            if "(" in time_part:
                                time_part = time_part.split("(")[0].strip()
                            
                            # Split time slots by comma
                            time_slots = [slot.strip() for slot in time_part.split(",")]
                            
                            for time_slot in time_slots:
                                if "-" in time_slot and ":" in time_slot:
                                    # Parse individual slot like "10:00-10:15"
                                    start_time, end_time = time_slot.split("-")
                                    start_time = start_time.strip()
                                    end_time = end_time.strip()
                                    
                                    # Create a formatted line for this individual slot
                                    individual_slot_line = f"{date_part}: {start_time} - {end_time}"
                                    
                                    # Try to parse this individual slot
                                    slot_info = parse_slot_line(individual_slot_line)
                                    if slot_info:
                                        available_slots.append({
                                            "start_time": slot_info["start_time"],
                                            "end_time": slot_info["end_time"],
                                            "display": individual_slot_line
                                        })
                                    else:
                                        # Fallback - create a display entry without parsing
                                        available_slots.append({
                                            "start_time": None,
                                            "end_time": None,
                                            "display": individual_slot_line
                                        })
                    except Exception as e:
                        print(f"Error parsing slot line: {line}, error: {e}")
                        # If all parsing fails, just include the original line
                        available_slots.append({
                            "start_time": None,
                            "end_time": None,
                            "display": line.strip()
                        })
            
            if is_today_request:
                message = (
                    f"Yes, Dr. {doctor} is available today ({weekday_name}). "
                    f"Here are the available time slots:\n\n{slots_text}\n\n"
                    f"Which slot would you like to book?"
                )
            else:
                message = (
                    f"Yes, Dr. {doctor} is available"
                    + (f" on {weekday_name}" if weekday_name else "")
                    + f". Here are the time slots:\n\n{slots_text}\n\n"
                    + f"Which slot would you like to book?"
                )
            
            state["available_slot_lines"] = available_slots
        else:
            # No slots available or error message
            if is_today_request:
                message = (
                    f"Dr. {doctor} is not available today ({weekday_name}). "
                    f"Here's what I found:\n\n{slots_text}"
                )
            elif weekday_name:
                message = (
                    f"Dr. {doctor} is not available on {weekday_name}. "
                    f"Here's what I found:\n\n{slots_text}"
                )
            else:
                message = f"Here are the available slots for Dr. {doctor}:\n\n{slots_text}"

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

        # Enhanced access control for sensitive tools
        if tool_name in SENSITIVE_TOOLS:
            if role != "doctor":
                results.append(f"❌ Access denied to `{tool_name}`. Doctor authentication required.")
                continue
            else:
                # Verify doctor is accessing their own data
                requested_doctor = arguments.get("doctor_name", "")
                authenticated_doctor = state.get("doctor_authenticated_name", "")
                
                if requested_doctor and authenticated_doctor:
                    # Extract last name for comparison
                    requested_lastname = requested_doctor.replace("Dr.", "").strip().split()[-1].lower()
                    auth_lastname = authenticated_doctor.split()[-1].lower()
                    
                    if requested_lastname != auth_lastname:
                        results.append(f"❌ Access denied. You can only access your own appointment information.")
                        continue

        try:
            result = call_mcp_tool(tool_name, arguments)
            print(f"Tool '{tool_name}' result:", result)
            results.append(result)

            if tool_name == "book_appointment_tool":
                print("📥 Incoming booking payload:", arguments)
                booking_confirmation = result
            elif tool_name in ["get_appointments", "get_next_client_info", "summarize_calendar_today"]:
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
