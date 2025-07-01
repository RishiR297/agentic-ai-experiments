# # ==============================================
# # File: nodes.py
# # Purpose: Define LangGraph nodes that process the agent’s internal state
# # ==============================================
import re
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
from typing import Optional
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from agent.state import AgentState
from tool_server import MCP_TOOL_REGISTRY, MCP_FUNCTION_LOOKUP
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from tools.doctor import get_branch_id_for_doctor, get_services_for_doctor, is_service_valid_for_doctor, suggest_doctor_for_service
from agent.tools.mcp_client import call_mcp_tool


load_dotenv()
print("Loaded nodes.py at runtime")

# Safe conversion to avoid serialization errors
SAFE_TOOL_REGISTRY = [
    tool if isinstance(tool, dict) else tool.dict() for tool in MCP_TOOL_REGISTRY
]


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

def welcome_node(state: dict) -> dict:
    if not state.get("chat_history"):
        return {
            **state,
            "final_answer": "👋 Hi! How can I help you today?",
            "next": "answer"
        }
    return state

def extract_slots(user_input: str) -> dict:
    prompt = f"""
    You are a JSON extractor. Given a user's message, extract the following fields if present:
    - doctor_name
    - patient_name
    - branch_id
    - service_name
    - start_time
    - end_time
    - weekday (0 for Monday to 6 for Sunday)

    Return a JSON object ONLY. No explanation.

    Example:
    {{
        "doctor_name": "Dr. Antonella",
        "weekday": 1
    }}

    User: "{user_input}"
    """

    try:
        response = llm_basic.invoke([HumanMessage(content=prompt)])
        extracted = json.loads(response.content)

        if isinstance(extracted, dict):
            print("Extracted fields:", extracted)

            # Handle weekday → ISO after
            if "weekday" in extracted:
                now = datetime.now()
                weekday = extracted["weekday"]
                days_ahead = (weekday - now.weekday() + 7) % 7
                next_date = now + timedelta(days=days_ahead)

                extracted["after"] = next_date.isoformat()
                print("Interpreted weekday as:", next_date.date())

            return extracted
        return {}
    except Exception as e:
        print("extract_slots failed:", e)
        return {}


    
def check_missing_fields(state: dict) -> list:
    """
    Checks which required fields are missing for booking.
    """
    tool_name = state.get("intermediate_steps", [{}])[0].get("tool_name")
    if tool_name != "book_appointment_tool":
        return []

    missing_fields = []
    for field in AgentState.REQUIRED_FIELDS:
        if state.get(field) is None:
            missing_fields.append(field)
    return missing_fields

def ask_for_missing_fields_node(state: dict) -> dict:
    missing_fields = state.get("missing_fields", [])
    if not missing_fields:
        return state

    prompts = {
        "doctor_name": "the doctor's name",
        "patient_name": "your name",
        "branch_id": "the branch ID",
        "service_name": "the service needed",
        "start_time": "the preferred start time",
        "end_time": "the preferred end time"
    }

    questions = [f"Please provide {prompts.get(field, field)}." for field in missing_fields]
    message = "To proceed with booking, I need the following:\n" + "\n".join(questions)

    return {
        **state,
        "final_answer": message
    }

print("Tools registered:")
for i, tool in enumerate(MCP_TOOL_REGISTRY):
    print(f"Tool #{i + 1}: type={type(tool)}")
    print(json.dumps(tool, indent=2) if isinstance(tool, dict) else tool)

for tool in MCP_TOOL_REGISTRY:
    print(f"- {tool['function']['name']}")


# MCP-based planner node
MCP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that decides the next action."),
    ("human", "{user_input}")
])


def planner_node(state: dict) -> dict:
    user_input = state.get("user_input")
    if not user_input:
        raise ValueError("Missing 'user_input' in agent state")

    # ----------------------------------------
    # 🧠 Step 1: Extract structured fields
    # ----------------------------------------
    extracted_fields = {}
    should_extract = not state.get("awaiting_confirmation") and not state.get("available_slot_lines")

    if should_extract:
        try:
            extracted_fields = extract_slots(user_input)
            extracted_fields.setdefault("patient_name", "User")
            state.update(extracted_fields)
        except Exception as e:
            print(f"extract_slots failed: {e}")

    # Autofill branch_id if missing but doctor known
    if state.get("doctor_name") and not state.get("branch_id"):
        inferred_branch_id = get_branch_id_for_doctor(state["doctor_name"])
        if inferred_branch_id:
            state["branch_id"] = inferred_branch_id

    # Clear previous outputs
    state.pop("final_answer", None)
    state["tool_results"] = []

    # ----------------------------------------
    # 🔁 Step 2: Handle booking confirmation
    # ----------------------------------------
    if state.get("awaiting_confirmation"):
        user_input_lower = user_input.lower()
        if user_input_lower in ["yes", "yeah", "yep", "confirm", "go ahead", "sure"]:
            # Confirm booking
            state["intermediate_steps"] = [state.pop("proposed_booking")]
            state["awaiting_confirmation"] = False
            return state
        else:
            # Cancel booking
            state["awaiting_confirmation"] = False
            state.pop("proposed_booking", None)
            state.pop("start_time", None)
            state.pop("end_time", None)
            state["final_answer"] = "Okay, booking cancelled. Let me know if you'd like to try a different slot."
            state.pop("available_slot_lines", None)

            return {**state, "next": "answer"}

    # ----------------------------------------
    # 🗓️ Step 3: Detect slot selection
    # ----------------------------------------
    selected = detect_selected_slot(state)
    if selected:
        # User selected a slot -> prepare confirmation
        start_time = selected.get("start_time")
        end_time = selected.get("end_time")
        doctor = state.get("doctor_name", "the doctor")
        patient = state.get("patient_name", "User")

        # 🔁 Autofill missing branch ID now based on doctor
        if not state.get("branch_id") and doctor:
            inferred_branch_id = get_branch_id_for_doctor(doctor)
            print(f"[DEBUG autofill] Inferred branch ID for {doctor}: {inferred_branch_id}")
            if inferred_branch_id:
                state["branch_id"] = inferred_branch_id

        state["awaiting_confirmation"] = True
        state["start_time"] = start_time
        state["end_time"] = end_time

        state["proposed_booking"] = {
            "tool_name": "book_appointment_tool",
            "args": {
                "doctor_name": doctor,
                "start_time": start_time,
                "end_time": end_time,
                "patient_name": patient,
                "branch_id": state.get("branch_id"),
                "service_name": state.get("service_name"),
            },
        }
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
    # 🤖 Step 4: Fallback to LLM planner
    # ----------------------------------------
    
    if not state.get("awaiting_confirmation"):
        # Build tool call for suggesting slots if doctor_name & weekday known
        if "doctor_name" in state and "weekday" in state:
            state["intermediate_steps"] = [{
                "tool_name": "suggest_appointment_slots",
                "args": {
                    "doctor_name": state["doctor_name"],
                    "weekday": state["weekday"]
                }
            }]
            return state
        
        
    system_prompt = SystemMessage(
        content="You are a helpful assistant. Use the available tools to assist the user."
    )
    chat_history = state.get("chat_history", [])[-6:]
    messages: list[BaseMessage] = [system_prompt] + chat_history


    response = llm_with_tools.invoke(messages)

    if not getattr(response, "tool_calls", None):
        chat_history.append(response)

    # ----------------------------------------
    # 🛠️ Step 5: Parse tool calls
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

            # ✅ Merge only expected arguments
            allowed_keys = set(raw_args.keys())
            filtered_fields = {k: v for k, v in extracted_fields.items() if k in allowed_keys}
            final_args = {**raw_args, **filtered_fields}

            # 🛠 Fix outdated 'after'
            if "after" in final_args:
                try:
                    parsed = datetime.fromisoformat(final_args["after"].split("T")[0])
                    if parsed.date() < datetime.now().date():
                        final_args["after"] = datetime.now().isoformat()
                except:
                    final_args["after"] = datetime.now().isoformat()

            # ⚠️ Weekday mismatch warning
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
    # ❓ Step 6: No tool calls fallback
    # ----------------------------------------
    print("No tool calls returned by LLM.")
    return {
        **state,
        "intermediate_steps": [],
        "chat_history": chat_history,
        "final_answer": "I couldn't process that. Could you please rephrase or provide more details?"
    }



# Router node
def route_node(state: dict) -> dict:
    steps = state.get("intermediate_steps", [])
    if not steps:
        state["next"] = "answer"
        return state

    # ✅ If we already have all tool args
    missing = check_missing_fields(state)
    if missing:
        state["next"] = "ask_missing_info"
        state["missing_fields"] = missing
    else:
        state["next"] = "tool"

    return state



# ✅ FIX: Use MCP_TOOL_REGISTRY as is
TOOL_LOOKUP = MCP_FUNCTION_LOOKUP

import difflib
from datetime import datetime

def clean_date_line(line: str) -> str:
    # Remove emoji but preserve the original formatting
    return line.replace("📅", "").strip()

def respond_naturally_node(state: dict) -> dict:
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
        if weekday is not None:
            weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]

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
            message = (
                f"Yes, Dr. {doctor} is available"
                + (f" on {weekday_name}" if weekday_name else "")
                + ". Here are the time slots:\n\n"
                + "\n".join([slot["display"] for slot in available_slots])
                + "\n\nWhich slot would you like to book?"
            )
            state["available_slot_lines"] = available_slots
        else:
            if weekday_name:
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


SENSITIVE_TOOLS = {"get_next_client_info", "summarize_appointments"}

def call_tool_node(state: dict) -> dict:
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

        # 🔐 Restrict access to certain tools
        if tool_name in SENSITIVE_TOOLS and role != "doctor":
            results.append(f"Access denied to `{tool_name}`. Doctor access required.")
            continue

        try:
            result = call_mcp_tool(tool_name, arguments)
            print(f"Tool '{tool_name}' result:", result)
            results.append(result)

            if tool_name == "book_appointment_tool":
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



def detect_selected_slot(state: dict) -> dict:
    """
    Detects if user input refers to a specific available slot.
    Parses natural language dates or index references like "the second slot".
    Returns a dict with start_time and end_time if match is found.
    """
    user_input = state.get("user_input", "").lower()
    slot_lines = state.get("available_slot_lines", [])  # Now a list of dicts
    print(f"[DEBUG detect_selected_slot] user_input: '{user_input}'")
    print(f"[DEBUG detect_selected_slot] available slots: {slot_lines}")

    ordinal_map = {
        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
        "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4
    }
    # Match ordinal words like "first slot"
    for word, idx in ordinal_map.items():
        if word in user_input and idx < len(slot_lines):
            print(f"[DEBUG detect_selected_slot] Matched ordinal '{word}' to slot index {idx}")
            return {
                "start_time": slot_lines[idx]["start_time"],
                "end_time": slot_lines[idx]["end_time"]
            }

    # Match date phrases like "July 7"
    for slot in slot_lines:
        display = slot.get("display", "").lower()
        # We'll try matching month and day from user input against display
        # Extract month and day from user input
        # For simplicity, regex find all month/day combos in user input
        import re
        date_matches = re.findall(r"(\b\w{3,9}) (\d{1,2})(?:st|nd|rd|th)?", user_input)
        for (month_str, day_str) in date_matches:
            for suffix in ["", "st", "nd", "rd", "th"]:
                date_phrase = f"{month_str} {int(day_str)}{suffix}".lower()
                if date_phrase in display:
                    print(f"[DEBUG detect_selected_slot] Matched date phrase '{date_phrase}' in slot: {display}")
                    return {
                        "start_time": slot["start_time"],
                        "end_time": slot["end_time"]
                    }

    print("[DEBUG detect_selected_slot] No slot matched.")
    return {}


def parse_slot_line(line: str) -> dict:
    match = re.search(r"(\w{3,}), (\w{3}) (\d{1,2}): (\d{1,2}:\d{2}) - (\d{1,2}:\d{2})", line)
    if not match:
        return {}

    _, month, day, start_t, end_t = match.groups()
    try:
        base_year = datetime.now().year
        slot_date = date_parser.parse(f"{month} {day} {base_year}")
        # Ensure we always return the next *valid future date*
        now = datetime.now()
        slot_date = date_parser.parse(f"{month} {day} {base_year}")
        if slot_date.date() < now.date():
            slot_date = date_parser.parse(f"{month} {day} {base_year + 1}")


        start_dt = datetime.combine(slot_date.date(), datetime.strptime(start_t, "%H:%M").time())
        end_dt = datetime.combine(slot_date.date(), datetime.strptime(end_t, "%H:%M").time())

        return {
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat()
        }
    except Exception as e:
        print(f"[Slot Parse Error] {e}")
        return {}

    
# Final answer serializer

def generate_final_answer(state: dict) -> dict:
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
        # 👇 Combine both
        final_output = f"{natural}\n\n{tool_str}"
    else:
        final_output = natural or "I couldn't process that."
        print("Final output generated:", final_output)
    return {
        **state,
        "final_answer": final_output
    }
