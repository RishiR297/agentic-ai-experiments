# # ==============================================
# # File: nodes.py
# # Purpose: Define LangGraph nodes that process the agent’s internal state
# # ==============================================

import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
from datetime import datetime, timedelta
from agent.state import AgentState
from agent.tools.appointment import MCP_TOOL_REGISTRY, MCP_FUNCTION_LOOKUP
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
print("✅ Loaded nodes.py at runtime")

# Azure OpenAI LLM setup
llm_with_tools = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview",
    model_kwargs={"tools": MCP_TOOL_REGISTRY}

)

# LLM without tools for prompting tasks like slot-filling
llm_basic = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview",
)

def extract_slots(user_input: str) -> dict:
    prompt = f"""
        You are a JSON parser. Extract the following fields from the user's message if present:
        - doctor_name
        - patient_name
        - branch_id
        - service_name
        - start_time
        - end_time
        - weekday (integer, where Monday = 0 and Sunday = 6)

        Only return a JSON object. Do not include explanations.

        Example format:
        {{
        "doctor_name": "Dr. Antonella",
        "weekday": 0
        }}

        User message: "{user_input}"
        """

    response = llm_basic.invoke([HumanMessage(content=prompt)])

    try:
        extracted = json.loads(response.content)
        if isinstance(extracted, dict):
            print("🧠 LLM extracted:", extracted)  # <== ADD THIS
            # Convert weekday → next date string like '2025-06-30T00:00:00'
            if "weekday" in extracted:
                now = datetime.now()
                print("🧪 Current datetime.now():", now)  # <== AND THIS
                today = now.date()
                weekday = extracted["weekday"]
                days_ahead = (weekday - today.weekday() + 7) % 7
                if days_ahead == 0:
                    days_ahead = 7  # always next, not today
                next_day = today + timedelta(days=days_ahead)
                next_day = datetime.combine(next_day, datetime.min.time())
                extracted["after"] = next_day.isoformat()
                print("🗓️ Computed next target date:", extracted["after"])  # <== AND THIS
            return extracted
        return {}
    except:
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

print("🧪 Tools registered:")
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

    # 🧠 Try to extract slots like doctor_name, start_time etc.
    extracted_fields = extract_slots(user_input)
    state.update(extracted_fields)
    state["tool_results"] = []  # ✅ Clear before new cycle starts
    # 🤖 Now invoke LLM with MCP tools
    response = llm_with_tools.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=user_input)
    ])

    clean_steps = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
                raw_args = tool_call.get("args", {})
            else:
                tool_name = getattr(tool_call, "name", None)
                raw_args = tool_call.args
                if hasattr(raw_args, "model_dump"):
                    raw_args = raw_args.model_dump()
                elif not isinstance(raw_args, dict):
                    raw_args = dict(raw_args)

            # ✅ Merge LLM args with extracted fields — extracted wins
            final_args = {**raw_args, **extracted_fields}
            # Force override if stale 'after' value detected from LLM
            if "after" in final_args:
                try:
                    parsed = datetime.fromisoformat(final_args["after"].split("T")[0])
                    if parsed.date() < datetime.now().date():
                        final_args["after"] = datetime.now().isoformat()
                except:
                    final_args["after"] = datetime.now().isoformat()

            print("🧪 Tool call:", tool_name)
            print("📦 Raw LLM args:", raw_args)
            print("🧠 Extracted fields:", extracted_fields)
            print("✅ Final merged args:", final_args)

            clean_steps.append({
                "tool_name": tool_name,
                "args": final_args
            })

        return {
            **state,
            "intermediate_steps": clean_steps,
        }

    else:
        print("⚠️ No tool calls returned by LLM.")
        return {
            **state,
            "intermediate_steps": [],
        }


# Router node
def route_node(state: dict) -> dict:
    steps = state.get("intermediate_steps", [])
    if not steps:
        state["next"] = "answer"
        return state

    # Check if all required fields are present
    missing = check_missing_fields(state)
    if missing:
        state["next"] = "ask_missing_info"
        state["missing_fields"] = missing
    else:
        state["next"] = "tool"
    return state


# ✅ FIX: Use MCP_TOOL_REGISTRY as is
TOOL_LOOKUP = MCP_FUNCTION_LOOKUP

def respond_naturally_node(state: dict) -> dict:
    steps = state.get("intermediate_steps", [])
    if not steps:
        return state

    tool_name = steps[0]["tool_name"]
    args = steps[0]["args"]
    doctor = args.get("doctor_name", state.get("doctor_name", "the doctor")).replace("Dr.", "").strip()

    service = args.get("service_name", "the requested service")
    weekday = state.get("weekday")

    if tool_name == "suggest_appointment_slots":
        tool_results = state.get("tool_results", [])
        slots_text = tool_results[0] if tool_results else ""

        if weekday is not None:
            weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]

            # ✅ check if any slot lines contain the weekday
            available_lines = [line for line in slots_text.splitlines() if weekday_name in line]

            if available_lines:
                message = f"✅ Yes, Dr. {doctor} is available on {weekday_name}. Here are the time slots:\n\n" + "\n".join(available_lines)
            else:
                message = f"❌ Dr. {doctor} is not available on {weekday_name}. But here are some nearby available slots:\n\n{slots_text}"
        else:
            message = f"👍 Let me check available time slots for Dr. {doctor}.\n\n{slots_text}"

        state["tool_results"] = []

    elif tool_name == "book_appointment_tool":
        message = f"📅 Booking your {service} with Dr. {doctor}. One moment..."

    elif tool_name == "get_appointments":
        message = f"🔍 Fetching current appointments for Dr. {doctor}."

    else:
        message = f"🧠 I'm working on your request using {tool_name}. Hang tight!"

    return {
        **state,
        "final_answer": message
    }

# Tool executor node
def call_tool_node(state: dict) -> dict:
    tool_calls = state.get("intermediate_steps", [])
    if not tool_calls:
        raise ValueError("No tool_calls found in state.")

    results = []
    booking_confirmation = None
    appointments_output = None

    for call in tool_calls:
        tool_name = call["tool_name"]
        arguments = dict(call["args"])
        
        # 👇 Inject weekday if present and relevant
        if tool_name == "suggest_appointment_slots" and "weekday" in state:
            arguments["weekday"] = state["weekday"]

        tool_fn = TOOL_LOOKUP.get(tool_name)
        if not tool_fn:
            results.append(f"❌ Unknown tool: {tool_name}")
        else:
            try:
                result = tool_fn.invoke(arguments)
                print(f"🔍 Tool '{tool_name}' result:", result)
                results.append(result)

                if tool_name == "book_appointment_tool":
                    booking_confirmation = result
                elif tool_name == "get_appointments":
                    appointments_output = result

            except Exception as e:
                results.append(f"❌ Error calling tool {tool_name}: {e}")


    return {
        **state,
        "tool_results": results,
        "booking_confirmation": booking_confirmation,
        "appointments_output": appointments_output,
        "next": "answer"
    }


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
        final_output = natural or "🤖 I couldn't process that."

    return {
        **state,
        "final_answer": final_output
    }
