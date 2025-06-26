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
from tool_server import MCP_TOOL_REGISTRY, MCP_FUNCTION_LOOKUP
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from agent.tools.mcp_client import call_mcp_tool

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
            print("🧠 Extracted fields:", extracted)

            # Handle weekday → ISO after
            if "weekday" in extracted:
                now = datetime.now()
                weekday = extracted["weekday"]
                days_ahead = (weekday - now.weekday() + 7) % 7 or 7
                next_date = datetime.combine(now.date() + timedelta(days=days_ahead), datetime.min.time())
                extracted["after"] = next_date.isoformat()
                print("📅 Interpreted weekday as:", next_date.date())

            return extracted
        return {}
    except Exception as e:
        print("❌ extract_slots failed:", e)
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

    # 🧠 Extract fields like doctor_name, weekday, etc.
    extracted_fields = extract_slots(user_input)
    state.update(extracted_fields)
    state["tool_results"] = []  # Clear previous tool results

    # 🧠 Load short-term memory (chat history)
    chat_history = state.get("chat_history", [])[-6:]  # Keep last 6 messages
    chat_history.append(HumanMessage(content=user_input))

    # 📢 System prompt (important!)
    system_prompt = SystemMessage(
        content="You are a helpful assistant. You can use the available tools to assist the user. "
                "If the user wants to check availability, suggest slots, or book, use the relevant tool."
    )

    # 🧠 Compose full message sequence
    messages: list[BaseMessage] = [system_prompt] + chat_history

    # 🧪 Invoke LLM
    response = llm_with_tools.invoke(messages)

    # 🧠 Append LLM reply to memory
    if hasattr(response, "content"):
        chat_history.append(response)

    # 🔧 Tool call extraction
    clean_steps = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name")
                raw_args = tool_call.get("args", {})
            else:
                tool_name = getattr(tool_call, "name", None)
                raw_args = getattr(tool_call, "args", {})
                if hasattr(raw_args, "model_dump"):
                    raw_args = raw_args.model_dump()
                elif not isinstance(raw_args, dict):
                    raw_args = dict(raw_args)

            # 🧠 Merge extracted values over LLM-suggested args
            final_args = {**raw_args, **extracted_fields}

            # 🛠 Fix old timestamps
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
            "chat_history": chat_history
        }

    # ❌ No tool calls made — fallback
    print("⚠️ No tool calls returned by LLM.")
    return {
        **state,
        "intermediate_steps": [],
        "chat_history": chat_history,
        "final_answer": "🤖 I couldn't process that. Could you please rephrase or provide more details?"
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

    if tool_name == "suggest_appointment_slots":
        tool_results = state.get("tool_results", [])
        slots_text = tool_results[0] if tool_results else ""
        lines = slots_text.splitlines()
        available_lines = []

        if weekday is not None:
            weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday]

            for line in lines:
                try:
                    cleaned = clean_date_line(line)
                    date_part = cleaned.split(":")[0].strip()
                    date_with_year = f"{date_part} {datetime.now().year}"
                    parsed_date = datetime.strptime(date_with_year, "%A, %b %d %Y")

                    if parsed_date.weekday() == weekday:
                        available_lines.append(line)
                except Exception:
                    # Fallback: include line if weekday name is in it
                    if weekday_name in line:
                        available_lines.append(line)

            if available_lines:
                message = f"✅ Yes, Dr. {doctor} is available on {weekday_name}. Here are the time slots:\n\n" + "\n".join(available_lines)
            else:
                message = f"❌ Dr. {doctor} is not available on {weekday_name}. But here are some nearby available slots:\n\n" + slots_text
        else:
            message = f"👍 Let me check available time slots for Dr. {doctor}.\n\n" + slots_text

        state["tool_results"] = []  # Clear to prevent re-display in final serializer

    elif tool_name == "book_appointment_tool":
        message = f"📅 Booking your {service} with Dr. {doctor}. One moment..."

    elif tool_name == "get_appointments":
        message = f"🔍 Fetching current appointments for Dr. {doctor}."

    else:
        message = f"🧠 I'm working on your request using {tool_name}. Hang tight!"

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
                f"🤔 I interpreted that as '{weekday_name}'. Let me know if you meant a different day.\n\n"
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
            results.append(f"🔒 Access denied to `{tool_name}`. Doctor access required.")
            continue

        try:
            result = call_mcp_tool(tool_name, arguments)
            print(f"🔍 Tool '{tool_name}' result:", result)
            results.append(result)

            if tool_name == "book_appointment_tool":
                booking_confirmation = result
            elif tool_name == "get_appointments":
                appointments_output = result

        except Exception as e:
            results.append(f"❌ Error calling MCP tool '{tool_name}': {e}")


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
        print("📢 Final output generated:", final_output)
    return {
        **state,
        "final_answer": final_output
    }
