# # ==============================================
# # File: nodes.py
# # Purpose: Define LangGraph nodes that process the agent’s internal state
# # ==============================================

import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json

from agent.tools.appointment import MCP_TOOL_REGISTRY, MCP_FUNCTION_LOOKUP
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Azure OpenAI LLM setup
llm_with_tools = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview",
    model_kwargs={"tools": MCP_TOOL_REGISTRY}

)

import json

print("🧪 Tools registered:")
for i, tool in enumerate(MCP_TOOL_REGISTRY):
    print(f"Tool #{i + 1}: type={type(tool)}")
    print(json.dumps(tool, indent=2) if isinstance(tool, dict) else tool)

for tool in MCP_TOOL_REGISTRY:
    print(f"- {tool.get('name', '<no name>')}")


# MCP-based planner node
MCP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that decides the next action."),
    ("human", "{user_input}")
])

def planner_node(state: dict) -> dict:
    user_input = state.get("user_input")
    if not user_input:
        raise ValueError("Missing 'user_input' in agent state")

    response = llm_with_tools.invoke([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=user_input)
    ])

    clean_steps = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            # 🔥 Convert args safely
            args_dict = (
                tool_call.args.model_dump()
                if hasattr(tool_call.args, "model_dump")
                else dict(tool_call.args)
            )
            clean_steps.append({
                "tool_name": tool_call.name,
                "args": args_dict
            })
        return {
            **state,
            "intermediate_steps": clean_steps,
            "final_answer": response.content
        }
    else:
        return {
            **state,
            "intermediate_steps": [],
            "final_answer": response.content
        }

# Router node
def route_node(state: dict) -> dict:
    steps = state.get("intermediate_steps", [])
    state["next"] = "tool" if steps else "answer"
    return state

# ✅ FIX: Use MCP_TOOL_REGISTRY as is
TOOL_LOOKUP = MCP_FUNCTION_LOOKUP

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
        arguments = call["args"]

        tool_fn = TOOL_LOOKUP.get(tool_name)
        if not tool_fn:
            results.append(f"❌ Unknown tool: {tool_name}")
        else:
            try:
                result = tool_fn(**arguments)
                print(f"🔍 Tool '{tool_name}' result:", result)
                results.append(result)

                # ✅ Save specific outputs
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
    if "tool_results" in state:
        safe_results = []
        for item in state["tool_results"]:
            if isinstance(item, BaseModel):
                safe_results.append(item.model_dump())
            elif not isinstance(item, (str, int, float, dict, list)):
                safe_results.append(str(item))
            else:
                safe_results.append(item)

        final_output = "\n".join([
            json.dumps(r, indent=2) if isinstance(r, (dict, list)) else str(r)
            for r in safe_results
        ])
    else:
        final_output = state.get("final_answer", "🤖 I couldn't process that.")

    return {
        **state,
        "final_answer": final_output
    }
