from agent.state import AgentState
from agent.tools.appointment import get_appointments
from typing import Optional
import re
import os

from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()


# Initialize LLM once
llm = AzureChatOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    deployment_name=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
    api_version="2024-02-15-preview"
)


def extract_doctor_name(user_input: str) -> Optional[str]:
    # Match "Dr. Antonella" or similar
    match = re.search(r"Dr\.?\s+([A-Za-z']+)", user_input)
    if match:
        return match.group(1)
    return None


# Node 1: input handler (already has user_input in state)
def get_user_input(state: AgentState) -> AgentState:
    return state


# Node 2: query appointments from tool
def query_appointments(state: AgentState) -> AgentState:
    user_input = state.get("user_input", "")
    doctor_name = extract_doctor_name(user_input)
    output = get_appointments.invoke({"doctor_name": doctor_name})
    return {
        **state,
        "appointments_output": output
    }


# Node 3: summarize the results naturally using Azure OpenAI
def generate_final_answer(state: AgentState) -> AgentState:
    appointments_output = state.get("appointments_output", "")
    user_input = state.get("user_input", "")

    messages = [
        SystemMessage(content="You are a friendly appointment assistant."),
        HumanMessage(content=f"The user asked: {user_input}\n"
                             f"Here are the appointment results: {appointments_output}\n"
                             f"Please generate a helpful and natural response.")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "final_answer": response.content
    }
