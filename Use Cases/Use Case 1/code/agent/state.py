from typing import TypedDict, Optional
from typing import List, Literal, Union
from langchain_core.messages import BaseMessage

class AgentState(TypedDict, total=False):
    # --- Core fields ---
    user_input: str

    # --- Tool inputs ---
    doctor_name: Optional[str]
    patient_name: Optional[str]
    branch_id: Optional[int]
    service_name: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    status: Optional[str]
    after: Optional[str]
    tool_name: Optional[str]
    requested_weekday: Optional[int]

    # --- Tool outputs ---
    appointments_output: Optional[str]
    booking_confirmation: Optional[str]

    # --- Final answer ---
    final_answer: Optional[str]

    # ✅ New for memory
    identity: Optional[str]  # like a user ID or session ID
    chat_history: Optional[List[BaseMessage]]  # short-term memory for planner context

    REQUIRED_FIELDS = ["doctor_name", "patient_name", "branch_id", "service_name", "start_time", "end_time"]
