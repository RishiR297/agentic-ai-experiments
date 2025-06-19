from typing import TypedDict, Optional


class AgentState(TypedDict, total=False):
    # --- Core fields ---
    user_input: str                   # Raw user input (e.g., "I want to book with Dr. Smith")
    
    # --- Tool inputs ---
    doctor_name: Optional[str]
    patient_name: Optional[str]
    branch_id: Optional[int]
    service_name: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    status: Optional[str]
    after: Optional[str]
    tool_name: Optional[str]          # Name of the tool to call (e.g., "book_appointment", "get_appointments")
    # --- Tool outputs ---
    appointments_output: Optional[str]       # Result from get_appointments
    booking_confirmation: Optional[str]      # Result from book_appointment_tool

    # --- Final agent output ---
    final_answer: Optional[str]              # Natural language response from LLM
