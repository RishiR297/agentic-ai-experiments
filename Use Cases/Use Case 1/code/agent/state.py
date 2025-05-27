from typing import TypedDict, Optional

class AgentState(TypedDict):
    user_input: str
    appointments_output: Optional[str]
    final_answer: Optional[str]
