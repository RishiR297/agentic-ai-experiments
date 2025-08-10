from typing import Any, Dict, Optional

class AgentState(Dict[str, Any]):
    """
    A flexible state object for the agent, allowing attribute and dict-style access.
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value

class AgentConfig:
    """
    Configuration object for the agent, including LLM, prompts, and other settings.
    """
    def __init__(self, llm, prompts: Optional[Dict[str, str]] = None, **kwargs):
        self.llm = llm
        self.prompts = prompts or {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_system_prompt(self, key: str) -> str:
        return self.prompts.get(key, "")

    def get_role_permissions(self, role: str):
        # Implement role-based permissions as needed
        return ["appointment_booking", "schedule_query", "patient_lookup", "history_query", "appointment_lookup"]