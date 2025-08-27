"""
Core module for LangGraph Medical Assistant

This module contains the essential core components for the medical assistant agent:

Essential Files:
- config.py: Agent configuration, LLM setup, and system prompts
- graph.py: Main LangGraph workflow definition and agent class
- state.py: AgentState TypedDict definition and state management functions

The core module is organized to contain only the essential files needed 
for the medical assistant agent to function properly.
"""

# Core exports
from .state import AgentState, create_initial_state
from .config import AgentConfig
from .graph import MedicalAssistantAgent, create_medical_agent_graph

__all__ = [
    'AgentState',
    'create_initial_state', 
    'AgentConfig',
    'MedicalAssistantAgent',
    'create_medical_agent_graph'
]
