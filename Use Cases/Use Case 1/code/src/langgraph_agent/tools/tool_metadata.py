"""
Centralized tool metadata for agent and MCP integration.
This module provides tool name constants and required user-facing fields for each tool.
If the MCP server exposes this via API, replace this with a dynamic fetch.
"""

TOOL_NAMES = [
    "appointment_booking",
    "schedule_query",
    "appointment_lookup",
    "patient_lookup",
    # Add more tool names as needed
]

# User-facing fields required for each tool (do not include backend/internal fields)
TOOL_USER_FIELDS = {
    "appointment_booking": ["service_name", "patient_name", "start_time", "appointment_date"],
    "schedule_query": ["doctor_name", "appointment_date"],
    "appointment_lookup": ["patient_name", "appointment_date"],
    "patient_lookup": ["patient_name"],
    # Add more tool-specific fields as needed
}

def get_tool_user_fields(tool_name: str):
    """Get required user-facing fields for a tool."""
    return TOOL_USER_FIELDS.get(tool_name, [])

def get_all_tool_names():
    return TOOL_NAMES
