# ============================================
# File: tool_server.py
# Purpose: FastAPI server to expose tools via HTTP (MCP-style)
# ============================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Import raw tool functions
from agent.tools.appointment import (
    book_appointment_tool,
    get_appointments,
    suggest_appointment_slots,
    get_earliest_available_slot,
    get_next_client_info,
    summarize_calendar_today,
)

# FastAPI app
app = FastAPI()

# -----------------------------
# MCP Tool Registry (OpenAI spec)
# -----------------------------
MCP_TOOL_REGISTRY = [
    {
        "type": "function",
        "function": {
            "name": "book_appointment_tool",
            "description": "Book a new appointment given doctor, patient, and time details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"},
                    "patient_name": {"type": "string"},
                    "branch_id": {"type": "integer"},
                    "service_name": {"type": "string"},
                    "start_time": {"type": "string", "format": "date-time"},
                    "end_time": {"type": "string", "format": "date-time"}
                },
                "required": ["doctor_name", "patient_name", "start_time", "end_time"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_appointments",
            "description": "Get all appointments for a doctor, optionally filtered by status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"},
                    "status": {"type": "string"}
                },
                "required": ["doctor_name"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_appointment_slots",
            "description": "Suggest available appointment slots for a doctor after a given time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"},
                    "after": {"type": "string", "format": "date-time"},
                    "weekday": {"type": "integer"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["doctor_name"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_earliest_available_slot",
            "description": "Return the earliest available time for a given doctor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"}
                },
                "required": ["doctor_name"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_next_client_info",
            "description": "Return the next client and their info for a given doctor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"}
                },
                "required": ["doctor_name"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_calendar_today",
            "description": "Summarize today's appointments for a doctor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"}
                },
                "required": ["doctor_name"]
            },
        },
    },
]

# MCP function lookup map for the agent to call actual Python implementations
MCP_FUNCTION_LOOKUP = {
    "book_appointment_tool": book_appointment_tool,
    "get_appointments": get_appointments,
    "suggest_appointment_slots": suggest_appointment_slots,
    "get_earliest_available_slot": get_earliest_available_slot,
    "get_next_client_info": get_next_client_info,
    "summarize_calendar_today": summarize_calendar_today,
}

# -----------------------------
# Shared request schema
# -----------------------------
class ToolRequest(BaseModel):
    doctor_name: Optional[str] = None
    patient_name: Optional[str] = None
    branch_id: Optional[int] = None
    service_name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    after: Optional[str] = None
    limit: Optional[int] = 5
    weekday: Optional[int] = None
    status: Optional[str] = None


# -----------------------------
# Tool Endpoints
# -----------------------------
@app.post("/tools/book_appointment_tool")
def call_book_appointment(req: ToolRequest):
    return book_appointment_tool(req.model_dump(exclude_none=True))


@app.post("/tools/get_appointments")
def call_get_appointments(req: ToolRequest):
    return get_appointments(req.model_dump(exclude_none=True))


@app.post("/tools/suggest_appointment_slots")
def call_suggest_slots(req: ToolRequest):
    return suggest_appointment_slots(**req.model_dump(exclude_none=True))


@app.post("/tools/get_earliest_available_slot")
def call_earliest_slot(req: ToolRequest):
    return get_earliest_available_slot(req.model_dump(exclude_none=True))


@app.post("/tools/get_next_client_info")
def call_next_client(req: ToolRequest):
    return get_next_client_info(req.model_dump(exclude_none=True))


@app.post("/tools/summarize_calendar_today")
def call_calendar_summary(req: ToolRequest):
    return summarize_calendar_today(req.model_dump(exclude_none=True))
