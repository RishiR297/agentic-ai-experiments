# ============================================
# File: tool_server.py
# Purpose: FastAPI server to expose appointment tools via HTTP (MCP-style)
#
# This server provides HTTP endpoints for all appointment-related tools,
# following the Model Context Protocol (MCP) specification. It serves as
# a microservice that the main agent can call to perform specific actions
# like booking appointments, checking availability, and managing schedules.
#
# Key Features:
# - MCP-compliant tool registry with OpenAI function calling schema
# - Proper argument validation and filtering for each tool
# - Error handling and informative responses
# - Support for all core appointment operations
#
# Endpoints:
# - /tools/book_appointment_tool: Book new appointments
# - /tools/get_appointments: Retrieve existing appointments
# - /tools/suggest_appointment_slots: Find available time slots
# - /tools/get_earliest_available_slot: Get earliest availability
# - /tools/get_next_client_info: Next patient information
# - /tools/summarize_calendar_today: Daily schedule summary
# ============================================

from fastapi import FastAPI, HTTPException
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
            "description": "Get appointments for a doctor, optionally filtered by status, time range, and limited by count. Defaults to upcoming appointments within the next week if no time filters specified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"},
                    "status": {"type": "string"},
                    "after": {"type": "string", "format": "date-time"},
                    "before": {"type": "string", "format": "date-time"},
                    "limit": {"type": "integer", "default": 10}
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
    before: Optional[str] = None
    limit: Optional[int] = 5
    weekday: Optional[int] = None
    status: Optional[str] = None


# -----------------------------
# Tool Endpoints
# -----------------------------
@app.post("/tools/book_appointment_tool")
def call_book_appointment(req: ToolRequest):
    # Validate required fields explicitly for this tool
    print("📥 Incoming booking payload:", req.dict())
    required_fields = ["doctor_name", "patient_name", "branch_id", "service_name", "start_time", "end_time"]
    missing = [f for f in required_fields if getattr(req, f) is None]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing)}")
    
    # Only pass the fields that book_appointment_tool expects
    booking_args = {
        "doctor_name": req.doctor_name,
        "patient_name": req.patient_name,
        "branch_id": req.branch_id,
        "service_name": req.service_name,
        "start_time": req.start_time,
        "end_time": req.end_time
    }
    print("📥 Filtered booking payload:", booking_args)
    return book_appointment_tool(**booking_args)


@app.post("/tools/get_appointments")
def call_get_appointments(req: ToolRequest):
    # doctor_name is required
    if not req.doctor_name:
        raise HTTPException(status_code=400, detail="Missing required field: doctor_name")
    
    # Only pass the fields that get_appointments expects
    args = {}
    if req.doctor_name:
        args["doctor_name"] = req.doctor_name
    if req.status:
        args["status"] = req.status
    if req.after:
        args["after"] = req.after
    if req.before:
        args["before"] = req.before
    if req.limit:
        args["limit"] = req.limit
    
    return get_appointments(**args)


@app.post("/tools/suggest_appointment_slots")
def call_suggest_slots(req: ToolRequest):
    if not req.doctor_name:
        raise HTTPException(status_code=400, detail="Missing required field: doctor_name")
    args = req.model_dump(exclude_none=True)
    return suggest_appointment_slots(**args)


@app.post("/tools/get_earliest_available_slot")
def call_earliest_slot(req: ToolRequest):
    if not req.doctor_name:
        raise HTTPException(status_code=400, detail="Missing required field: doctor_name")
    return get_earliest_available_slot(req.doctor_name)


@app.post("/tools/get_next_client_info")
def call_next_client(req: ToolRequest):
    if not req.doctor_name:
        raise HTTPException(status_code=400, detail="Missing required field: doctor_name")
    return get_next_client_info(req.doctor_name)


@app.post("/tools/summarize_calendar_today")
def call_calendar_summary(req: ToolRequest):
    if not req.doctor_name:
        raise HTTPException(status_code=400, detail="Missing required field: doctor_name")
    return summarize_calendar_today(req.doctor_name)

