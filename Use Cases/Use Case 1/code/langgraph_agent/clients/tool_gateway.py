"""
Tool gateway abstraction for LangGraph node execution.

This is the first migration seam toward the intended architecture:
LangGraph as the MCP client, and tools hosted behind a server boundary.

For now, the local implementation delegates to the existing in-process tools.
Later, the same interface can be backed by an MCP client.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Set

import httpx


class ToolGateway:
    """Abstract tool gateway interface."""

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class LocalToolGateway(ToolGateway):
    """
    Local tool gateway used as a controlled migration step.

    It preserves current behavior while giving the agent a stable interface
    that can later be swapped for a true MCP-backed implementation.
    """

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "schedule_query":
            from ..tools.database import schedule_query

            doctor_id = args["doctor_id"]
            date = args.get("date")
            include_availability = args.get("include_availability", True)
            return schedule_query.func(doctor_id, date, include_availability)

        if tool_name == "appointment_query_executor":
            from ..tools.database import appointment_query_executor

            doctor_id = int(args["doctor_id"])
            query_type = args["query_type"]
            date = args.get("date")
            patient_name = args.get("patient_name")
            return appointment_query_executor.func(doctor_id, query_type, date, patient_name)

        if tool_name == "doctor_availability":
            from ..tools.database import schedule_query

            doctor_id = int(args["doctor_id"])
            date = args.get("date")
            result = schedule_query.func(doctor_id, date, True)
            if result.get("success"):
                result["query_type"] = "doctor_availability"
            return result

        if tool_name == "calendar_summary":
            from ..tools.database import appointment_query_executor

            doctor_id = int(args["doctor_id"])
            date = args.get("date")
            result = appointment_query_executor.func(doctor_id, "daily_schedule", date, None)
            if result.get("success"):
                result["query_type"] = "calendar_summary"
            return result

        if tool_name == "resolve_booking_context":
            from ..tools.database import (
                execute_query,
                get_doctor_default_branch,
                get_or_create_patient_id,
                get_service_id_and_duration,
            )

            patient_name = args.get("patient_name")
            service_name = args.get("service_name")
            doctor_id = args.get("doctor_id")

            patient_id = None
            if patient_name:
                patient_id = get_or_create_patient_id(patient_name)

            service_info = None
            if service_name:
                service_info = get_service_id_and_duration(service_name)

            branch_info = None
            doctor_name = "Unknown Doctor"
            if doctor_id:
                branch_info = get_doctor_default_branch(doctor_id)
                doctor_results = execute_query(
                    "SELECT DoctorName FROM View_Appointments WHERE DoctorId = ? AND DoctorName IS NOT NULL AND DoctorName != '' LIMIT 1",
                    (doctor_id,),
                )
                if doctor_results and doctor_results[0].get("DoctorName"):
                    doctor_name = doctor_results[0]["DoctorName"]

            return {
                "success": True,
                "patient_id": patient_id,
                "service_info": service_info,
                "branch_info": branch_info,
                "doctor_name": doctor_name,
            }

        if tool_name == "resolve_patient_identity":
            from ..tools.database import get_or_create_patient_id

            patient_name = args.get("patient_name")
            patient_id = get_or_create_patient_id(patient_name) if patient_name else None
            return {
                "success": bool(patient_id),
                "patient_id": patient_id,
                "patient_name": patient_name,
                "error": None if patient_id else "Patient could not be resolved",
            }

        if tool_name == "resolve_doctor_identity":
            from ..tools.database import execute_query

            doctor_id = args.get("doctor_id")
            doctor_name = args.get("doctor_name")

            if doctor_id is not None and str(doctor_id).strip().isdigit():
                rows = execute_query(
                    """
                    SELECT DoctorId, DoctorName
                    FROM View_Appointments
                    WHERE DoctorId = ?
                    AND DoctorName IS NOT NULL AND DoctorName != ''
                    LIMIT 1
                    """,
                    [int(str(doctor_id).strip())],
                )
                if rows:
                    return {
                        "success": True,
                        "doctor_id": str(rows[0]["DoctorId"]),
                        "doctor_name": rows[0]["DoctorName"],
                        "error": None,
                    }

            lookup_name = (doctor_name or doctor_id or "").strip()
            normalized_name = re.sub(r"\bdr\.?\s*", "", lookup_name, flags=re.IGNORECASE).strip()
            normalized_name = re.sub(r"\s+", " ", normalized_name)

            if not normalized_name:
                return {
                    "success": False,
                    "doctor_id": None,
                    "doctor_name": None,
                    "error": "Doctor could not be resolved",
                }

            rows = execute_query(
                """
                SELECT DoctorId, DoctorName
                FROM View_Appointments
                WHERE LOWER(TRIM(DoctorName)) = LOWER(TRIM(?))
                LIMIT 1
                """,
                [normalized_name],
            )

            if not rows:
                rows = execute_query(
                    """
                    SELECT DoctorId, DoctorName
                    FROM View_Appointments
                    WHERE LOWER(TRIM(DoctorName)) LIKE LOWER(TRIM(?))
                    ORDER BY DoctorName
                    LIMIT 1
                    """,
                    [f"%{normalized_name}%"],
                )

            if rows:
                return {
                    "success": True,
                    "doctor_id": str(rows[0]["DoctorId"]),
                    "doctor_name": rows[0]["DoctorName"],
                    "error": None,
                }

            return {
                "success": False,
                "doctor_id": None,
                "doctor_name": None,
                "error": f"Doctor '{lookup_name}' could not be resolved",
            }

        if tool_name == "find_appointment_for_rescheduling":
            from ..tools.database import find_appointment_for_rescheduling

            return find_appointment_for_rescheduling(
                patient_name=args.get("patient_name"),
                doctor_id=args.get("doctor_id"),
                service_name=args.get("service_name"),
                current_date=args.get("current_date"),
            )

        if tool_name == "find_appointment_for_cancellation":
            from ..tools.database import find_appointment_for_cancellation

            return find_appointment_for_cancellation(
                patient_name=args.get("patient_name"),
                doctor_id=args.get("doctor_id"),
                service_name=args.get("service_name"),
                current_date=args.get("current_date"),
            )

        raise ValueError(f"Unsupported tool in LocalToolGateway: {tool_name}")


class MCPToolGateway(ToolGateway):
    """
    MCP-backed gateway skeleton.

    This keeps the same interface as LocalToolGateway so individual tools can be
    moved over one by one once the MCP client wiring is ready.
    """

    SUPPORTED_TOOLS = {
        "schedule_query",
        "appointment_query_executor",
        "doctor_availability",
        "calendar_summary",
        "resolve_booking_context",
        "resolve_patient_identity",
        "resolve_doctor_identity",
        "find_appointment_for_rescheduling",
        "find_appointment_for_cancellation",
    }

    def __init__(self, server_url: Optional[str] = None) -> None:
        self.server_url = server_url

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.SUPPORTED_TOOLS:
            raise ValueError(f"Unsupported tool in MCPToolGateway: {tool_name}")
        return self._execute_via_mcp(tool_name, args)

    def _execute_via_mcp(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "doctor_availability":
            if not self.server_url:
                raise ValueError("MCPToolGateway requires server_url for doctor_availability")

            response = httpx.post(
                f"{self.server_url.rstrip('/')}/tools/doctor_availability",
                json={
                    "doctor_id": args["doctor_id"],
                    "date": args.get("date"),
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()

        if tool_name == "calendar_summary":
            if not self.server_url:
                raise ValueError("MCPToolGateway requires server_url for calendar_summary")

            response = httpx.post(
                f"{self.server_url.rstrip('/')}/tools/calendar_summary",
                json={
                    "doctor_id": args["doctor_id"],
                    "date": args.get("date"),
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()

        if tool_name == "schedule_query":
            if not self.server_url:
                raise ValueError("MCPToolGateway requires server_url for schedule_query")

            response = httpx.post(
                f"{self.server_url.rstrip('/')}/tools/schedule_query",
                json={
                    "doctor_id": args["doctor_id"],
                    "date": args.get("date"),
                    "include_availability": args.get("include_availability", True),
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()

        if tool_name == "appointment_query_executor":
            if not self.server_url:
                raise ValueError("MCPToolGateway requires server_url for appointment_query_executor")

            response = httpx.post(
                f"{self.server_url.rstrip('/')}/tools/appointment_query_executor",
                json={
                    "doctor_id": args["doctor_id"],
                    "query_type": args["query_type"],
                    "date": args.get("date"),
                    "patient_name": args.get("patient_name"),
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()

        raise NotImplementedError(
            f"MCP execution is not wired yet for tool '{tool_name}'. "
            "Use LocalToolGateway until the MCP client integration is implemented."
        )


class HybridToolGateway(ToolGateway):
    """
    Gateway that prefers MCP for selected tools and falls back to local execution.
    """

    def __init__(
        self,
        local_gateway: Optional[ToolGateway] = None,
        mcp_gateway: Optional[ToolGateway] = None,
        mcp_tools: Optional[Set[str]] = None,
    ) -> None:
        self.local_gateway = local_gateway or LocalToolGateway()
        self.mcp_gateway = mcp_gateway or MCPToolGateway()
        self.mcp_tools = set(mcp_tools or set())

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name in self.mcp_tools:
            try:
                return self.mcp_gateway.execute(tool_name, args)
            except Exception:
                return self.local_gateway.execute(tool_name, args)
        return self.local_gateway.execute(tool_name, args)


_tool_gateway: ToolGateway = LocalToolGateway()


def get_tool_gateway() -> ToolGateway:
    """Return the currently configured tool gateway."""
    return _tool_gateway


def set_tool_gateway(tool_gateway: ToolGateway) -> None:
    """Replace the active tool gateway implementation."""
    global _tool_gateway
    _tool_gateway = tool_gateway


def configure_tool_gateway(mode: str = "local", server_url: Optional[str] = None) -> ToolGateway:
    """Configure and return the active tool gateway."""
    normalized_mode = (mode or "local").strip().lower()

    if normalized_mode == "local":
        gateway: ToolGateway = LocalToolGateway()
    elif normalized_mode == "mcp":
        gateway = MCPToolGateway(server_url=server_url)
    elif normalized_mode == "hybrid":
        gateway = HybridToolGateway(
            local_gateway=LocalToolGateway(),
            mcp_gateway=MCPToolGateway(server_url=server_url),
            mcp_tools={"doctor_availability", "calendar_summary", "schedule_query", "appointment_query_executor"},
        )
    else:
        raise ValueError(f"Unknown tool gateway mode: {mode}")

    set_tool_gateway(gateway)
    return gateway
