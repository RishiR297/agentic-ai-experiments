# ==============================================
# File: agent/tools/appointment.py
# Purpose: View and book appointments via database, with LangGraph integration
# ==============================================

# -----------------------------
# Imports
# -----------------------------
from typing import Optional, List
from datetime import datetime
from utils.db import get_db_connection
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool


# -----------------------------
# Helper: format_time
# -----------------------------
def format_time(ts: str) -> str:
    """
    Converts an ISO timestamp string to 'YYYY-MM-DD HH:MM' format.
    Falls back to original string if parsing fails.
    """
    try:
        dt = datetime.fromisoformat(ts.split('.')[0])  # Strip microseconds
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return ts


# -----------------------------
# Core Function: list_appointments
# -----------------------------
def list_appointments(
    doctor_name: Optional[str] = None,
    patient_name: Optional[str] = None,
    status: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None
) -> List[str]:
    """
    Queries appointments from the View_Appointments view, optionally filtered by doctor, patient, status, and time.
    Returns a list of human-readable appointment descriptions.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    SELECT 
        AppointmentId,
        PatientName,
        DoctorName,
        BranchName,
        ServiceName,
        StartDateTime,
        EndDateTime,
        Status
    FROM View_Appointments
    WHERE 1=1
    """
    params = []

    if doctor_name:
        query += " AND DoctorName LIKE ?"
        params.append(f"%{doctor_name}%")
    if patient_name:
        query += " AND PatientName LIKE ?"
        params.append(f"%{patient_name}%")
    if status:
        query += " AND Status = ?"
        params.append(status)
    if after:
        query += " AND StartDateTime >= ?"
        params.append(after)
    if before:
        query += " AND StartDateTime <= ?"
        params.append(before)

    query += " ORDER BY StartDateTime ASC"
    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    formatted = []
    for row in results:
        appt_id, patient, doctor, branch, service, start_time, end_time, status = row
        formatted.append(
            f"[{status}] {format_time(start_time)} - {format_time(end_time)} | {service} with Dr. {doctor} at {branch} for {patient}"
        )

    if not formatted:
        return ["No appointments found for the specified filters."]
    return formatted


# -----------------------------
# LangGraph Tool: get_appointments
# -----------------------------
@tool
def get_appointments(
    doctor_name: Optional[str] = None,
    status: Optional[str] = None,
    after: Optional[str] = None
) -> str:
    """
    LangChain-compatible tool that fetches and returns formatted appointments as a string.
    Filters: doctor_name, status, start_time (after).
    """
    appointments = list_appointments(
        doctor_name=doctor_name,
        status=status,
        after=after
    )
    return "\n".join(appointments)


# -----------------------------
# Logic: is_doctor_available
# -----------------------------
def is_doctor_available(doctor_id: int, desired_start: str, desired_end: str) -> bool:
    """
    Checks if the doctor is available between desired_start and desired_end.
    Returns True if:
    - Doctor has working hours in COR_DoctorSchedule covering that time
    - No overlapping appointments exist in View_Appointments
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if doctor is scheduled to work at that time
    cursor.execute("""
        SELECT 1 FROM COR_DoctorSchedule
        WHERE DoctorUserId = ?
          AND ? >= StartTime AND ? <= EndTime
    """, (doctor_id, desired_start, desired_end))
    schedule_exists = cursor.fetchone() is not None

    if not schedule_exists:
        conn.close()
        return False

    # Check for conflicting appointments
    cursor.execute("""
        SELECT 1 FROM View_Appointments
        WHERE DoctorUserId = ?
          AND (? < EndDateTime AND ? > StartDateTime)
    """, (doctor_id, desired_start, desired_end))
    conflict_exists = cursor.fetchone() is not None

    conn.close()
    return not conflict_exists


# -----------------------------
# Logic: create_appointment
# -----------------------------
def create_appointment(
    doctor_id: int,
    doctor_name: str,
    patient_name: str,
    branch_id: int,
    service_name: str,
    start_time: str,
    end_time: str
) -> str:
    """
    Inserts a new appointment into the View_Appointments_Setup table.
    Assumes availability has already been verified.
    Returns a confirmation message or an error string.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO View_Appointments_Setup (
                DoctorUserId,
                DoctorName,
                PatientName,
                BranchId,
                ServiceName,
                StartDateTime,
                EndDateTime,
                Status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doctor_id,
            doctor_name,
            patient_name,
            branch_id,
            service_name,
            start_time,
            end_time,
            'Pending'
        ))
        conn.commit()
        return f"✅ Appointment booked successfully with Dr. {doctor_name} at Branch {branch_id} from {start_time} to {end_time} for {patient_name}."
    except Exception as e:
        return f"❌ Failed to book appointment: {e}"
    finally:
        conn.close()


# -----------------------------
# LangGraph Tool: book_appointment_tool
# -----------------------------
@tool
def book_appointment_tool(
    doctor_name: str,
    patient_name: str,
    branch_id: int,
    service_name: str,
    start_time: str,
    end_time: str
) -> str:
    """
    LangChain-compatible tool that books an appointment for a patient with a doctor.
    Checks doctor availability first.
    Returns confirmation or error message.
    """
    # Step 1: Find doctor by name
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT UserId, Firstname, Lastname FROM COR_Doctor
        WHERE DoctorName LIKE ?
    """, (f"%{doctor_name}%",))
    match = cursor.fetchone()
    conn.close()

    if not match:
        return f"❌ No doctor found matching '{doctor_name}'"

    doctor_id = match[0]
    full_name = f"{match[1]} {match[2]}"

    # Step 2: Check availability
    if not is_doctor_available(doctor_id, start_time, end_time):
        return f"❌ Dr. {full_name} is not available from {start_time} to {end_time}"

    # Step 3: Book the appointment
    return create_appointment(
        doctor_id=doctor_id,
        doctor_name=full_name,
        patient_name=patient_name,
        branch_id=branch_id,
        service_name=service_name,
        start_time=start_time,
        end_time=end_time
    )

# -----------------------------
# MCP-Compatible Tool Registry
# -----------------------------
# Registry for LLM (OpenAI schema dicts)
MCP_TOOL_REGISTRY = [
    convert_to_openai_tool(book_appointment_tool),
    convert_to_openai_tool(get_appointments)
]

# Registry for function execution
MCP_FUNCTION_LOOKUP = {
    "book_appointment_tool": book_appointment_tool,
    "get_appointments": get_appointments
}