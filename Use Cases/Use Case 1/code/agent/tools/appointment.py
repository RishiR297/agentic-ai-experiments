# code/agent/tools/appointment.py

from typing import Optional, List
from datetime import datetime
from utils.db import get_db_connection
from langchain_core.tools import tool

def format_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.split('.')[0])  # Remove microseconds
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return ts

def list_appointments(
    doctor_name: Optional[str] = None,
    patient_name: Optional[str] = None,
    status: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None
) -> List[str]:
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

# LangGraph-exposed tool
@tool
def get_appointments(
    doctor_name: Optional[str] = None,
    status: Optional[str] = None,
    after: Optional[str] = None
) -> str:
    """
    Fetches appointments from the database filtered by doctor name, status, or start date (after).
    """
    appointments = list_appointments(
        doctor_name=doctor_name,
        status=status,
        after=after
    )
    return "\n".join(appointments)
