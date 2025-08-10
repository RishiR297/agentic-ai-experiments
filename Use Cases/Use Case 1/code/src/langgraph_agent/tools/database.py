"""
Database Tools for LangGraph Medical Assistant

This module provides database connectivity and common query functions
extracted from the original llm_tool_server.py.
"""

import sqlite3
import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Database path - adjusted for running from src directory
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'db', 'output.db')
DB_PATH = os.path.abspath(DB_PATH)
logger.info(f"Resolved database path: {DB_PATH}")

def lookup_patient_id(patient_name: str) -> int | None:
    logger.debug(f"Looking up patient ID for name: {patient_name}")
    query = "SELECT PatientId FROM View_Appointments WHERE LOWER(PatientName) = LOWER(?) LIMIT 1"
    results = execute_query(query, (patient_name,))
    if results:
        logger.debug(f"Found PatientId for {patient_name}: {results[0]['PatientId']}")
        return results[0]["PatientId"]
    logger.debug(f"No PatientId found for {patient_name}")
    return None

def get_service_id_and_duration(service_name: str) -> Optional[Dict[str, Any]]:
    """
    Given a service name, resolve the most likely ServiceId and Duration from View_Appointments
    by matching ServiceName (case-insensitive, DB values are all caps).
    Returns None if not found.
    """
    logger.debug(f"Fetching service_id and average duration for: {service_name}")
    service_name_upper = service_name.strip().upper()
    query = """
        SELECT 
          ServiceId, 
          ServiceName, 
          AVG((strftime('%s', EndDateTime) - strftime('%s', StartDateTime)) / 60.0) as avg_duration_minutes,
          COUNT(*) as freq
        FROM View_Appointments
        WHERE UPPER(ServiceName) = ?
        GROUP BY ServiceId, ServiceName
        ORDER BY freq DESC
        LIMIT 1
    """
    results = execute_query(query, (service_name_upper,))
    if results:
        logger.debug(f"Found service info for {service_name}: {results[0]}")
        return {
            "service_id": results[0]["ServiceId"],
            "duration": results[0].get("avg_duration_minutes"),
            "service_name": results[0]["ServiceName"]
        }
    logger.debug(f"No service info found for {service_name}")
    return None
def get_doctor_default_branch(doctor_id: str | int) -> dict | tuple | None:
    logger.debug(f"Fetching default branch for doctor_id={doctor_id}")
    query = "SELECT BranchName, BranchId FROM View_Appointments WHERE DoctorId = ? LIMIT 1"
    results = execute_query(query, (doctor_id,))
    if results:
        logger.debug(f"Found branch info for doctor_id={doctor_id}: {results[0]}")
        return (results[0]["BranchName"], results[0]["BranchId"])
    logger.warning(f"No default branch found for doctor_id={doctor_id}")
    return None

def get_or_create_patient_id(patient_name: str) -> int:
    """
    Given a patient name, return the PatientId if exists, otherwise create a new patient and return the new id.
    """
    # Try to find existing patient
    query = "SELECT PatientId FROM View_Appointments WHERE LOWER(PatientName) = LOWER(?) LIMIT 1"
    results = execute_query(query, (patient_name,))
    if results:
        return results[0]["PatientId"]
    # Insert new patient
    insert_query = "INSERT INTO View_Appointments (PatientName) VALUES (?) RETURNING PatientId"
    insert_results = execute_query(insert_query, (patient_name,))
    if insert_results:
        return insert_results[0]["PatientId"]
    raise Exception(f"Failed to get or create PatientId for {patient_name}")


def get_db_connection():
    """Get database connection"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")


def execute_query(query: str, params: tuple = ()) -> List[Dict]:
    """Execute database query and return results. For INSERT, return RETURNING results if present, else lastrowid/rowcount."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(query, params)
        # Detect query type
        query_type = query.strip().split()[0].upper()
        if query_type == "SELECT":
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            logger.info(f"SELECT executed. {len(results)} rows returned.")
            return results
        elif query_type == "INSERT":
            # Check if RETURNING is present
            if "RETURNING" in query.upper():
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                logger.info(f"INSERT ... RETURNING executed. {len(results)} rows returned.")
                logger.info(f"Rowcount: {cursor.rowcount}, Lastrowid: {cursor.lastrowid}")
                return results
            else:
                conn.commit()
                logger.info(f"INSERT executed. Rowcount: {cursor.rowcount}, Lastrowid: {cursor.lastrowid}")
                return [{"rowcount": cursor.rowcount, "lastrowid": cursor.lastrowid}]
        else:
            # For UPDATE, DELETE, etc.
            conn.commit()
            logger.info(f"{query_type} executed. Rowcount: {cursor.rowcount}")
            return [{"rowcount": cursor.rowcount}]
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        return []
    finally:
        conn.close()


def resolve_doctor_uuid_to_id(doctor_uuid: str) -> Optional[int]:
    """
    Resolve doctor UUID to integer DoctorId used in View_Appointments.
    If multiple DoctorIds exist for the same doctor name, prioritize the one with recent appointments.
    """
    try:
        # Get doctor info from COR_Doctor table using UserId (which is the UUID)
        query = "SELECT DisplayName FROM COR_Doctor WHERE UserId = ?"
        conn = get_db_connection()
        cursor = conn.execute(query, (doctor_uuid,))
        row = cursor.fetchone()
        
        if row:
            doctor_name = row[0]
            logger.info(f"Found doctor name: {doctor_uuid} -> {doctor_name}")
            
            # Find all DoctorIds for this doctor name, prioritizing by recent appointments
            query2 = """
            SELECT DoctorId, COUNT(*) as appointment_count, MAX(StartDateTime) as latest_appointment
            FROM View_Appointments 
            WHERE DoctorName = ? 
            GROUP BY DoctorId 
            ORDER BY latest_appointment DESC, appointment_count DESC
            """
            cursor2 = conn.execute(query2, (doctor_name,))
            results = cursor2.fetchall()
            conn.close()
            
            if results:
                # Use the DoctorId with the most recent appointments
                doctor_id = results[0][0]
                appointment_count = results[0][1]
                latest_appointment = results[0][2]
                logger.info(f"Mapped UUID to DoctorId: {doctor_uuid} -> {doctor_id} (has {appointment_count} appointments, latest: {latest_appointment})")
                
                # Log other options if they exist
                if len(results) > 1:
                    logger.info(f"Other DoctorId options for {doctor_name}: {[(r[0], r[1], r[2]) for r in results[1:]]}")
                
                return doctor_id
        
        conn.close()
        logger.warning(f"No DoctorId mapping found for UUID: {doctor_uuid}")
        return None
        
    except Exception as e:
        logger.error(f"Error mapping doctor UUID to ID for {doctor_uuid}: {e}")
        return None


def resolve_doctor_name_from_uuid(doctor_uuid: str) -> Optional[str]:
    """
    Resolve doctor UUID to human-readable name using database lookup.
    """
    try:
        # Get doctor info from COR_Doctor table using UserId (which is the UUID)
        query = "SELECT DisplayName, Firstname, Lastname FROM COR_Doctor WHERE UserId = ?"
        conn = get_db_connection()
        cursor = conn.execute(query, (doctor_uuid,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            display_name = row[0]
            first_name = row[1] 
            last_name = row[2]
            full_name = f"Dr. {first_name} {last_name}" if first_name and last_name else display_name
            logger.info(f"Resolved doctor name: {doctor_uuid} -> {full_name}")
            return full_name
        
        logger.warning(f"No doctor name found for UUID: {doctor_uuid}")
        return None
        
    except Exception as e:
        logger.error(f"Error resolving doctor name for {doctor_uuid}: {e}")
        return None


def get_next_appointment(doctor_uuid: str) -> List[Dict]:
    """Get the next upcoming appointment for a doctor using UUID or direct DoctorId."""
    # Check if doctor_uuid is already an integer DoctorId
    try:
        doctor_id = int(doctor_uuid)
        logger.info(f"Using direct DoctorId: {doctor_id}")
    except ValueError:
        # If not an integer, try to map UUID to integer DoctorId
        doctor_id = resolve_doctor_uuid_to_id(doctor_uuid)
        if doctor_id is None:
            logger.warning(f"Could not map doctor UUID {doctor_uuid} to DoctorId")
            return []
        logger.info(f"Mapped UUID {doctor_uuid} to DoctorId: {doctor_id}")
    
    query = """
    SELECT * FROM View_Appointments 
    WHERE DoctorId = ? AND StartDateTime > datetime('now') 
    ORDER BY StartDateTime LIMIT 1
    """
    return execute_query(query, (doctor_id,))


def get_patient_history(patient_identifier: str) -> List[Dict]:
    """Get patient history by name or ID."""
    query = """
    SELECT * FROM View_Appointments 
    WHERE PatientName LIKE ? OR PatientId = ?
    ORDER BY StartDateTime DESC
    """
    return execute_query(query, (f"%{patient_identifier}%", patient_identifier))


def get_doctor_schedule(doctor_uuid: str, date: str) -> List[Dict]:
    """Get doctor's schedule for a specific date using UUID."""
    # Map UUID to integer DoctorId
    doctor_id = resolve_doctor_uuid_to_id(doctor_uuid)
    if doctor_id is None:
        logger.warning(f"Could not map doctor UUID {doctor_uuid} to DoctorId")
        return []
    
    query = """
    SELECT * FROM View_Appointments 
    WHERE DoctorId = ? AND DATE(StartDateTime) = ?
    ORDER BY StartDateTime
    """
    return execute_query(query, (doctor_id, date))


def get_doctor_info(doctor_uuid: str) -> Optional[Dict]:
    """Get doctor information using UUID."""
    query = "SELECT * FROM COR_Doctor WHERE UserId = ?"
    results = execute_query(query, (doctor_uuid,))
    return results[0] if results else None

def get_status_id(status_name: str) -> Optional[int]:
    """
    Given a status name (e.g., 'Booked'), return the StatusId from View_Appointments.
    Returns None if not found.
    """
    query = "SELECT StatusId FROM View_Appointments WHERE UPPER(Status) = UPPER(?) LIMIT 1"
    results = execute_query(query, (status_name,))
    if results:
        return results[0]["StatusId"]
    return None

# --- Pure tool: Check for appointment overlap ---
def check_appointment_overlap(doctor_id: int, start_time: str, end_time: str) -> bool:
    """
    Returns True if there is an overlapping appointment for the doctor in the given time window.
    start_time and end_time should be in 'YYYY-MM-DD HH:MM:SS' format.
    """
    query = """
    SELECT 1 FROM View_Appointments
    WHERE DoctorId = ?
      AND ((StartDateTime < ? AND EndDateTime > ?) OR (StartDateTime >= ? AND StartDateTime < ?))
    LIMIT 1
    """
    # Overlap if: (existing starts before new ends AND existing ends after new starts)
    results = execute_query(query, (doctor_id, end_time, start_time, start_time, end_time))
    return bool(results)

# --- Pure tool: Check if appointment is within doctor's working hours ---
def check_doctor_working_hours(doctor_id: int, appt_time: str) -> bool:
    """
    Returns True if the appointment time is within the doctor's working hours (from COR_DoctorSchedule).
    appt_time should be in 'YYYY-MM-DD HH:MM:SS' format.
    """
    from datetime import datetime
    dt = datetime.strptime(appt_time, "%Y-%m-%d %H:%M:%S")
    # Map Python weekday (Mon=0) to DB weekday (Mon=1, ..., Sun=6)
    weekday = dt.weekday() + 1
    appt_time_str = dt.strftime("%H:%M:%S")
    # Query for working schedule for this doctor and weekday, not off, active
    query = """
        SELECT FromTime, ToTime FROM COR_DoctorSchedule
        WHERE DoctorId = ? AND WeekDay = ? AND (IsOff IS NULL OR IsOff = 0) AND (IsActive IS NULL OR IsActive = 1)
    """
    results = execute_query(query, (doctor_id, weekday))
    if not results:
        return False  # No working schedule for this day
    # Check if appt_time is within any working interval
    for row in results:
        from_time = row['FromTime']
        to_time = row['ToTime']
        if from_time and to_time:
            if from_time <= appt_time_str < to_time:
                return True
    return False

# --- Pure tool: Check if appointment is during doctor's off schedule (e.g., weekends) ---
def check_doctor_off_schedule(doctor_id: int, appt_time: str) -> bool:
    """
    Returns True if the appointment time is during the doctor's off schedule (from COR_DoctorOffSchedule).
    appt_time should be in 'YYYY-MM-DD HH:MM:SS' format.
    """
    from datetime import datetime
    dt = datetime.strptime(appt_time, "%Y-%m-%d %H:%M:%S")
    weekday = dt.weekday() + 1
    appt_time_str = dt.strftime("%H:%M:%S")
    appt_date_str = dt.strftime("%Y-%m-%d")
    # Check for explicit off schedule for this doctor, day, and time
    query = """
        SELECT FromTime, ToTime, Date FROM COR_DoctorOffSchedule
        WHERE DoctorId = ? AND (IsOff = 1) AND (IsActive IS NULL OR IsActive = 1)
          AND (WeekDay = ? OR (Date IS NOT NULL AND Date = ?))
    """
    results = execute_query(query, (doctor_id, weekday, appt_date_str))
    for row in results:
        # If Date matches, it's a specific off day
        if row.get('Date') == appt_date_str:
            from_time = row['FromTime']
            to_time = row['ToTime']
            if (not from_time and not to_time) or (from_time <= appt_time_str < to_time):
                return True
        # If WeekDay matches and no specific date, check time
        elif row.get('WeekDay') == weekday:
            from_time = row['FromTime']
            to_time = row['ToTime']
            if (not from_time and not to_time) or (from_time <= appt_time_str < to_time):
                return True
    return False