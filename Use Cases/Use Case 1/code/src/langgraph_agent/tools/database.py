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

# Database path
DB_PATH = "./db/output.db"


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
    """Execute database query and return results"""
    conn = get_db_connection()
    try:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        logger.info(f"Query executed successfully. {len(results)} rows returned.")
        return results
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
    """Get the next upcoming appointment for a doctor using UUID."""
    # Map UUID to integer DoctorId
    doctor_id = resolve_doctor_uuid_to_id(doctor_uuid)
    if doctor_id is None:
        logger.warning(f"Could not map doctor UUID {doctor_uuid} to DoctorId")
        return []
    
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
