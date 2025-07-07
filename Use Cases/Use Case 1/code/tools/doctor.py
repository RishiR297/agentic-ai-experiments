# ==============================================
# File: tools/doctor.py
# Purpose: Provide utility functions for doctor lookup and listing
# ==============================================

# -----------------------------
# Imports
# -----------------------------
import re
from utils.db import get_db_connection


# -----------------------------
# Function: Doctor Name Processing
# -----------------------------
def process_doctor_name(name: str, for_display: bool = True) -> str:
    """
    Process doctor name for either display or database lookup.
    
    Args:
        name: The doctor name to process
        for_display: If True, formats for display ("Dr. Name"). 
                    If False, cleans for database lookup ("name")
    
    Returns:
        Processed doctor name
    """
    if not name:
        return name
    
    # Remove common prefixes (case insensitive) - handles "doctor", "Dr.", "dr", etc.
    clean_name = re.sub(r'^(dr\.?\s*|doctor\s*)', '', name, flags=re.IGNORECASE).strip()
    
    if for_display:
        # Format for user-facing display
        return f"Dr. {clean_name.title()}"
    else:
        # Format for database lookup
        return clean_name.lower()


# -----------------------------
# Function: Find Doctor by Name
# -----------------------------
def find_doctor_by_name(name: str):
    """
    Searches for doctors whose name matches the given input using SQL LIKE.

    Note: Assumes 'DoctorName' is a valid column in COR_Doctor.
    If the schema separates Firstname and Lastname, this will need to be revised.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM COR_Doctor WHERE DoctorName LIKE ?"
    cursor.execute(query, (f"%{name}%",))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


# -----------------------------
# Function: List All Doctors
# -----------------------------
def list_all_doctors():
    """
    Fetches all doctors from COR_Doctor and returns a list of tuples:
    (UserId, Firstname, Lastname, SpecialtyId)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT UserId, Firstname, Lastname, SpecialtyId FROM COR_Doctor")
    doctors = cursor.fetchall()

    conn.close()
    return doctors


def get_branch_id_for_doctor(doctor_name: str) -> int | None:
    """
    Looks up the branch_id for a given doctor from the view_appointments view.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Use the centralized cleaning function
    clean_name = process_doctor_name(doctor_name, for_display=False)

    cursor.execute("""
        SELECT BranchId
        FROM View_Appointments
        WHERE LOWER(DoctorName) LIKE ?
        LIMIT 1
    """, (f"%{clean_name}%",))

    row = cursor.fetchone()
    conn.close()

    return row[0] if row else None

def get_services_for_doctor(doctor_name: str) -> list[str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT ServiceName
        FROM View_Appointments
        WHERE DoctorName LIKE ?
    """, (f"%{doctor_name}%",))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows if row[0]]


def is_service_valid_for_doctor(doctor_name: str, service_name: str) -> bool:
    # Extract clean name without "Dr." prefix for database lookup
    clean_name = process_doctor_name(doctor_name, for_display=False)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*)
        FROM View_Appointments
        WHERE DoctorName LIKE ? AND ServiceName = ?
    """, (f"%{clean_name}%", service_name))
    result = cursor.fetchone()[0]
    conn.close()
    return result > 0


def suggest_doctor_for_service(service_name: str) -> list[str]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT DoctorName
        FROM View_Appointments
        WHERE ServiceName = ?
    """, (service_name,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows if row[0]]


# -----------------------------
# Test Block
# -----------------------------
if __name__ == "__main__":
    print("All Doctors:")
    for doc in list_all_doctors():
        user_id, fname, lname, spec_id = doc
        print(f"{user_id}: Dr. {fname} {lname} (Specialty ID: {spec_id})")
