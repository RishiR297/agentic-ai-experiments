# ==============================================
# File: tools/doctor.py
# Purpose: Provide utility functions for doctor lookup and listing
# ==============================================

# -----------------------------
# Imports
# -----------------------------
from utils.db import get_db_connection


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

    # Strip "Dr." prefix and lowercase everything for a relaxed match
    clean_name = doctor_name.replace("Dr.", "").strip().lower()

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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*)
        FROM View_Appointments
        WHERE DoctorName LIKE ? AND ServiceName = ?
    """, (f"%{doctor_name}%", service_name))
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
