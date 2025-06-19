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


# -----------------------------
# Test Block
# -----------------------------
if __name__ == "__main__":
    print("All Doctors:")
    for doc in list_all_doctors():
        user_id, fname, lname, spec_id = doc
        print(f"{user_id}: Dr. {fname} {lname} (Specialty ID: {spec_id})")
