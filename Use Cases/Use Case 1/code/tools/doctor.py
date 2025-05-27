from utils.db import get_db_connection

def find_doctor_by_name(name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM COR_Doctor WHERE DoctorName LIKE ?"
    cursor.execute(query, (f"%{name}%",))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

from utils.db import get_db_connection

def list_all_doctors():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT UserId, Firstname, Lastname, SpecialtyId FROM COR_Doctor")
    doctors = cursor.fetchall()
    conn.close()
    return doctors

if __name__ == "__main__":
    print("All Doctors:")
    for doc in list_all_doctors():
        user_id, fname, lname, spec_id = doc
        print(f"{user_id}: Dr. {fname} {lname} (Specialty ID: {spec_id})")
