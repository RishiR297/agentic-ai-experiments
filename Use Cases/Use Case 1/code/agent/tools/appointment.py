# ==============================================
# File: agent/tools/appointment.py
# Purpose: View and book appointments via database, with LangGraph integration
# ==============================================

# -----------------------------
# Imports
# -----------------------------
from typing import Optional, List
from datetime import datetime, timedelta
from utils.db import get_db_connection
from langchain_core.tools import tool


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
        suggestions = suggest_appointment_slots(doctor_name=full_name, after=start_time)
        suggestion_list = "\n".join(suggestions)
        return (
            f"❌ Dr. {full_name} is not available from {start_time} to {end_time}.\n"
            "Here are some upcoming available slots:\n" + suggestion_list
        )

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


def get_earliest_available_slot(doctor_name: str) -> str:
    """
    Returns the earliest available appointment slot for a given doctor.
    """
    suggestions = suggest_appointment_slots(doctor_name=doctor_name, limit=1)
    return f"Earliest available slot:\n{suggestions}"



def suggest_appointment_slots(
    doctor_name: str,
    after: Optional[str] = None,
    limit: int = 5,
    weekday: Optional[int] = None
) -> str:
    """
    Suggest available appointment slots for a doctor after a given datetime,
    considering weekly schedule, off days, and existing appointments.
    """
    if not after:
        after_dt = datetime.now()
    else:
        try:
            after_dt = datetime.fromisoformat(after)
            if after_dt < datetime.now():
                after_dt = datetime.now()
            if weekday is not None:
                after_dt = after_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        except:
            after_dt = datetime.now()

    conn = get_db_connection()
    cursor = conn.cursor()

    clean_name = doctor_name.replace("Dr.", "").replace("Dr", "").strip().lower()

    # Step 1: Resolve doctor ID
    cursor.execute("""
        SELECT DISTINCT DoctorId FROM View_Appointments
        WHERE LOWER(DoctorName) LIKE ?
        LIMIT 1
    """, (f"%{clean_name}%",))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return f"❌ Could not find doctor matching '{doctor_name}'"
    doctor_id = row[0]

    # Step 2: Get working schedule
    cursor.execute("""
        SELECT WeekDay, FromTime, ToTime
        FROM COR_DoctorSchedule
        WHERE DoctorId = ? AND IsActive = 1 AND (IsOff IS NULL OR IsOff = 0)
    """, (doctor_id,))
    schedule = cursor.fetchall()

    if not schedule:
        conn.close()
        return f"❌ No active schedule found for doctor with ID {doctor_id}"

    weekday_schedule_map = {int(wd): (from_t, to_t) for wd, from_t, to_t in schedule}

    # Step 3: Get off days
    cursor.execute("""
        SELECT Date FROM COR_DoctorOffSchedule
        WHERE DoctorId = ? AND IsActive = 1 AND Date >= ?
    """, (doctor_id, after_dt.strftime('%Y-%m-%d')))
    off_dates = {datetime.fromisoformat(row[0]).date() for row in cursor.fetchall()}

    results = []
    day_cursor = after_dt
    checked = 0

    while len(results) < limit and checked < 14:
        py_weekday = day_cursor.weekday()
        db_weekday = py_weekday + 1
        day_date = day_cursor.date()

        if weekday is not None and py_weekday != weekday:
            day_cursor += timedelta(days=1)
            checked += 1
            continue

        if day_date in off_dates:
            day_cursor += timedelta(days=1)
            checked += 1
            continue

        if db_weekday not in weekday_schedule_map:
            day_cursor += timedelta(days=1)
            checked += 1
            continue

        from_time_str, to_time_str = weekday_schedule_map[db_weekday]
        work_start = datetime.combine(day_date, datetime.strptime(from_time_str[:5], "%H:%M").time())
        work_end = datetime.combine(day_date, datetime.strptime(to_time_str[:5], "%H:%M").time())

        # Step 4: Get appointments that day
        cursor.execute("""
            SELECT StartDateTime, EndDateTime FROM View_Appointments
            WHERE DoctorId = ? AND DATE(StartDateTime) = ?
        """, (doctor_id, day_date.isoformat()))
        appointments = [(datetime.fromisoformat(s), datetime.fromisoformat(e)) for s, e in cursor.fetchall()]
        appointments.sort()

        # Step 5: Subtract appointments from working hours
        current = work_start
        free_slots = []

        for appt_start, appt_end in appointments:
            if appt_start > current:
                free_slots.append((current, appt_start))
            current = max(current, appt_end)

        if current < work_end:
            free_slots.append((current, work_end))

        if free_slots:
            formatted = ", ".join(f"{s.strftime('%H:%M')} - {e.strftime('%H:%M')}" for s, e in free_slots)
            results.append(f"📅 {day_cursor.strftime('%A, %b %d')}: {formatted}")

        day_cursor += timedelta(days=1)
        checked += 1

    conn.close()

    if not results:
        return "❌ No available slots found in the next two weeks."
    return "\n".join(results)


def get_next_client_info(doctor_name: str) -> str:
    """
    Returns the next appointment (with patient details) for a given doctor.
    """
    upcoming = list_appointments(doctor_name=doctor_name, status="Confirmed", after=datetime.now().isoformat())
    return upcoming[0] if upcoming else "No upcoming clients found."


def summarize_calendar_today(doctor_name: str) -> str:
    """
    Provides a summary of today's confirmed appointments for a doctor.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT PatientName, ServiceName, StartDateTime
        FROM View_Appointments
        WHERE DoctorName LIKE ?
        AND DATE(StartDateTime) = ?
        AND Status IN ('Confirmed', 'Booked', 'Started', 'Arrived', 'Cancelled', 'Completed')
        ORDER BY StartDateTime
    """, (f"%{doctor_name}%", today))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No confirmed appointments today."

    return "\n".join(
        f"{format_time(start)}: {service} for {patient}"
        for patient, service, start in rows
    )

# -----------------------------
# MCP-Compatible Tool Registry
# -----------------------------
# Registry for LLM (OpenAI schema dicts)