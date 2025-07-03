# ==============================================
# File: agent/tools/appointment.py
# Purpose: View and book appointments via database, with LangGraph integration
# ==============================================

# -----------------------------
# Imports
# -----------------------------
import re
import json
from typing import Optional, List
from datetime import datetime, timedelta
from utils.db import get_db_connection
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


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

    # Parse the desired times to get day of week and hour/minute
    try:
        start_dt = datetime.fromisoformat(desired_start)
        end_dt = datetime.fromisoformat(desired_end)
    except:
        conn.close()
        return False

    # Get day of week (Monday=1, Sunday=7 in database)
    db_weekday = start_dt.weekday() + 1  # Convert Python weekday to DB weekday
    start_time_str = start_dt.strftime("%H:%M:%S")
    end_time_str = end_dt.strftime("%H:%M:%S")

    # Check if doctor is scheduled to work at that time
    # Note: Database times include microseconds, so we need to handle that
    cursor.execute("""
        SELECT FromTime, ToTime FROM COR_DoctorSchedule
        WHERE DoctorId = ?
          AND WeekDay = ?
          AND IsActive = 1 AND (IsOff IS NULL OR IsOff = 0)
    """, (doctor_id, db_weekday))
    schedule_row = cursor.fetchone()
    
    if not schedule_row:
        conn.close()
        return False
    
    # Parse database times (they include microseconds)
    from_time_db, to_time_db = schedule_row
    from_time_clean = from_time_db.split('.')[0]  # Remove microseconds
    to_time_clean = to_time_db.split('.')[0]      # Remove microseconds
    
    # Compare times
    if start_time_str < from_time_clean or end_time_str > to_time_clean:
        conn.close()
        return False

    # Check for conflicting appointments
    cursor.execute("""
        SELECT 1 FROM View_Appointments
        WHERE DoctorId = ?
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
    Inserts a new appointment into the View_Appointments table.
    Assumes availability has already been verified.
    Returns a confirmation message or an error string.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get the next available AppointmentId
        cursor.execute("SELECT MAX(AppointmentId) FROM View_Appointments")
        max_id = cursor.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Get PatientId - for simplicity, we'll use a hash of the patient name
        # In a real system, this would be a proper lookup or insertion
        import hashlib
        patient_id = int(hashlib.md5(patient_name.encode()).hexdigest()[:8], 16) % 10000
        
        # Get branch name from View_Appointments_Setup if available
        cursor.execute("""
            SELECT DISTINCT BranchName FROM View_Appointments_Setup 
            WHERE BranchId = ?
            LIMIT 1
        """, (branch_id,))
        branch_result = cursor.fetchone()
        branch_name = branch_result[0] if branch_result else f"Branch {branch_id}"

        cursor.execute("""
            INSERT INTO View_Appointments (
                AppointmentId,
                PatientId,
                PatientName,
                DoctorId,
                DoctorName,
                BranchId,
                BranchName,
                CategoryId,
                CategoryName,
                ServiceId,
                ServiceName,
                MachineId,
                MachineName,
                StartDateTime,
                EndDateTime,
                StatusId,
                Status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            next_id,
            patient_id,
            patient_name,
            doctor_id,
            doctor_name,
            branch_id,
            branch_name,
            None,  # CategoryId
            None,  # CategoryName
            None,  # ServiceId
            service_name,
            None,  # MachineId
            None,  # MachineName
            start_time,
            end_time,
            1,  # StatusId for 'Booked'
            'Booked'
        ))
        conn.commit()
        return f"Appointment #{next_id} booked successfully with {doctor_name} at {branch_name} from {format_time(start_time)} to {format_time(end_time)} for {patient_name}."
    except Exception as e:
        return f"Failed to book appointment: {e}"
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
    # Step 1: Find doctor by name using normalized lookup
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Import here to avoid circular imports
    from tools.doctor import process_doctor_name
    clean_name = process_doctor_name(doctor_name, for_display=False)
    
    # First try to find the doctor ID from View_Appointments
    cursor.execute("""
        SELECT DISTINCT DoctorId, DoctorName FROM View_Appointments
        WHERE LOWER(DoctorName) LIKE ?
        LIMIT 1
    """, (f"%{clean_name}%",))
    match = cursor.fetchone()
    
    if not match:
        # Fallback: search in COR_Doctor using Firstname and Lastname
        cursor.execute("""
            SELECT UserId, Firstname, Lastname FROM COR_Doctor
            WHERE LOWER(Firstname || ' ' || Lastname) LIKE ? OR LOWER(DisplayName) LIKE ?
        """, (f"%{clean_name}%", f"%{clean_name}%"))
        doctor_row = cursor.fetchone()
        
        if not doctor_row:
            conn.close()
            return f"No doctor found matching '{doctor_name}'"
        
        doctor_id = doctor_row[0]
        full_name = f"{doctor_row[1]} {doctor_row[2]}"
    else:
        doctor_id = match[0]
        full_name = match[1]

    conn.close()

    # Step 2: Check availability
    if not is_doctor_available(doctor_id, start_time, end_time):
        suggestions = suggest_appointment_slots(doctor_name=full_name, after=start_time)
        return (
            f"Dr. {full_name} is not available from {format_time(start_time)} to {format_time(end_time)}.\n"
            "Here are some upcoming available slots:\n" + suggestions
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

    # Import here to avoid circular imports
    from tools.doctor import process_doctor_name
    clean_name = process_doctor_name(doctor_name, for_display=False)

    # Step 1: Resolve doctor ID
    cursor.execute("""
        SELECT DISTINCT DoctorId FROM View_Appointments
        WHERE LOWER(DoctorName) LIKE ?
        LIMIT 1
    """, (f"%{clean_name}%",))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return f" Could not find doctor matching '{doctor_name}'"
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
        return f" No active schedule found for doctor with ID {doctor_id}"

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
        return " No available slots found in the next two weeks."
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

# -----------------------------
# Imports
# -----------------------------
import re
import json
from typing import Optional, List
from datetime import datetime, timedelta
from utils.db import get_db_connection
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Import the LLM for slot detection (will need to be passed as parameter)
# from agent.nodes import llm_basic  # This will cause circular import


# -----------------------------
# Helper: Slot Parsing and Detection
# -----------------------------
def clean_date_line(line: str) -> str:
    """Remove emoji but preserve the original formatting."""
    return line.replace("📅", "").strip()


def parse_slot_line(line: str) -> dict:
    """Parse a slot line to extract start and end times."""
    match = re.search(r"(\w{3,}), (\w{3}) (\d{1,2}): (\d{1,2}:\d{2}) - (\d{1,2}:\d{2})", line)
    if not match:
        return {}

    _, month, day, start_t, end_t = match.groups()
    try:
        from dateutil import parser as date_parser
        base_year = datetime.now().year
        slot_date = date_parser.parse(f"{month} {day} {base_year}")
        
        # Ensure we always return the next valid future date
        now = datetime.now()
        if slot_date.date() < now.date():
            slot_date = date_parser.parse(f"{month} {day} {base_year + 1}")

        start_dt = datetime.combine(slot_date.date(), datetime.strptime(start_t, "%H:%M").time())
        end_dt = datetime.combine(slot_date.date(), datetime.strptime(end_t, "%H:%M").time())

        return {
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat()
        }
    except Exception as e:
        print(f"[Slot Parse Error] {e}")
        return {}


def detect_selected_slot_simple(user_input: str, slot_lines: List[dict]) -> dict:
    """
    Simple slot detection using ordinal matching.
    Returns a dict with start_time and end_time if match is found.
    """
    if not slot_lines:
        return {}
    
    user_input_lower = user_input.lower()
    
    # Simple ordinal matching
    ordinal_map = {
        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
        "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4
    }
    
    for word, idx in ordinal_map.items():
        if word in user_input_lower and idx < len(slot_lines):
            print(f"[DEBUG detect_selected_slot] Matched ordinal '{word}' to slot index {idx}")
            return {
                "start_time": slot_lines[idx]["start_time"],
                "end_time": slot_lines[idx]["end_time"]
            }
    
    return {}


def detect_selected_slot_with_llm(user_input: str, slot_lines: List[dict], llm_basic) -> dict:
    """
    Advanced slot detection using LLM inference.
    Requires an LLM instance to be passed in to avoid circular imports.
    """
    if not slot_lines:
        print("[DEBUG detect_selected_slot] No available slots to match against")
        return {}
    
    print(f"[DEBUG detect_selected_slot] user_input: '{user_input}'")
    print(f"[DEBUG detect_selected_slot] available slots: {len(slot_lines)} slots")

    # Try simple detection first
    simple_result = detect_selected_slot_simple(user_input, slot_lines)
    if simple_result:
        return simple_result

    # Use LLM for more complex date/time inference
    slot_displays = [slot.get("display", "") for slot in slot_lines]
    
    prompt = f"""
    You are a slot selection assistant. The user wants to select one of the available appointment slots.
    
    User input: "{user_input}"
    
    Available slots:
    {chr(10).join([f"{i+1}. {display}" for i, display in enumerate(slot_displays)])}
    
    Instructions:
    - Analyze the user's input to determine which slot they're referring to
    - Look for date references (like "jul 10", "july 10", "10th", "today", etc.)
    - Look for ordinal references (like "first", "second", "1st", "2nd", etc.)
    - Consider context and natural language variations
    - If the user mentions a specific date, match it to the corresponding slot
    - Current date: {datetime.now().strftime('%A, %B %d, %Y')}
    
    Return ONLY a JSON object with the slot number (1-based index) or null if no clear match:
    {{"selected_slot": 2}} or {{"selected_slot": null}}
    
    Do not include any explanation, just the JSON.
    """

    try:
        response = llm_basic.invoke([HumanMessage(content=prompt)])
        result = json.loads(response.content.strip())
        
        selected_slot = result.get("selected_slot")
        if selected_slot is not None and 1 <= selected_slot <= len(slot_lines):
            slot_index = selected_slot - 1  # Convert to 0-based index
            print(f"[DEBUG detect_selected_slot] LLM selected slot {selected_slot} (index {slot_index})")
            print(f"[DEBUG detect_selected_slot] Selected slot: {slot_displays[slot_index]}")
            
            return {
                "start_time": slot_lines[slot_index]["start_time"],
                "end_time": slot_lines[slot_index]["end_time"]
            }
        else:
            print(f"[DEBUG detect_selected_slot] LLM returned invalid slot number: {selected_slot}")
            
    except Exception as e:
        print(f"[DEBUG detect_selected_slot] LLM inference failed: {e}")

    print("[DEBUG detect_selected_slot] No slot matched.")
    return {}