# ==============================================
# File: agent/tools/appointment.py
# Purpose: View and book appointments via database, with LangGraph integration
# ==============================================

# -----------------------------
# Imports
# -----------------------------
import re
import json
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
from utils.db import get_db_connection
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# Service duration map for quick reference (using uppercase keys to match formatted names)
SERVICE_DURATION_MAP = {
    'CONSULTATION': 15,
    'BOTOX': 14,
    'BOTOX TREATMENT': 14,
    'BBL': 14,
    'FACIAL': 14,
    'EVOLVE X': 14,
    'SCULPTRA': 14,
    'RADIESSE': 14,
    'DNA TEST': 14,
    'ULTHERAPY': 39,
    'PRP': 29,
    'PRP THERAPY': 29,
    'LASER HAIR REMOVAL': 60,
}

# -----------------------------
# Helper Functions (defined early for use in other functions)
# -----------------------------

def format_patient_name(name: str) -> str:
    """
    Formats patient name to Title Case (First Letter Capitalized for each word).
    Examples: 
    - "john doe" -> "John Doe"
    - "mary-jane smith" -> "Mary-Jane Smith"
    - "JANE DOE" -> "Jane Doe"
    """
    if not name:
        return ""
    
    # Handle hyphenated names and multiple words
    return ' '.join(
        '-'.join(part.capitalize() for part in word.split('-'))
        for word in name.strip().split()
    )


def format_doctor_name(name: str) -> str:
    """
    Formats doctor name to Title Case for consistency.
    Handles "Dr." prefix properly.
    Examples:
    - "john smith" -> "John Smith"
    - "dr. jane doe" -> "Dr. Jane Doe"
    - "ANTONELLA" -> "Antonella"
    """
    if not name:
        return ""
    
    # Split by spaces and capitalize each part
    parts = name.strip().split()
    formatted_parts = []
    
    for part in parts:
        if part.lower() == "dr." or part.lower() == "dr":
            formatted_parts.append("Dr.")
        else:
            formatted_parts.append(part.capitalize())
    
    return " ".join(formatted_parts)


def format_service_name(service: str) -> str:
    """
    Formats service name to ALL UPPERCASE to match database convention.
    Examples:
    - "consultation" -> "CONSULTATION"
    - "Botox Treatment" -> "BOTOX TREATMENT"
    - "laser hair removal" -> "LASER HAIR REMOVAL"
    """
    if not service:
        return ""
    
    # Clean up multiple spaces and convert to uppercase
    return ' '.join(service.strip().split()).upper()


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
# Helper: format_patient_name
# -----------------------------
def format_patient_name(name: str) -> str:
    """
    Formats patient name to Title Case (First Letter Capitalized for each word).
    Examples: 
    - "john doe" -> "John Doe"
    - "mary-jane smith" -> "Mary-Jane Smith"
    - "JANE DOE" -> "Jane Doe"
    """
    if not name:
        return ""
    
    # Handle hyphenated names and multiple words
    return ' '.join(
        '-'.join(part.capitalize() for part in word.split('-'))
        for word in name.strip().split()
    )


# -----------------------------
# Helper: format_doctor_name  
# -----------------------------
def format_doctor_name(name: str) -> str:
    """
    Formats doctor name to Title Case for consistency.
    Handles "Dr." prefix properly.
    Examples:
    - "john smith" -> "John Smith"
    - "dr. jane doe" -> "Dr. Jane Doe"
    - "ANTONELLA" -> "Antonella"
    """
    if not name:
        return ""
    
    # Split by spaces and capitalize each part
    parts = name.strip().split()
    formatted_parts = []
    
    for part in parts:
        if part.lower() == "dr." or part.lower() == "dr":
            formatted_parts.append("Dr.")
        else:
            formatted_parts.append(part.capitalize())
    
    return " ".join(formatted_parts)


# -----------------------------
# Helper: format_service_name
# -----------------------------
def format_service_name(service: str) -> str:
    """
    Formats service name to ALL UPPERCASE to match database convention.
    Examples:
    - "consultation" -> "CONSULTATION"
    - "Botox Treatment" -> "BOTOX TREATMENT"
    - "laser hair removal" -> "LASER HAIR REMOVAL"
    """
    if not service:
        return ""
    
    # Clean up multiple spaces and convert to uppercase
    return ' '.join(service.strip().split()).upper()


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
    Queries appointments from the View_Appointments view, optionally filtered by doctor, patient, status, and time range.
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
    after: Optional[str] = None,
    limit: Optional[int] = None,
    before: Optional[str] = None
) -> str:
    """
    LangChain-compatible tool that fetches and returns formatted appointments as a string.
    Filters: doctor_name, status, start_time (after), end_time (before), and optionally limits results.
    
    Smart time filtering:
    - If no time filters are provided, shows appointments from today onwards (includes today)
    - If only doctor_name is provided, shows recent and upcoming appointments for that doctor
    - If specific time filters are provided, uses those exactly
    """
    from datetime import datetime, timedelta
    
    # Smart default time filtering if none provided
    if after is None and before is None:
        # Start from beginning of today to include today's appointments
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        after = today_start.isoformat()
        
        # If looking for a specific doctor, show wider range (past 2 weeks + next 2 weeks)
        if doctor_name:
            after = (today_start - timedelta(weeks=2)).isoformat()
            before = (today_start + timedelta(weeks=2)).isoformat()
        else:
            # Default to next week for general queries
            before = (today_start + timedelta(weeks=1)).isoformat()
    
    # Default limit if none provided
    if limit is None:
        limit = 20  # Increased limit for better visibility
    
    appointments = list_appointments(
        doctor_name=doctor_name,
        status=status,
        after=after,
        before=before
    )
    
    # Apply limit if specified
    if limit > 0:
        appointments = appointments[:limit]
    
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
        # Format inputs to match database conventions
        formatted_patient_name = format_patient_name(patient_name)
        formatted_service_name = format_service_name(service_name)
        
        # Get the next available AppointmentId
        cursor.execute("SELECT MAX(AppointmentId) FROM View_Appointments")
        max_id = cursor.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Get PatientId - for simplicity, we'll use a hash of the formatted patient name
        # In a real system, this would be a proper lookup or insertion
        import hashlib
        patient_id = int(hashlib.md5(formatted_patient_name.encode()).hexdigest()[:8], 16) % 10000
        
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
            formatted_patient_name,  # Use formatted patient name
            doctor_id,
            doctor_name,
            branch_id,
            branch_name,
            None,  # CategoryId
            None,  # CategoryName
            None,  # ServiceId
            formatted_service_name,  # Use formatted service name
            None,  # MachineId
            None,  # MachineName
            start_time,
            end_time,
            1,  # StatusId for 'Booked'
            'Booked'
        ))
        conn.commit()
        return f"Appointment #{next_id} booked successfully with {doctor_name} at {branch_name} from {format_time(start_time)} to {format_time(end_time)} for {formatted_patient_name}."
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
    end_time: Optional[str] = None
) -> str:
    """
    Enhanced LangChain-compatible tool that books an appointment with conflict checking.
    Now includes:
    - Automatic name and service formatting to match database conventions
    - Automatic duration calculation based on service type
    - Conflict prevention with existing appointments
    - Improved error messages with alternative suggestions
    """
    # Format inputs to match database conventions
    formatted_patient_name = format_patient_name(patient_name)
    formatted_service_name = format_service_name(service_name)
    
    try:
        # Parse start time
        start_dt = datetime.fromisoformat(start_time.replace('T', ' '))
        
        # Auto-calculate end time if not provided, based on service duration
        if not end_time:
            service_duration = get_service_duration(formatted_service_name)
            end_dt = start_dt + timedelta(minutes=service_duration)
            end_time = end_dt.isoformat()
        else:
            end_dt = datetime.fromisoformat(end_time.replace('T', ' '))
        
        # Check for appointment conflicts BEFORE attempting to book
        if check_appointment_conflict(doctor_name, start_dt, end_dt):
            suggestions = suggest_appointment_slots(
                doctor_name=doctor_name, 
                after=start_time,
                service_name=formatted_service_name,
                limit=3
            )
            return (
                f"❌ CONFLICT: Dr. {doctor_name} already has an appointment during "
                f"{start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%H:%M')}.\n\n"
                f"📅 Alternative slots:\n{suggestions}"
            )
            
    except Exception as e:
        return f"❌ Invalid time format: {e}"
    
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
            return f"❌ No doctor found matching '{doctor_name}'"
        
        doctor_id = doctor_row[0]
        full_name = f"{doctor_row[1]} {doctor_row[2]}"
    else:
        doctor_id = match[0]
        full_name = match[1]

    conn.close()

    # Step 2: Check doctor's working hours for that day
    working_hours = get_doctor_working_hours(doctor_name, start_dt)
    if not working_hours:
        return (
            f"❌ Dr. {full_name} is not working on {start_dt.strftime('%A, %B %d, %Y')}.\n"
            f"Please choose a different date."
        )
    
    work_start, work_end = working_hours
    if start_dt < work_start or end_dt > work_end:
        return (
            f"❌ Requested time {start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')} "
            f"is outside Dr. {full_name}'s working hours "
            f"({work_start.strftime('%H:%M')}-{work_end.strftime('%H:%M')}) on {start_dt.strftime('%A')}."
        )

    # Step 3: Book the appointment
    result = create_appointment(
        doctor_id=doctor_id,
        doctor_name=full_name,
        patient_name=formatted_patient_name,  # Use formatted patient name
        branch_id=branch_id,
        service_name=formatted_service_name,  # Use formatted service name
        start_time=start_time,
        end_time=end_time
    )
    
    # Add confirmation details
    duration = (end_dt - start_dt).total_seconds() / 60
    return (
        f"✅ BOOKING CONFIRMED!\n"
        f"📅 {start_dt.strftime('%A, %B %d, %Y')}\n"
        f"🕐 {start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')} ({duration:.0f} min)\n"
        f"👨‍⚕️ Dr. {full_name}\n"
        f"👤 Patient: {formatted_patient_name}\n"
        f"🔬 Service: {formatted_service_name}\n\n"
        f"{result}"
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
    weekday: Optional[int] = None,
    service_name: Optional[str] = "CONSULTATION"
) -> str:
    """
    Enhanced slot suggestion that considers service durations and prevents conflicts.
    Suggest available appointment slots for a doctor after a given datetime,
    taking into account:
    - Service duration (based on service type)
    - Doctor's working schedule 
    - Existing appointments (to prevent conflicts)
    - Off days
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

    # Default and format service name
    if not service_name:
        service_name = "CONSULTATION"
    
    formatted_service_name = format_service_name(service_name)

    # Get service duration
    service_duration = get_service_duration(formatted_service_name)
    
    results = []
    day_cursor = after_dt
    checked = 0

    while len(results) < limit and checked < 14:  # Check up to 2 weeks
        py_weekday = day_cursor.weekday()
        day_date = day_cursor.date()

        # Skip if specific weekday requested and doesn't match
        if weekday is not None and py_weekday != weekday:
            day_cursor += timedelta(days=1)
            checked += 1
            continue

        # Generate available slots for this day using enhanced logic
        available_slots = generate_available_slots(doctor_name, day_cursor, formatted_service_name)
        
        if available_slots:
            # Format the first few slots for this day
            day_slots = []
            for start_dt, end_dt in available_slots[:3]:  # Show up to 3 slots per day
                start_time = start_dt.strftime('%H:%M')
                end_time = end_dt.strftime('%H:%M')
                day_slots.append(f"{start_time}-{end_time}")
            
            if day_slots:
                slots_str = ", ".join(day_slots)
                results.append(f"📅 {day_cursor.strftime('%A, %b %d')}: {slots_str} ({service_duration} min slots)")

        day_cursor += timedelta(days=1)
        checked += 1

    if not results:
        return f"❌ No available {service_duration}-minute slots found for {formatted_service_name} in the next two weeks."
    
    return "🎯 Available appointment slots:\n" + "\n".join(results)


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
    Simple slot detection using ordinal and date matching.
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
    
    # Enhanced date matching
    for i, slot in enumerate(slot_lines):
        display = slot.get("display", "").lower()
        
        # Check for various date patterns
        if "jul 07" in display or "july 07" in display:
            if any(pattern in user_input_lower for pattern in ["july 7", "jul 7", "7th", "july 07", "jul 07"]):
                print(f"[DEBUG detect_selected_slot] Matched date pattern for Jul 07 to slot index {i}")
                return {
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"]
                }
        elif "jul 14" in display or "july 14" in display:
            if any(pattern in user_input_lower for pattern in ["july 14", "jul 14", "14th", "july 14th"]):
                print(f"[DEBUG detect_selected_slot] Matched date pattern for Jul 14 to slot index {i}")
                return {
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"]
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

    # Enhanced date matching before LLM
    user_lower = user_input.lower()
    
    # Check for date patterns like "july 7", "jul 7", "7th", etc.
    for i, slot in enumerate(slot_lines):
        display = slot.get("display", "").lower()
        
        # Extract date parts from display (e.g., "Monday, Jul 07: 10:00 - 18:00")
        if "jul 07" in display or "july 07" in display or "july 7" in display:
            if any(pattern in user_lower for pattern in ["july 7", "jul 7", "7th", "july 07", "jul 07"]):
                print(f"[DEBUG detect_selected_slot] Direct date match found for slot {i+1}")
                return {
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"]
                }
        elif "jul 14" in display or "july 14" in display:
            if any(pattern in user_lower for pattern in ["july 14", "jul 14", "14th", "july 14th"]):
                print(f"[DEBUG detect_selected_slot] Direct date match found for slot {i+1}")
                return {
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"]
                }
    
    # Use LLM for more complex date/time inference
    slot_displays = [slot.get("display", "") for slot in slot_lines]
    
    prompt = f"""You are a slot selection assistant. The user wants to select one of the available appointment slots.

User input: "{user_input}"

Available slots:
{chr(10).join([f"{i+1}. {display}" for i, display in enumerate(slot_displays)])}

Instructions:
- Analyze the user's input to determine which slot they're referring to
- Look for date references (like "jul 7", "july 7", "7th", "today", etc.)
- Look for ordinal references (like "first", "second", "1st", "2nd", etc.)
- Consider context and natural language variations
- If the user mentions a specific date, match it to the corresponding slot
- Current date: {datetime.now().strftime('%A, %B %d, %Y')}

Examples:
- "july 7th" or "jul 7" should match slot with "Jul 07"
- "july 14" or "14th" should match slot with "Jul 14"
- "first" or "1st" should match slot 1
- "second" or "2nd" should match slot 2

Return ONLY a JSON object with the slot number (1-based index) or null if no clear match:
{{"selected_slot": 1}} or {{"selected_slot": null}}

Do not include any explanation, just the JSON."""

    try:
        response = llm_basic.invoke([HumanMessage(content=prompt)])
        response_content = response.content.strip()
        print(f"[DEBUG detect_selected_slot] LLM response: {response_content}")
        
        # Clean up the response in case there are extra characters
        if response_content.startswith('```json'):
            response_content = response_content[7:-3].strip()
        elif response_content.startswith('```'):
            response_content = response_content[3:-3].strip()
        
        result = json.loads(response_content)
        
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
        print(f"[DEBUG detect_selected_slot] Raw response: {response.content if 'response' in locals() else 'No response'}")

    print("[DEBUG detect_selected_slot] No slot matched.")
    return {}


# -----------------------------
# Enhanced Appointment Logic Functions
# -----------------------------

def get_service_duration(service_name: str) -> int:
    """
    Returns the duration in minutes for a given service.
    Falls back to 15 minutes (consultation) if service not found.
    """
    if not service_name:
        return 15
    
    # Format service name to match the map
    formatted_service = format_service_name(service_name) if service_name else "CONSULTATION"
    
    # Return duration from map, default to 15 minutes
    return SERVICE_DURATION_MAP.get(formatted_service, 15)


def get_doctor_working_hours(doctor_name: str, date_dt: datetime) -> Optional[Tuple[datetime, datetime]]:
    """
    Returns a tuple of (work_start_datetime, work_end_datetime) for a doctor on a specific date.
    Returns None if the doctor is not working that day.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get the weekday (Monday=1, Sunday=7 in database)
        db_weekday = date_dt.weekday() + 1
        
        # Find doctor's schedule for that weekday using View_Appointments to map doctor name to DoctorId
        # Handle both "Dr. Name" and "Name" formats
        doctor_name_clean = doctor_name.replace('Dr. ', '').strip()
        cursor.execute("""
            SELECT ds.FromTime, ds.ToTime
            FROM COR_DoctorSchedule ds
            WHERE ds.DoctorId = (
                SELECT DISTINCT DoctorId 
                FROM View_Appointments 
                WHERE DoctorName LIKE ? OR DoctorName LIKE ?
                LIMIT 1
            )
              AND ds.WeekDay = ?
              AND ds.IsActive = 1 
              AND (ds.IsOff IS NULL OR ds.IsOff = 0)
            LIMIT 1
        """, (f"%{doctor_name}%", f"%{doctor_name_clean}%", db_weekday))
        
        result = cursor.fetchone()
        if not result:
            return None
            
        from_time_str, to_time_str = result
        
        # Parse times (handle microseconds if present)
        from_time_clean = from_time_str.split('.')[0] if '.' in from_time_str else from_time_str
        to_time_clean = to_time_str.split('.')[0] if '.' in to_time_str else to_time_str
        
        # Combine date with times
        from_time = datetime.strptime(from_time_clean, "%H:%M:%S").time()
        to_time = datetime.strptime(to_time_clean, "%H:%M:%S").time()
        
        work_start = datetime.combine(date_dt.date(), from_time)
        work_end = datetime.combine(date_dt.date(), to_time)
        
        return (work_start, work_end)
        
    except Exception as e:
        print(f"Error getting doctor working hours: {e}")
        return None
    finally:
        conn.close()


def check_appointment_conflict(doctor_name: str, start_dt: datetime, end_dt: datetime) -> bool:
    """
    Checks if there's a conflicting appointment for the doctor at the given time.
    Returns True if there's a conflict, False otherwise.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check for overlapping appointments
        # Handle both "Dr. Name" and "Name" formats
        doctor_name_clean = doctor_name.replace('Dr. ', '').strip()
        cursor.execute("""
            SELECT COUNT(*)
            FROM View_Appointments
            WHERE (DoctorName LIKE ? OR DoctorName LIKE ?)
              AND Status IN ('Booked', 'Confirmed', 'Started', 'Arrived')
              AND (? < EndDateTime AND ? > StartDateTime)
        """, (f"%{doctor_name}%", f"%{doctor_name_clean}%", start_dt.isoformat(), end_dt.isoformat()))
        
        conflict_count = cursor.fetchone()[0]
        return conflict_count > 0
        
    except Exception as e:
        print(f"Error checking appointment conflict: {e}")
        return True  # Assume conflict if there's an error
    finally:
        conn.close()


def generate_available_slots(doctor_name: str, date_dt: datetime, service_name: str) -> List[Tuple[datetime, datetime]]:
    """
    Generates available appointment slots for a doctor on a specific date.
    Returns a list of (start_datetime, end_datetime) tuples.
    """
    # Get doctor's working hours for this day
    working_hours = get_doctor_working_hours(doctor_name, date_dt)
    if not working_hours:
        return []
    
    work_start, work_end = working_hours
    service_duration = get_service_duration(service_name)
    
    # Generate all possible slots with 15-minute intervals
    available_slots = []
    current_time = work_start
    
    while current_time + timedelta(minutes=service_duration) <= work_end:
        slot_end = current_time + timedelta(minutes=service_duration)
        
        # Check if this slot conflicts with existing appointments
        if not check_appointment_conflict(doctor_name, current_time, slot_end):
            available_slots.append((current_time, slot_end))
        
        # Move to next 15-minute interval
        current_time += timedelta(minutes=15)
    
    return available_slots


def get_doctor_existing_appointments(doctor_name: str, date_dt: datetime) -> List[Tuple[datetime, datetime]]:
    """
    Gets existing appointments for a doctor on a specific date.
    Returns a list of (start_datetime, end_datetime) tuples.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        date_str = date_dt.strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT StartDateTime, EndDateTime
            FROM View_Appointments
            WHERE DoctorName LIKE ?
              AND DATE(StartDateTime) = ?
              AND Status IN ('Booked', 'Confirmed', 'Started', 'Arrived')
            ORDER BY StartDateTime
        """, (f"%{doctor_name}%", date_str))
        
        results = cursor.fetchall()
        appointments = []
        
        for start_str, end_str in results:
            start_dt = datetime.fromisoformat(start_str.split('.')[0])
            end_dt = datetime.fromisoformat(end_str.split('.')[0])
            appointments.append((start_dt, end_dt))
            
        return appointments
        
    except Exception as e:
        print(f"Error getting existing appointments: {e}")
        return []
    finally:
        conn.close()


def format_available_slots(slots: List[Tuple[datetime, datetime]]) -> str:
    """
    Formats a list of available slots into a human-readable string.
    """
    if not slots:
        return "No available slots"
    
    formatted_slots = []
    for start_dt, end_dt in slots[:5]:  # Limit to 5 slots
        start_time = start_dt.strftime('%H:%M')
        end_time = end_dt.strftime('%H:%M')
        formatted_slots.append(f"{start_time}-{end_time}")
    
    return ", ".join(formatted_slots)


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
    Queries appointments from the View_Appointments view, optionally filtered by doctor, patient, status, and time range.
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
    after: Optional[str] = None,
    limit: Optional[int] = None,
    before: Optional[str] = None
) -> str:
    """
    LangChain-compatible tool that fetches and returns formatted appointments as a string.
    Filters: doctor_name, status, start_time (after), end_time (before), and optionally limits results.
    
    Smart time filtering:
    - If no time filters are provided, shows appointments from today onwards (includes today)
    - If only doctor_name is provided, shows recent and upcoming appointments for that doctor
    - If specific time filters are provided, uses those exactly
    """
    from datetime import datetime, timedelta
    
    # Smart default time filtering if none provided
    if after is None and before is None:
        # Start from beginning of today to include today's appointments
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        after = today_start.isoformat()
        
        # If looking for a specific doctor, show wider range (past 2 weeks + next 2 weeks)
        if doctor_name:
            after = (today_start - timedelta(weeks=2)).isoformat()
            before = (today_start + timedelta(weeks=2)).isoformat()
        else:
            # Default to next week for general queries
            before = (today_start + timedelta(weeks=1)).isoformat()
    
    # Default limit if none provided
    if limit is None:
        limit = 20  # Increased limit for better visibility
    
    appointments = list_appointments(
        doctor_name=doctor_name,
        status=status,
        after=after,
        before=before
    )
    
    # Apply limit if specified
    if limit > 0:
        appointments = appointments[:limit]
    
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
        # Format inputs to match database conventions
        formatted_patient_name = format_patient_name(patient_name)
        formatted_service_name = format_service_name(service_name)
        
        # Get the next available AppointmentId
        cursor.execute("SELECT MAX(AppointmentId) FROM View_Appointments")
        max_id = cursor.fetchone()[0]
        next_id = (max_id or 0) + 1
        
        # Get PatientId - for simplicity, we'll use a hash of the formatted patient name
        # In a real system, this would be a proper lookup or insertion
        import hashlib
        patient_id = int(hashlib.md5(formatted_patient_name.encode()).hexdigest()[:8], 16) % 10000
        
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
            formatted_patient_name,  # Use formatted patient name
            doctor_id,
            doctor_name,
            branch_id,
            branch_name,
            None,  # CategoryId
            None,  # CategoryName
            None,  # ServiceId
            formatted_service_name,  # Use formatted service name
            None,  # MachineId
            None,  # MachineName
            start_time,
            end_time,
            1,  # StatusId for 'Booked'
            'Booked'
        ))
        conn.commit()
        return f"Appointment #{next_id} booked successfully with {doctor_name} at {branch_name} from {format_time(start_time)} to {format_time(end_time)} for {formatted_patient_name}."
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
    end_time: Optional[str] = None
) -> str:
    """
    Enhanced LangChain-compatible tool that books an appointment with conflict checking.
    Now includes:
    - Automatic name and service formatting to match database conventions
    - Automatic duration calculation based on service type
    - Conflict prevention with existing appointments
    - Improved error messages with alternative suggestions
    """
    # Format inputs to match database conventions
    formatted_patient_name = format_patient_name(patient_name)
    formatted_service_name = format_service_name(service_name)
    
    try:
        # Parse start time
        start_dt = datetime.fromisoformat(start_time.replace('T', ' '))
        
        # Auto-calculate end time if not provided, based on service duration
        if not end_time:
            service_duration = get_service_duration(formatted_service_name)
            end_dt = start_dt + timedelta(minutes=service_duration)
            end_time = end_dt.isoformat()
        else:
            end_dt = datetime.fromisoformat(end_time.replace('T', ' '))
        
        # Check for appointment conflicts BEFORE attempting to book
        if check_appointment_conflict(doctor_name, start_dt, end_dt):
            suggestions = suggest_appointment_slots(
                doctor_name=doctor_name, 
                after=start_time,
                service_name=formatted_service_name,
                limit=3
            )
            return (
                f"❌ CONFLICT: Dr. {doctor_name} already has an appointment during "
                f"{start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%H:%M')}.\n\n"
                f"📅 Alternative slots:\n{suggestions}"
            )
            
    except Exception as e:
        return f"❌ Invalid time format: {e}"
    
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
            return f"❌ No doctor found matching '{doctor_name}'"
        
        doctor_id = doctor_row[0]
        full_name = f"{doctor_row[1]} {doctor_row[2]}"
    else:
        doctor_id = match[0]
        full_name = match[1]

    conn.close()

    # Step 2: Check doctor's working hours for that day
    working_hours = get_doctor_working_hours(doctor_name, start_dt)
    if not working_hours:
        return (
            f"❌ Dr. {full_name} is not working on {start_dt.strftime('%A, %B %d, %Y')}.\n"
            f"Please choose a different date."
        )
    
    work_start, work_end = working_hours
    if start_dt < work_start or end_dt > work_end:
        return (
            f"❌ Requested time {start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')} "
            f"is outside Dr. {full_name}'s working hours "
            f"({work_start.strftime('%H:%M')}-{work_end.strftime('%H:%M')}) on {start_dt.strftime('%A')}."
        )

    # Step 3: Book the appointment
    result = create_appointment(
        doctor_id=doctor_id,
        doctor_name=full_name,
        patient_name=formatted_patient_name,  # Use formatted patient name
        branch_id=branch_id,
        service_name=formatted_service_name,  # Use formatted service name
        start_time=start_time,
        end_time=end_time
    )
    
    # Add confirmation details
    duration = (end_dt - start_dt).total_seconds() / 60
    return (
        f"✅ BOOKING CONFIRMED!\n"
        f"📅 {start_dt.strftime('%A, %B %d, %Y')}\n"
        f"🕐 {start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')} ({duration:.0f} min)\n"
        f"👨‍⚕️ Dr. {full_name}\n"
        f"👤 Patient: {formatted_patient_name}\n"
        f"🔬 Service: {formatted_service_name}\n\n"
        f"{result}"
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
    weekday: Optional[int] = None,
    service_name: Optional[str] = "CONSULTATION"
) -> str:
    """
    Enhanced slot suggestion that considers service durations and prevents conflicts.
    Suggest available appointment slots for a doctor after a given datetime,
    taking into account:
    - Service duration (based on service type)
    - Doctor's working schedule 
    - Existing appointments (to prevent conflicts)
    - Off days
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

    # Default and format service name
    if not service_name:
        service_name = "CONSULTATION"
    
    formatted_service_name = format_service_name(service_name)

    # Get service duration
    service_duration = get_service_duration(formatted_service_name)
    
    results = []
    day_cursor = after_dt
    checked = 0

    while len(results) < limit and checked < 14:  # Check up to 2 weeks
        py_weekday = day_cursor.weekday()
        day_date = day_cursor.date()

        # Skip if specific weekday requested and doesn't match
        if weekday is not None and py_weekday != weekday:
            day_cursor += timedelta(days=1)
            checked += 1
            continue

        # Generate available slots for this day using enhanced logic
        available_slots = generate_available_slots(doctor_name, day_cursor, formatted_service_name)
        
        if available_slots:
            # Format the first few slots for this day
            day_slots = []
            for start_dt, end_dt in available_slots[:3]:  # Show up to 3 slots per day
                start_time = start_dt.strftime('%H:%M')
                end_time = end_dt.strftime('%H:%M')
                day_slots.append(f"{start_time}-{end_time}")
            
            if day_slots:
                slots_str = ", ".join(day_slots)
                results.append(f"📅 {day_cursor.strftime('%A, %b %d')}: {slots_str} ({service_duration} min slots)")

        day_cursor += timedelta(days=1)
        checked += 1

    if not results:
        return f"❌ No available {service_duration}-minute slots found for {formatted_service_name} in the next two weeks."
    
    return "🎯 Available appointment slots:\n" + "\n".join(results)


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
    Simple slot detection using ordinal and date matching.
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
    
    # Enhanced date matching
    for i, slot in enumerate(slot_lines):
        display = slot.get("display", "").lower()
        
        # Check for various date patterns
        if "jul 07" in display or "july 07" in display:
            if any(pattern in user_input_lower for pattern in ["july 7", "jul 7", "7th", "july 07", "jul 07"]):
                print(f"[DEBUG detect_selected_slot] Matched date pattern for Jul 07 to slot index {i}")
                return {
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"]
                }
        elif "jul 14" in display or "july 14" in display:
            if any(pattern in user_input_lower for pattern in ["july 14", "jul 14", "14th", "july 14th"]):
                print(f"[DEBUG detect_selected_slot] Matched date pattern for Jul 14 to slot index {i}")
                return {
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"]
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

    # Enhanced date matching before LLM
    user_lower = user_input.lower()
    
    # Check for date patterns like "july 7", "jul 7", "7th", etc.
    for i, slot in enumerate(slot_lines):
        display = slot.get("display", "").lower()
        
        # Extract date parts from display (e.g., "Monday, Jul 07: 10:00 - 18:00")
        if "jul 07" in display or "july 07" in display or "july 7" in display:
            if any(pattern in user_lower for pattern in ["july 7", "jul 7", "7th", "july 07", "jul 07"]):
                print(f"[DEBUG detect_selected_slot] Direct date match found for slot {i+1}")
                return {
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"]
                }
        elif "jul 14" in display or "july 14" in display:
            if any(pattern in user_lower for pattern in ["july 14", "jul 14", "14th", "july 14th"]):
                print(f"[DEBUG detect_selected_slot] Direct date match found for slot {i+1}")
                return {
                    "start_time": slot["start_time"],
                    "end_time": slot["end_time"]
                }
    
    # Use LLM for more complex date/time inference
    slot_displays = [slot.get("display", "") for slot in slot_lines]
    
    prompt = f"""You are a slot selection assistant. The user wants to select one of the available appointment slots.

User input: "{user_input}"

Available slots:
{chr(10).join([f"{i+1}. {display}" for i, display in enumerate(slot_displays)])}

Instructions:
- Analyze the user's input to determine which slot they're referring to
- Look for date references (like "jul 7", "july 7", "7th", "today", etc.)
- Look for ordinal references (like "first", "second", "1st", "2nd", etc.)
- Consider context and natural language variations
- If the user mentions a specific date, match it to the corresponding slot
- Current date: {datetime.now().strftime('%A, %B %d, %Y')}

Examples:
- "july 7th" or "jul 7" should match slot with "Jul 07"
- "july 14" or "14th" should match slot with "Jul 14"
- "first" or "1st" should match slot 1
- "second" or "2nd" should match slot 2

Return ONLY a JSON object with the slot number (1-based index) or null if no clear match:
{{"selected_slot": 1}} or {{"selected_slot": null}}

Do not include any explanation, just the JSON."""

    try:
        response = llm_basic.invoke([HumanMessage(content=prompt)])
        response_content = response.content.strip()
        print(f"[DEBUG detect_selected_slot] LLM response: {response_content}")
        
        # Clean up the response in case there are extra characters
        if response_content.startswith('```json'):
            response_content = response_content[7:-3].strip()
        elif response_content.startswith('```'):
            response_content = response_content[3:-3].strip()
        
        result = json.loads(response_content)
        
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
        print(f"[DEBUG detect_selected_slot] Raw response: {response.content if 'response' in locals() else 'No response'}")

    print("[DEBUG detect_selected_slot] No slot matched.")
    return {}