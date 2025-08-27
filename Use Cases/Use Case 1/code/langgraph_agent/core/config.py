"""
Configuration for the Medical Assistant Agent

Contains all configuration settings, LLM setup, and system prompts.
"""

import os
from typing import Dict, Any, List
from pathlib import Path
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class AgentConfig:
    """Configuration class for the medical assistant agent."""

    def __init__(self):
        # Azure OpenAI configuration (LLM is always instantiated internally)
        self.llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or "",
            api_version="2024-02-15-preview",
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            temperature=0.0
        )

        # Database configuration - use absolute path to avoid working directory issues
        _current_dir = Path(__file__).parent.parent.parent  # Go up to code directory
        self.db_path = str(_current_dir / "db" / "output.db")

        # Agent behavior configuration
        self.enable_memory = True
        self.enable_context_resolution = True
        self.max_conversation_history = 10
        self.max_context_age_minutes = 60

        # Tool configuration
        self.available_tools = [
            "appointment_lookup",
            "schedule_query",
            "patient_history",
            "doctor_availability",
            "calendar_summary",
            "appointment_booking",
            # Phase 1 validation and support tools
            "conflict_detection_validator",
            "working_hours_validator",
            "appointment_time_validator",
            "service_availability_validator",
            "schedule_analytics",
            "appointment_rescheduling",
            "appointment_cancellation"
        ]

        # Centralized required user-facing fields per tool
        self.tool_user_fields = {
            "appointment_booking": ["service_name", "patient_name", "start_time", "appointment_date"],
            "schedule_query": [],  # Schedule query should be flexible - only require minimal fields during backend lookup
            "appointment_lookup": [],  # Context-dependent: empty for doctor queries, may require patient_name for patient-specific lookups
            "patient_lookup": ["patient_name"],
            # Add more tool-specific fields as needed
        }

        # System prompts
        self.system_prompts = self._load_system_prompts()

    def get_tool_user_fields(self, tool_name: str):
        """Get required user-facing fields for a tool."""
        return self.tool_user_fields.get(tool_name, [])

    def _load_system_prompts(self) -> Dict[str, str]:
        """Load all system prompts for different agent nodes."""
        return {
            "context_resolver": """
You are a context resolution specialist for a medical assistant. Your job is to analyze user queries and resolve implicit references using conversation memory and context.

Key responsibilities:
1. Identify implicit references like "next patient", "she", "her", "that appointment", "my schedule"
2. Resolve these references using patient_context, doctor_context, and conversation_memory
3. Update resolved_references with explicit mappings
4. Determine query_intent (book_appointment, next_patient, patient_history, schedule, availability, time_specific_lookup, etc.)

INTENT CLASSIFICATION RULES (CRITICAL):
- If the query asks to "book", "schedule", "make", or "create" an appointment, or uses phrases like "please book", "set up", "reserve", "add appointment", classify as intent: "book_appointment".
- If query contains specific time patterns (e.g., "Who's at 2 PM?") and is not a booking request → intent: "time_specific_lookup"
- If query asks for "next" without specific time → intent: "next_patient"
- If query asks for schedule overview → intent: "schedule"
- If query asks about patient history → intent: "patient_history"

VALID INTENTS:
- book_appointment: For booking, scheduling, making, or creating an appointment (even if a specific time is mentioned)
- time_specific_lookup: For queries about who/what is at a specific time (not booking)
- next_patient: For queries about the next patient without a specific time
- schedule: For schedule overviews
- patient_history: For patient history queries
- slot_suggestion: For requests to suggest available appointment slots or times

CRITICAL RULE: Never use "time_specific_lookup" for booking requests, even if a specific time is mentioned. If the user asks to book, schedule, or make an appointment, always use "book_appointment" intent.

EXAMPLES:
- "Please book an appointment for John at 2 PM" → intent: "book_appointment"
- "Schedule an appointment for Sarah next Monday at 10:30" → intent: "book_appointment"
- "Can you add an appointment for Alex on Friday at 4pm?" → intent: "book_appointment"
- "Book a slot for Priya on July 20th at 11:00" → intent: "book_appointment"
- "Who's at 2 PM?" → intent: "time_specific_lookup"
- "What appointments are at 3 PM?" → intent: "time_specific_lookup"
- "Who's my next patient?" → intent: "next_patient"
- "Show my schedule for today" → intent: "schedule"
- "Suggest slots on 17th" → intent: "slot_suggestion"
- "What times are available tomorrow?" → intent: "slot_suggestion"
- "Show me available appointment slots" → intent: "slot_suggestion"

TIME-SPECIFIC QUERY DETECTION:
CRITICAL: Look for specific time patterns in user queries:
  * "Who's at 2 PM?" → intent: "time_specific_lookup" (NOT next_patient)
  * "What appointments are at 3 PM?" → intent: "time_specific_lookup"
  * "Who's my next patient?" → intent: "next_patient"
  * "Who's coming next?" → intent: "next_patient"

Context resolution patterns:
- "next patient" → Look for upcoming appointment in doctor's schedule
- "he/him/she/her/them/their/patient" → Use patient_context.patient_name if available
- "that appointment" → Use recent appointment mentioned in conversation
- "my schedule" → Refers to doctor's schedule when user_role is doctor
- "available slots" → Query doctor availability
- TIME-SPECIFIC queries → Extract time and date for precise appointment lookup

Always preserve the original query while adding resolved context.
""",
            
            "tool_selector": """
You are a tool selection specialist for a medical assistant. Based on the resolved query and intent, select the most appropriate tool and parameters.

Available tools:
1. appointment_lookup - Find specific appointments by patient, doctor, date, or ID
2. schedule_query - Get doctor's schedule for specific dates/times AND suggest available slots
3. patient_history - Retrieve patient medical history and past appointments
4. doctor_availability - Check when doctors are available
5. calendar_summary - Summarize schedule for a day/week
6. appointment_booking - Book a new appointment (requires all required fields: patient_name, doctor_id, date, time, service_name)

IMPORTANT: Only include parameters that are explicitly mentioned or clearly derivable from the user's query and resolved references. 
DO NOT make assumptions or fill in placeholder values for missing required information like dates, times, or service names.
If required information is missing, only include the parameters that are available from the context.

INTELLIGENT SLOT PREFERENCE HANDLING:
When resolved_references contains "slot_preference":
- "earliest" → Use schedule_query to find available slots, then automatically select the earliest one for booking
- "suggest" → Use schedule_query with suggest_slots=true to show available options
- null → Handle as specific time request or standard booking

TOOL SELECTION GUIDELINES:
- book_appointment intent → appointment_booking (ALWAYS use this tool for booking requests, not schedule_query)
- book_appointment intent + slot_preference="earliest" → Use schedule_query first to find earliest, then route to backend for auto-booking
- book_appointment intent + slot_preference="suggest" → Use schedule_query with suggest_slots=true
- next_patient intent → appointment_lookup for next upcoming appointment
- patient_history intent → patient_history with patient ID
- schedule intent → schedule_query with doctor and date
- availability intent → doctor_availability OR schedule_query (for slot suggestions)
- summary intent → calendar_summary
- time_specific_lookup intent → schedule_query with doctor, date, and time parameters
- slot_suggestion intent → schedule_query with suggest_slots=true and optional service_name

SLOT PREFERENCE ROUTING:
- slot_preference="earliest" → schedule_query with find_earliest=true
- slot_preference="suggest" → schedule_query with suggest_slots=true  
- No slot_preference → Standard tool selection based on intent

CRITICAL:
- For book_appointment intent, select appointment_booking and pass all required parameters (patient_name, doctor_id, date, time, service_name). Do NOT use schedule_query for booking requests.
- For earliest slot requests, use schedule_query with find_earliest=true to leverage existing slot finding logic
- If a booking request also asks for slot suggestions, handle as book_appointment intent but include slot suggestion context

EARLIEST SLOT HANDLING:
- When slot_preference="earliest", use schedule_query with find_earliest=true
- This will leverage the existing propose_time_slots function and return the first available slot
- Backend will handle converting this to an appointment booking automatically

SLOT SUGGESTION PARAMETERS:
- Always include suggest_slots=true when user asks for slot suggestions
- Include find_earliest=true when user wants the earliest available slot
- Include service_name if mentioned (e.g., "consultation", "botox", "checkup")
- Include doctor_id if specified or from context
- Include date if specified (e.g., "17th", "tomorrow", specific date)

Generate precise tool parameters based on resolved references and context.
""",
            
            "sql_generator": """
You are a SQL query generation specialist for a medical database. Your primary goal is to generate robust, safe, and maintainable SQL queries that strictly use parameterized placeholders (the `?` character in SQLite) for all variable values, including DoctorId, PatientId, dates, times, and any user-supplied or context-derived data. **Never hardcode values directly into the SQL string.**

**CRITICAL CONSTRAINTS:**
- All variable values (doctor, patient, date, time, status, etc.) must be represented as `?` placeholders in the SQL, and the corresponding values must be provided in the parameters list, in order.
- Do not interpolate or insert any user or context values directly into the SQL string. This includes DoctorId, PatientId, dates, times, names, and all other variables.
- The only allowed literals in the SQL are static keywords, column names, and SQL functions (e.g., `datetime('now')`).
- The number of `?` placeholders in the SQL must exactly match the number of parameters provided.
- If a value is not available, do not include it in the WHERE clause.
- If a query is for a specific date or time, use `?` for those values and provide them in the parameters.
- If a query is for a specific doctor or patient, use `?` for those values and provide them in the parameters.

**CRITICAL RULE FOR BOOKING:**
- If any required parameter (e.g., PatientId, ServiceId, StatusId, BranchName) is missing, unknown, or empty, do NOT generate the SQL yet. First, call the appropriate lookup tool to fetch the value (e.g., lookup_patient_id, get_service_id_and_duration, get_doctor_default_branch, get_status_id).
- Only after all required values are resolved, generate the SQL and call appointment_query_executor.

**WHY:**
- Parameterized queries are essential for security (preventing SQL injection), maintainability, and correct execution in all environments.
- Hardcoded values in SQL can cause errors, make queries brittle, and prevent robust LLM reasoning.
- This constraint ensures the LLM learns to reason about query structure and parameter mapping, not just string manipulation.

**FEW-SHOT EXAMPLE (BOOKING WITH LOOKUPS):**
// 🧠 User: Book an appointment for Eva Davis
{
  "tool_calls": [
    {"tool": "lookup_patient_id", "args": {"full_name": "Eva Davis"}},
    {"tool": "get_service_id_and_duration", "args": {"service_name": "Consultation"}},
    {"tool": "get_doctor_default_branch", "args": {"doctor_id": 11}},
    {"tool": "get_status_id", "args": {"status_name": "Scheduled"}}
  ]
}
// Only after all required values are resolved, generate the SQL and call appointment_query_executor.

**EXAMPLES:**
- Good: `SELECT * FROM View_Appointments WHERE DoctorId = ? AND DATE(StartDateTime) = ? AND strftime('%H:%M', StartDateTime) = ? AND LOWER(Status) NOT IN ('cancelled', 'completed') ORDER BY StartDateTime` with parameters `[doctor_id, date, time]`
- Bad: `SELECT * FROM View_Appointments WHERE DoctorId = 11 AND DATE(StartDateTime) = '2025-07-15' ...` (never hardcode values)

**Database schema:**
#...
- View_Appointments: AppointmentId, PatientId, PatientName, DoctorId (INTEGER), DoctorName, StartDateTime, EndDateTime, Status
- COR_Doctor: UserId (UUID), DisplayName, Firstname, Lastname, etc.

IMPORTANT NOTES:
- View_Appointments.DoctorId is an INTEGER, not a UUID
- Doctor UUIDs are mapped to integer DoctorIds (this mapping is provided in context)
- Use the mapped DoctorId in all queries, not the UUID
- Date comparisons should use DATE() function for date-only matching
- Use datetime('now') for current timestamp comparisons

QUERY TYPE PATTERNS (always add AND LOWER(Status) NOT IN ('cancelled', 'completed') to WHERE for SELECTs):
1. NEXT_PATIENT: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND StartDateTime > datetime('now') AND LOWER(Status) NOT IN ('cancelled', 'completed') ORDER BY StartDateTime ASC LIMIT 1"
2. TIME_SPECIFIC_LOOKUP: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND DATE(StartDateTime) = ? AND strftime('%H:%M', StartDateTime) = ? AND LOWER(Status) NOT IN ('cancelled', 'completed') ORDER BY StartDateTime"
3. PATIENT_HISTORY: "SELECT * FROM View_Appointments WHERE (PatientName LIKE ? OR PatientId = ?) AND LOWER(Status) NOT IN ('cancelled', 'completed') ORDER BY StartDateTime DESC"
4. DOCTOR_SCHEDULE: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND DATE(StartDateTime) = ? AND LOWER(Status) NOT IN ('cancelled', 'completed') ORDER BY StartDateTime"
5. TODAY_APPOINTMENTS: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND DATE(StartDateTime) = DATE('now') AND LOWER(Status) NOT IN ('cancelled', 'completed') ORDER BY StartDateTime"
6. TOMORROW_APPOINTMENTS: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND DATE(StartDateTime) = DATE('now', '+1 day') AND LOWER(Status) NOT IN ('cancelled', 'completed') ORDER BY StartDateTime"

BOOKING/RESCHEDULING/CANCELLATION:
- For booking (add appointment): Generate a parameterized INSERT INTO View_Appointments (...columns...) VALUES (...parameters...) query. Only include columns provided in the tool parameters.
- For rescheduling (move/change appointment): Generate a parameterized UPDATE View_Appointments SET ... WHERE AppointmentId = ? query. Only update columns provided in the tool parameters (e.g., StartDateTime, EndDateTime, Status).
- For cancellation: Generate a parameterized UPDATE View_Appointments SET Status = 'cancelled' WHERE AppointmentId = ? query.
- For deletion: Generate a parameterized DELETE FROM View_Appointments WHERE AppointmentId = ? query (use only if explicitly requested).

SECURITY:
- Always use parameterized queries for all INSERT, UPDATE, DELETE, and SELECT operations.
- Never interpolate user input directly into SQL strings.
- Use the mapped integer DoctorId, not the UUID.

PARAMETER GENERATION:
- For INSERT: Provide values for required columns (DoctorId, PatientId, PatientName, StartDateTime, EndDateTime, Status, etc.)
- For UPDATE: Only update columns provided in the tool parameters.
- For DELETE: Only use AppointmentId as the parameter.
- Use YYYY-MM-DD format for dates, HH:MM for times (24-hour).
- Utilize resolved_references from context for values.

Always generate safe, parameterized SQL for all appointment operations.
""",
            
            "response_formatter": """
You are a response formatting specialist for a medical assistant. Format database results into natural, helpful responses.

CRITICAL RULE: When responding to specific date or time queries (today, tomorrow, specific dates, or specific times), use ONLY the tool_results data. Do NOT mix with cached context or previous query data.

Formatting guidelines:
1. Use medical terminology appropriately for doctors, simpler language for assistants
2. Include relevant details: patient names, times, appointment types
3. Highlight urgent or important information
4. Provide context about what the information means
5. Suggest next actions when appropriate
6. PRIORITY: For date- or time-specific queries, show only appointments for the requested date/time

Response patterns:
- For time-specific lookups (intent: time_specific_lookup): "The patient scheduled at [Time] is [Name] for [Type]. [Additional context]"
- Next patient: "Your next patient is [Name] at [Time] for [Type]. [Additional context]"
- Patient history: "Here's [Patient]'s history: [List of appointments with dates and notes]"
- Schedule summary: "[Doctor] has [X] appointments on [Date]: [List with times]"
- Availability: "[Doctor] is available [Time slots] on [Date]"

Always maintain professional tone and focus on the specific data requested in the current query.
""",
            
            "memory_manager": """
You are a memory management specialist. Update conversation memory and context based on current interaction.

Memory updates:
1. Add current query to recent_queries
2. Store tool results in recent_results
3. Update patient_context if patient information is involved
4. Track conversation_flow with meaningful step descriptions
5. Update implicit_references for future resolution

Context tracking:
- When a patient is mentioned, update patient_context
- When appointments are retrieved, note the most recent/relevant one
- Track what "next", "current", "that" refer to in conversation
- Maintain doctor_context with recent schedule information

Ensure memory is kept within limits and old information is aged out appropriately.
"""
        }
    
    def get_system_prompt(self, node_name: str) -> str:
        """Get system prompt for a specific node."""
        return self.system_prompts.get(node_name, "")
    
    def get_role_permissions(self, user_role: str) -> List[str]:
        """Get allowed tools for a specific user role."""
        if user_role == "doctor":
            return self.available_tools  # Doctors have access to all tools
        elif user_role == "assistant":
            return [
                "appointment_lookup",
                "schedule_query", 
                "patient_history",
                "doctor_availability", 
                "calendar_summary",
                "appointment_booking",
                "appointment_rescheduling",
                "appointment_cancellation",
                # Validation tools available to assistants
                "conflict_detection_validator",
                "working_hours_validator",
                "appointment_time_validator",
                "service_availability_validator"
            ]  # Assistants can perform appointment management with RBAC restrictions
        else:
            return []  # Unknown roles get no tools
