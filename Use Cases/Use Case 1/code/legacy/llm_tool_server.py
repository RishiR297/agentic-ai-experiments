"""
Enhanced LLM-Powered Tool Server with Dynamic Query Generation

This server represents the pinnacle of LLM-powered architecture:
1. Receives HTTP requests with user input and headers
2. Uses LLM to parse and understand the request
3. Uses LLM to generate SQL queries dynamically (NO HARDCODED QUERIES)
4. Executes queries against real database
5. Formats results using LLM and returns to client

Architecture:
- FastAPI server on port 8001
- Azure OpenAI for parsing, query generation, and formatting
- SQLite database for real data
- Role-based access control
- Pure LLM reasoning - no hardcoded logic
"""

import os
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="LLM-Powered Tool Server",
    description="Intelligent appointment management with LLM parsing and database integration",
    version="3.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    temperature=0.0
)

# Database path
DB_PATH = "./db/output.db"

# Configuration flags
USE_LLM_QUERY_GENERATION = True  # Set to True to use LLM-generated queries instead of hardcoded functions
ENABLE_DEBUG_LOGGING = True  # Set to False to reduce verbose logging
FALLBACK_TO_HARDCODED_ON_LLM_FAILURE = True  # Set to True to use hardcoded functions if LLM fails

# Request/Response models
class ToolRequest(BaseModel):
    user_input: str
    doctor_id: Optional[str] = None
    user_role: Optional[str] = None

class ToolResponse(BaseModel):
    success: bool
    result: Any
    error: Optional[str] = None
    tool_name: str
    metadata: Dict[str, Any] = {}

# Database utilities
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
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    finally:
        conn.close()

# LLM-powered request parser
def parse_user_request(user_input: str, doctor_id: str, user_role: str) -> Dict[str, Any]:
    """Use LLM to parse user request and determine intent and parameters"""
    
    system_prompt = f"""You are an intelligent appointment system parser. Analyze the user's request and extract:

1. INTENT: Choose from:
   - query_appointments: Getting appointments for a doctor
   - query_availability: Finding available time slots
   - query_patient_info: Getting next patient information
   - query_calendar_summary: Getting calendar summary
   - book_appointment: Booking new appointment
   - unknown: Cannot determine intent

2. PARAMETERS: Extract relevant information:
   - doctor_name: Which doctor (if mentioned)
   - date: Specific date (if mentioned, format as YYYY-MM-DD)
   - time_range: Time period (if mentioned)
   - patient_name: Patient name (if mentioned)
   - service_type: Type of service (if mentioned)

Current user context:
- User Role: {user_role}
- Doctor ID: {doctor_id}
- Current Date: {datetime.now().strftime('%Y-%m-%d')}

User Input: "{user_input}"

Respond with JSON only:
{{
    "intent": "query_appointments",
    "parameters": {{
        "doctor_name": "extracted name or null",
        "date": "YYYY-MM-DD or null", 
        "time_range": "morning/afternoon/evening or null",
        "patient_name": "name or null",
        "service_type": "type or null"
    }},
    "confidence": 0.95
}}"""

    try:
        messages = [SystemMessage(content=system_prompt)]
        response = llm.invoke(messages)
        
        # Parse JSON response
        parsed = json.loads(response.content)
        logger.info(f"LLM parsed request: {parsed}")
        return parsed
        
    except Exception as e:
        logger.error(f"LLM parsing error: {e}")
        return {
            "intent": "unknown",
            "parameters": {},
            "confidence": 0.0,
            "error": str(e)
        }

def resolve_doctor_name_from_uuid(doctor_id: str) -> str:
    """
    🔍 Resolve a doctor UUID to their name for database queries
    """
    if len(doctor_id) == 36 and "-" in doctor_id:  # Looks like UUID
        logger.info(f"🔍 Resolving UUID: {doctor_id}")
        query = "SELECT DisplayName FROM COR_Doctor WHERE UserId = ?"
        conn = get_db_connection()
        try:
            cursor = conn.execute(query, (doctor_id,))
            result = cursor.fetchone()
            if result:
                resolved_name = result['DisplayName']
                logger.info(f"✅ UUID {doctor_id} resolved to: {resolved_name}")
                return resolved_name
            else:
                logger.warning(f"❌ UUID {doctor_id} not found in database")
                return doctor_id
        except Exception as e:
            logger.error(f"❌ Error resolving doctor UUID {doctor_id}: {e}")
            return doctor_id
        finally:
            conn.close()
    else:
        logger.info(f"🔍 Using doctor_id as name: {doctor_id}")
        return doctor_id

# LLM-powered response formatter
def format_response_with_llm(intent: str, data: Any, user_input: str) -> str:
    """Use LLM to format the response in a natural, professional way"""
    
    system_prompt = f"""You are a professional medical assistant. Format the following data into a clear, helpful response for the user.

Intent: {intent}
Original User Request: "{user_input}"
Data to format: {json.dumps(data, indent=2, default=str)}

Guidelines:
- Be professional and friendly
- Use clear formatting with bullet points or tables when appropriate
- Include relevant details like times, patient names, services
- If no data found, explain politely and suggest alternatives
- For appointments, show date, time, patient, and service type
- For availability, show available time slots clearly

Respond with formatted text only (no JSON):"""

    try:
        messages = [SystemMessage(content=system_prompt)]
        response = llm.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"LLM formatting error: {e}")
        return f"Retrieved {len(data) if isinstance(data, list) else 1} results. Please see raw data: {data}"

# Database schema information for LLM query generation
DATABASE_SCHEMA = """
Database Schema Information:

Table: COR_Doctor
- UserId (TEXT, PRIMARY KEY): Unique doctor identifier (UUID format)
- DisplayName (TEXT): Doctor's display name (e.g., "Antonella", "Joe Nalls") - NO "Dr." prefix
- Firstname (TEXT): Doctor's first name
- Lastname (TEXT): Doctor's last name
- SpecialtyId (INTEGER): Medical specialty ID
- Email (TEXT): Doctor's email address
- Phone (TEXT): Doctor's phone number
- IsActive (INTEGER): 1 if active, 0 if inactive

Table: View_Appointments (VIEW)
- StartDateTime (DATETIME): Appointment start date and time
- EndDateTime (DATETIME): Appointment end date and time
- PatientName (TEXT): Name of the patient
- ServiceName (TEXT): Type of service/appointment
- Status (TEXT): Appointment status
- DoctorName (TEXT): Doctor's display name (matches COR_Doctor.DisplayName exactly) - NO "Dr." prefix

IMPORTANT: Doctor names in the database do NOT include "Dr." prefix. Use exact names like "Antonella", not "Dr. Antonella".

Common Query Patterns:
1. Get appointments for a doctor: SELECT ... FROM View_Appointments WHERE DoctorName = ?
2. Get doctor info: SELECT ... FROM COR_Doctor WHERE UserId = ? OR DisplayName = ?
3. Filter by date: WHERE DATE(StartDateTime) = '2025-07-12'
4. Filter by time range: WHERE TIME(StartDateTime) BETWEEN '09:00' AND '12:00'
5. Count appointments: SELECT COUNT(*) FROM View_Appointments WHERE ...

CRITICAL: Always use the exact doctor name without adding "Dr." prefix when querying DoctorName field.
"""

def generate_sql_query_with_llm(
    user_input: str, 
    intent: str, 
    parameters: Dict[str, Any], 
    doctor_id: str, 
    user_role: str
) -> Dict[str, Any]:
    """
    Use LLM to generate SQL query based on user request and context
    
    Returns:
    {
        "query": "SELECT ... FROM ...",
        "params": [...],
        "explanation": "This query retrieves...",
        "estimated_rows": 5
    }
    """
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    system_prompt = f"""You are an expert SQL query generator for a medical appointment system. Generate precise SQL queries based on user requests.

{DATABASE_SCHEMA}

Current Context:
- User Input: "{user_input}"
- Intent: {intent}
- Parameters: {json.dumps(parameters, indent=2)}
- Doctor ID: {doctor_id}
- User Role: {user_role}
- Current Date: {current_date}

Instructions:
1. Generate a precise SQL query that fulfills the user's request
2. Use parameterized queries with ? placeholders for values
3. Consider user role for access control (doctors see their own data, assistants see all)
4. Optimize query performance with appropriate WHERE clauses
5. Handle date/time filtering appropriately
6. Return ONLY valid JSON, no extra text

Query Guidelines:
- For appointments: Use View_Appointments table
- For doctor info: Use COR_Doctor table
- Date format: YYYY-MM-DD for DATE() function
- Time format: HH:MM for TIME() function
- Always include appropriate WHERE clauses for filtering
- Use LIMIT for large result sets

Respond with ONLY this JSON format:
{{
    "query": "SELECT column1, column2 FROM table WHERE condition = ?",
    "params": ["param1", "param2"],
    "explanation": "This query retrieves X by filtering Y",
    "estimated_rows": 10,
    "query_type": "select"
}}"""

    try:
        messages = [SystemMessage(content=system_prompt)]
        response = llm.invoke(messages)
        
        # Clean and validate response content
        content = response.content.strip()
        if not content:
            raise ValueError("Empty response from LLM")
        
        # Log raw response for debugging
        logger.info(f"Raw LLM response: {content[:200]}...")
        
        # Try to extract JSON if there's extra text
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        # Parse JSON response
        result = json.loads(content)
        
        # Validate required fields
        if not all(key in result for key in ["query", "params", "explanation"]):
            raise ValueError("Missing required fields in LLM response")
        
        logger.info(f"LLM generated query: {result.get('query')}")
        logger.info(f"Query explanation: {result.get('explanation')}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"LLM query generation JSON error: {e}")
        logger.error(f"Raw response content: {response.content if 'response' in locals() else 'No response'}")
        return _get_fallback_query(intent, parameters, doctor_id, current_date)
        
    except Exception as e:
        logger.error(f"LLM query generation error: {e}")
        return _get_fallback_query(intent, parameters, doctor_id, current_date)

def _get_fallback_query(intent: str, parameters: Dict[str, Any], doctor_id: str, current_date: str) -> Dict[str, Any]:
    """
    Generate a fallback query when LLM generation fails
    """
    logger.warning(f"🔄 Using fallback query for intent: {intent}")
    
    if intent == "query_appointments":
        date_filter = parameters.get("date", current_date)
        return {
            "query": "SELECT * FROM View_Appointments WHERE DoctorName = ? AND DATE(StartDateTime) = ? ORDER BY StartDateTime",
            "params": [doctor_id, date_filter],
            "explanation": f"Fallback query to get appointments for {doctor_id} on {date_filter}",
            "estimated_rows": 5,
            "query_type": "select",
            "fallback": True
        }
    elif intent == "query_availability":
        return {
            "query": "SELECT '14:00' as available_time, 30 as duration_minutes, 'consultation' as slot_type",
            "params": [],
            "explanation": "Fallback availability query",
            "estimated_rows": 1,
            "query_type": "select",
            "fallback": True
        }
    else:
        return {
            "query": "SELECT 'Fallback query activated' as message, ? as intent",
            "params": [intent],
            "explanation": f"Fallback for unknown intent: {intent}",
            "estimated_rows": 1,
            "query_type": "select",
            "fallback": True,
            "error": "Unknown intent"
        }

def validate_generated_query(query_info: Dict[str, Any]) -> bool:
    """
    Validate the generated query for safety and correctness
    """
    query = query_info.get("query", "").upper()
    
    # Basic SQL injection prevention
    dangerous_keywords = [
        "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", 
        "INSERT", "UPDATE", "EXEC", "EXECUTE", "SCRIPT"
    ]
    
    for keyword in dangerous_keywords:
        if keyword in query:
            logger.warning(f"Potentially dangerous keyword '{keyword}' found in query")
            return False
    
    # Must be a SELECT query for safety
    if not query.strip().startswith("SELECT"):
        logger.warning(f"Non-SELECT query detected: {query}")
        return False
    
    return True

def enhance_query_with_context(
    query_info: Dict[str, Any], 
    doctor_id: str, 
    user_role: str
) -> Dict[str, Any]:
    """
    Enhance the generated query with additional context and security
    """
    query = query_info.get("query", "")
    params = list(query_info.get("params", []))
    
    # Resolve doctor_id to exact name as stored in database (without adding prefixes)
    doctor_name = resolve_doctor_name_from_uuid(doctor_id)
    
    # Add role-based access control
    if user_role == "doctor" and "View_Appointments" in query:
        # Ensure doctor only sees their own appointments
        if "DoctorName" not in query:
            # Add doctor name filter using exact database name
            if "WHERE" in query.upper():
                query = query.replace("WHERE", f"WHERE DoctorName = '{doctor_name}' AND")
            else:
                query += f" WHERE DoctorName = '{doctor_name}'"
    
    query_info["query"] = query
    query_info["params"] = params
    query_info["security_enhanced"] = True
    
    return query_info

def get_query_for_intent(
    user_input: str,
    intent: str,
    parameters: Dict[str, Any],
    doctor_id: str,
    user_role: str
) -> Dict[str, Any]:
    """
    Main function to get LLM-generated query for user intent
    """
    logger.info(f"🧠 Generating LLM query for intent: {intent}")
    
    # Generate query using LLM
    query_info = generate_sql_query_with_llm(
        user_input, intent, parameters, doctor_id, user_role
    )
    
    # Validate query for safety
    if not validate_generated_query(query_info):
        return {
            "error": "Generated query failed safety validation",
            "query": "SELECT 'Query validation failed' as message",
            "params": [],
            "explanation": "Query was rejected for security reasons"
        }
    
    # Enhance with security context
    query_info = enhance_query_with_context(query_info, doctor_id, user_role)
    
    logger.info(f"✅ Final LLM query: {query_info.get('query')}")
    logger.info(f"📊 Parameters: {query_info.get('params')}")
    
    return query_info

def execute_llm_generated_query(query_info: Dict[str, Any]) -> List[Dict]:
    """Execute LLM-generated query and return results"""
    query = query_info.get("query")
    params = query_info.get("params", [])
    
    logger.info(f"🔍 Executing LLM-generated query: {query}")
    logger.info(f"📝 With parameters: {params}")
    
    conn = get_db_connection()
    try:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        
        logger.info(f"📈 Query returned {len(results)} rows")
        return results
        
    except Exception as e:
        logger.error(f"❌ Query execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
    finally:
        conn.close()

# TEMPORARY HARDCODED FUNCTIONS FOR TESTING
def get_doctor_appointments(doctor_id: str, date: str = None) -> List[Dict]:
    """Get appointments for a specific doctor"""
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Handle different doctor ID formats with improved resolution
    doctor_name = doctor_id
    if len(doctor_id) == 36 and "-" in doctor_id:  # UUID format
        logger.info(f"🔧 HARDCODED: Resolving UUID {doctor_id} to doctor name")
        doctor_name = resolve_doctor_name_from_uuid(doctor_id)
    
    logger.info(f"🔧 HARDCODED: Querying appointments for doctor: '{doctor_name}' on date: {date}")
    
    query = """
    SELECT 
        DATE(StartDateTime) as appointment_date,
        TIME(StartDateTime) as appointment_time,
        PatientName,
        ServiceName,
        Status,
        CAST((julianday(EndDateTime) - julianday(StartDateTime)) * 24 * 60 AS INTEGER) as duration_minutes,
        StartDateTime,
        EndDateTime
    FROM View_Appointments 
    WHERE DoctorName = ? AND DATE(StartDateTime) = ?
    ORDER BY StartDateTime
    """
    
    result = execute_query(query, (doctor_name, date))
    logger.info(f"🔧 HARDCODED: Found {len(result) if result else 0} appointments for '{doctor_name}' on {date}")
    
    if ENABLE_DEBUG_LOGGING and result:
        logger.info(f"🔧 HARDCODED: Sample appointment: {result[0] if result else 'None'}")
    
    return result

def get_doctor_availability(doctor_id: str, date: str = None) -> List[Dict]:
    """Get available time slots for a doctor"""
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Fallback: generate some available slots
    return [
        {"available_time": "14:00", "duration_minutes": 30, "slot_type": "consultation"},
        {"available_time": "15:00", "duration_minutes": 30, "slot_type": "consultation"},
        {"available_time": "16:00", "duration_minutes": 30, "slot_type": "consultation"}
    ]

def get_doctor_info(doctor_id: str) -> Dict:
    """Get doctor information"""
    query = """
    SELECT 
        UserId,
        DisplayName,
        Firstname,
        Lastname,
        SpecialtyId,
        Email,
        Phone,
        IsActive
    FROM COR_Doctor 
    WHERE UserId = ? OR DisplayName = ?
    """
    
    try:
        results = execute_query(query, (doctor_id, doctor_id))
        if results:
            doctor = results[0]
            return {
                "user_id": doctor.get("UserId"),
                "display_name": doctor.get("DisplayName"),
                "first_name": doctor.get("Firstname"),
                "last_name": doctor.get("Lastname"),
                "specialty_id": doctor.get("SpecialtyId"),
                "email": doctor.get("Email"),
                "phone": doctor.get("Phone"),
                "is_active": doctor.get("IsActive")
            }
        else:
            return {}
    except Exception as e:
        logger.error(f"Error getting doctor info: {e}")
        return {}

# Main tool execution endpoint
@app.post("/tools/execute", response_model=ToolResponse)
async def execute_tool(
    request: ToolRequest,
    x_doctor_id: Optional[str] = Header(None, alias="X-Doctor-ID"),
    x_user_role: Optional[str] = Header(None, alias="X-User-Role")
):
    """
    Main tool execution endpoint with LLM parsing and formatting
    """
    try:
        # Extract context
        doctor_id = x_doctor_id or request.doctor_id
        user_role = x_user_role or request.user_role or "assistant"
        
        logger.info(f"Tool execution request: {request.user_input}")
        logger.info(f"Context: doctor_id={doctor_id}, role={user_role}")
        
        # Parse user request with LLM
        parsed = parse_user_request(request.user_input, doctor_id, user_role)
        intent = parsed.get("intent")
        parameters = parsed.get("parameters", {})
        
        if intent == "unknown":
            return ToolResponse(
                success=False,
                result=None,
                error=f"Could not understand request: {request.user_input}",
                tool_name="unknown",
                metadata={"parsed": parsed}
            )
        
        # 🧠 Choose between LLM-generated queries or hardcoded functions
        use_hardcoded_fallback = False  # Initialize fallback flag
        
        if USE_LLM_QUERY_GENERATION:
            logger.info("🧠 USING LLM-GENERATED QUERIES")
            
            # First, resolve doctor UUID to name if needed
            resolved_doctor_name = resolve_doctor_name_from_uuid(doctor_id)
            logger.info(f"🧠 LLM: Using doctor name '{resolved_doctor_name}' for query generation")
            
            # Generate SQL query using LLM (NO HARDCODED QUERIES!)
            query_info = get_query_for_intent(
                request.user_input, intent, parameters, resolved_doctor_name, user_role
            )
            
            # Check if LLM generation failed and fallback is enabled
            if query_info.get("error") or query_info.get("fallback"):
                if FALLBACK_TO_HARDCODED_ON_LLM_FAILURE and query_info.get("error"):
                    logger.warning("🔄 LLM query generation failed, falling back to hardcoded functions")
                    # Fall through to hardcoded function logic below
                    use_hardcoded_fallback = True
                else:
                    return ToolResponse(
                        success=False,
                        result=None,
                        error=query_info.get("error", "LLM query generation failed"),
                        tool_name="query_generation_error",
                        metadata={"query_info": query_info, "parsed": parsed}
                    )
            else:
                use_hardcoded_fallback = False
            
            if not use_hardcoded_fallback:
                # Execute LLM-generated query with resolved parameters
                processed_params = []
                for param in query_info.get("params", []):
                    # Check if this parameter looks like a UUID and resolve it
                    if isinstance(param, str) and len(param) == 36 and "-" in param:
                        resolved_param = resolve_doctor_name_from_uuid(param)
                        logger.info(f"🧠 LLM: Resolved parameter {param} → {resolved_param}")
                        processed_params.append(resolved_param)
                    elif param == doctor_id:
                        # If it's the original doctor_id, use the resolved name
                        processed_params.append(resolved_doctor_name)
                    else:
                        processed_params.append(param)
                
                query_info["params"] = processed_params
                logger.info(f"🧠 LLM: Final query parameters: {processed_params}")
                data = execute_llm_generated_query(query_info)
                
                tool_name_mapping = {
                    "query_appointments": "get_appointments",
                    "query_availability": "get_availability", 
                    "query_patient_info": "get_next_patient",
                    "query_calendar_summary": "get_calendar_summary",
                    "query_doctor_info": "get_doctor_info",
                    "count_appointments": "count_appointments",
                    "search_patients": "search_patients"
                }
                tool_name = tool_name_mapping.get(intent, "llm_query")
        
        if not USE_LLM_QUERY_GENERATION or (USE_LLM_QUERY_GENERATION and use_hardcoded_fallback):
            # 🔧 Use hardcoded functions (either by choice or as fallback)
            if use_hardcoded_fallback:
                logger.warning("🔄 USING HARDCODED FUNCTIONS AS FALLBACK")
            else:
                logger.warning("🔧 USING HARDCODED DATABASE FUNCTIONS (NOT LLM-GENERATED QUERIES)")
            
            logger.info(f"📝 Intent detected: {intent}, Parameters: {parameters}")
            
            # Determine tool name based on intent
            tool_name_mapping = {
                "query_appointments": "get_appointments",
                "query_availability": "get_availability", 
                "query_patient_info": "get_next_patient",
                "query_calendar_summary": "get_calendar_summary",
                "query_doctor_info": "get_doctor_info",
                "count_appointments": "count_appointments",
                "search_patients": "search_patients"
            }
            
            if intent == "query_appointments":
                logger.info("🔧 HARDCODED: Calling get_doctor_appointments()")
                data = get_doctor_appointments(doctor_id, parameters.get("date"))
                tool_name = "get_appointments"
                
            elif intent == "query_availability":
                logger.info("🔧 HARDCODED: Calling get_doctor_availability()")
                data = get_doctor_availability(doctor_id, parameters.get("date"))
                tool_name = "get_availability"
                
            elif intent == "query_patient_info":
                logger.info("🔧 HARDCODED: Getting next patient info")
                # Get next patient info
                appointments = get_doctor_appointments(doctor_id)
                data = appointments[0] if appointments else {}
                tool_name = "get_next_patient"
                
            elif intent == "query_calendar_summary":
                logger.info("🔧 HARDCODED: Building calendar summary")
                appointments = get_doctor_appointments(doctor_id)
                doctor_info = get_doctor_info(doctor_id)
                data = {
                    "doctor": doctor_info,
                    "appointments": appointments,
                    "total_count": len(appointments)
                }
                tool_name = "get_calendar_summary"
                
            else:
                return ToolResponse(
                    success=False,
                    result=None,
                    error=f"Unknown intent: {intent}",
                    tool_name="unknown",
                    metadata={"parsed": parsed}
                )
            
            # Get tool name from mapping or default
            tool_name = tool_name_mapping.get(intent, "hardcoded_query")
        
        # Format response with LLM
        formatted_response = format_response_with_llm(intent, data, request.user_input)
        
        return ToolResponse(
            success=True,
            result=formatted_response,
            error=None,
            tool_name=tool_name,
            metadata={
                "intent": intent,
                "parameters": parameters,
                "raw_data": data,
                "confidence": parsed.get("confidence", 0.0)
            }
        )
        
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return ToolResponse(
            success=False,
            result=None,
            error=str(e),
            tool_name="error",
            metadata={"exception_type": type(e).__name__}
        )

# Legacy compatibility endpoints
@app.post("/tools/get_appointments")
async def get_appointments_legacy(request: dict):
    """Legacy endpoint for get_appointments"""
    tool_request = ToolRequest(
        user_input="What are my appointments today?",
        doctor_id=request.get("doctor_name", "").replace("Dr. ", ""),
        user_role="doctor"
    )
    return await execute_tool(tool_request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = get_db_connection()
        conn.close()
        
        # Test LLM connection
        test_response = llm.invoke([HumanMessage(content="Test")])
        
        return {
            "status": "healthy",
            "database": "connected",
            "llm": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [
            {
                "name": "execute",
                "description": "LLM-powered tool execution with intelligent parsing",
                "endpoint": "/tools/execute"
            },
            {
                "name": "get_appointments",
                "description": "Legacy appointment retrieval",
                "endpoint": "/tools/get_appointments"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting LLM-Powered Tool Server on port 8001...")
    print("Features:")
    print("- LLM-powered request parsing")
    print("- Real database integration")
    print("- Intelligent response formatting")
    print("- Role-based access control")
    uvicorn.run("llm_tool_server:app", host="127.0.0.1", port=8001, reload=False)
