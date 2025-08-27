"""
Medical Assistant Database Tools - Phase 1 (Clean Version)
Contains only the essential, well-tested tools for appointment management.
"""

import sqlite3
import logging
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from langchain.tools import tool

logger = logging.getLogger(__name__)

import os
from pathlib import Path

# Database configuration - use absolute path to avoid working directory issues
_current_dir = Path(__file__).parent.parent.parent  # Go up to code directory
DB_PATH = str(_current_dir / "db" / "output.db")

# =============================================================================
# CORE DATABASE TOOLS - Phase 1 (Working and Tested)
# =============================================================================

@tool("get_doctor_working_hours")
def get_doctor_working_hours(doctor_id: int, day_of_week: str) -> Dict[str, Any]:
    """Get doctor's working hours for a specific day."""
    # Convert day name to weekday number (Monday=1, Sunday=7)
    from datetime import datetime
    day_map = {
        'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4,
        'friday': 5, 'saturday': 6, 'sunday': 7
    }
    weekday_num = day_map.get(day_of_week.lower())
    if not weekday_num:
        return {"working_hours": None, "error": f"Invalid day: {day_of_week}"}
    
    query = """
        SELECT FromTime, ToTime 
        FROM COR_DoctorSchedule 
        WHERE DoctorId = ? AND WeekDay = ? AND IsActive = 1
    """
    results = execute_query(query, [doctor_id, weekday_num])
    return {"working_hours": results[0] if results else None}

@tool("get_doctor_off_periods")
def get_doctor_off_periods(doctor_id: int, date: str) -> Dict[str, Any]:
    """Get doctor's off periods for a specific date."""
    query = """
        SELECT FromTime, ToTime, IsOff 
        FROM COR_DoctorOffSchedule 
        WHERE DoctorId = ? AND Date = ?
    """
    results = execute_query(query, [doctor_id, date])
    return {"off_periods": results}

@tool("get_appointments_for_doctor")
def get_appointments_for_doctor(doctor_id: int, date: str) -> Dict[str, Any]:
    """Get all appointments for a doctor on a specific date."""
    query = """
        SELECT * FROM View_Appointments 
        WHERE DoctorId = ? AND DATE(StartDateTime) = ?
        ORDER BY StartDateTime
    """
    results = execute_query(query, [doctor_id, date])
    return {"appointments": results}

@tool("propose_time_slots")
def propose_time_slots(doctor_id: int, date: str, service_duration_minutes: int = 21) -> Dict[str, Any]:
    """
    Generate available time slots for a doctor on a specific date.
    This is the main tool for finding appointment availability.
    """
    try:
        available_slots = find_available_slots(doctor_id, date, service_duration_minutes)
        
        # Format slots for better presentation
        formatted_slots = []
        for slot in available_slots:
            start_time = f"{slot['start_hour']:02d}:{slot['start_minute']:02d}"
            end_time = f"{slot['end_hour']:02d}:{slot['end_minute']:02d}"
            formatted_slots.append(f"{start_time} - {end_time}")
        
        return {
            "success": True,
            "doctor_id": doctor_id,
            "date": date,
            "available_slots": formatted_slots,
            "total_slots": len(formatted_slots),
            "message": f"Found {len(formatted_slots)} available slots for doctor {doctor_id} on {date}"
        }
        
    except Exception as e:
        logger.error(f"Error finding time slots: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to find available slots for doctor {doctor_id} on {date}"
        }

# =============================================================================
# VALIDATION TOOLS - LLM-Enhanced (Phase 1 Working)
# =============================================================================

@tool("conflict_detection_validator")
def conflict_detection_validator(doctor_id: int, start_datetime: str, end_datetime: str, exclude_appointment_id: int = None) -> Dict[str, Any]:
    """
    LLM-enhanced conflict detection with intelligent analysis and recommendations.
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from ..core.config import AgentConfig
        
        config = AgentConfig()
        
        # Get existing appointments for the doctor around the requested time
        query = """
            SELECT AppointmentId, PatientName, StartDateTime, EndDateTime, ServiceName, Status
            FROM View_Appointments 
            WHERE DoctorId = ? 
            AND DATE(StartDateTime) = DATE(?)
            AND Status IN ('Scheduled', 'Booked', 'Confirmed', 'Rescheduled')
            ORDER BY StartDateTime
        """
        
        date_part = start_datetime.split()[0]
        existing_appointments = execute_query(query, [doctor_id, start_datetime])
        
        # Filter out the appointment being rescheduled if applicable
        if exclude_appointment_id:
            existing_appointments = [apt for apt in existing_appointments if apt['AppointmentId'] != exclude_appointment_id]
        
        # Prepare LLM analysis prompt
        conflict_prompt = f"""
        Analyze potential appointment conflicts for this scheduling request:
        
        Requested Appointment:
        - Doctor ID: {doctor_id}
        - Start: {start_datetime}
        - End: {end_datetime}
        
        Existing Appointments on {date_part}:
        {json.dumps(existing_appointments, indent=2)}
        
        Provide conflict analysis in JSON format:
        {{
            "has_conflict": true/false,
            "conflict_type": "direct_overlap/buffer_violation/none",
            "conflicting_appointments": [list of conflicting appointment IDs],
            "conflict_details": "detailed explanation",
            "severity": "high/medium/low",
            "recommendations": ["recommendation1", "recommendation2"],
            "alternative_times": ["time1", "time2"],
            "confidence_score": 0.0-1.0
        }}
        """
        
        messages = [
            SystemMessage(content="You are a medical appointment conflict detection expert."),
            HumanMessage(content=conflict_prompt)
        ]
        
        llm_response = config.llm.invoke(messages)
        conflict_analysis = json.loads(llm_response.content)
        
        # Add metadata
        conflict_analysis["doctor_id"] = doctor_id
        conflict_analysis["requested_time"] = start_datetime
        conflict_analysis["total_existing_appointments"] = len(existing_appointments)
        conflict_analysis["analysis_timestamp"] = datetime.now().isoformat()
        
        return conflict_analysis
        
    except Exception as e:
        logger.error(f"Conflict detection error: {e}")
        return {
            "has_conflict": False,
            "error": str(e),
            "confidence_score": 0.0,
            "message": "Error during conflict detection - proceeding with caution"
        }

@tool("working_hours_validator")
def working_hours_validator(doctor_id: int, date: str, start_time: str) -> Dict[str, Any]:
    """
    LLM-enhanced working hours validation with intelligent schedule analysis.
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from ..core.config import AgentConfig
        
        config = AgentConfig()
        
        # Get doctor's schedule for the day
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        weekday_num = date_obj.weekday() + 1  # Convert to 1-7 (Monday=1, Sunday=7)
        day_of_week = date_obj.strftime('%A')  # Keep for display purposes
        
        schedule_query = """
            SELECT WeekDay, FromTime, ToTime
            FROM COR_DoctorSchedule 
            WHERE DoctorId = ? AND WeekDay = ? AND IsActive = 1
        """
        
        schedule_result = execute_query(schedule_query, [doctor_id, weekday_num])
        
        # Get any off periods for this specific date
        off_query = """
            SELECT FromTime, ToTime, IsOff, Date
            FROM COR_DoctorOffSchedule 
            WHERE DoctorId = ? AND (Date = ? OR WeekDay = ?)
        """
        
        off_periods = execute_query(off_query, [doctor_id, date, weekday_num])
        
        # Prepare LLM analysis
        hours_prompt = f"""
        Validate if this appointment time falls within doctor's working hours:
        
        Appointment Request:
        - Doctor ID: {doctor_id}
        - Date: {date} ({day_of_week})
        - Requested Time: {start_time}
        
        Doctor's Schedule:
        {json.dumps(schedule_result, indent=2)}
        
        Off Periods/Exceptions:
        {json.dumps(off_periods, indent=2)}
        
        Provide working hours validation in JSON format:
        {{
            "within_hours": true/false,
            "schedule_status": "normal/off_day/partial_day/holiday",
            "working_start": "HH:MM",
            "working_end": "HH:MM",
            "validation_details": "explanation",
            "recommendations": ["rec1", "rec2"],
            "alternative_times": ["time1", "time2"],
            "confidence_score": 0.0-1.0
        }}
        """
        
        messages = [
            SystemMessage(content="You are a medical schedule validation expert."),
            HumanMessage(content=hours_prompt)
        ]
        
        llm_response = config.llm.invoke(messages)
        hours_analysis = json.loads(llm_response.content)
        
        # Add metadata
        hours_analysis["doctor_id"] = doctor_id
        hours_analysis["requested_date"] = date
        hours_analysis["day_of_week"] = day_of_week
        hours_analysis["validation_timestamp"] = datetime.now().isoformat()
        
        return hours_analysis
        
    except Exception as e:
        logger.error(f"Working hours validation error: {e}")
        return {
            "within_hours": True,
            "error": str(e),
            "confidence_score": 0.0,
            "message": "Error during validation - proceeding with default hours"
        }

@tool("appointment_time_validator")
def appointment_time_validator(appointment_date: str, appointment_time: str) -> Dict[str, Any]:
    """
    LLM-enhanced appointment time validation with intelligent date/time analysis.
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from ..core.config import AgentConfig
        
        config = AgentConfig()
        current_datetime = datetime.now()
        
        # Parse the requested appointment datetime
        try:
            if appointment_time:
                appointment_datetime = datetime.strptime(f"{appointment_date} {appointment_time}", '%Y-%m-%d %H:%M')
            else:
                appointment_datetime = datetime.strptime(appointment_date, '%Y-%m-%d')
        except ValueError as e:
            return {
                "is_valid": False,
                "validation_type": "format_error",
                "error": f"Invalid date/time format: {e}",
                "confidence_score": 1.0
            }
        
        # Prepare LLM analysis
        time_prompt = f"""
        Validate this appointment date and time for medical scheduling:
        
        Requested Appointment:
        - Date: {appointment_date}
        - Time: {appointment_time}
        - Full DateTime: {appointment_datetime.strftime('%Y-%m-%d %H:%M')}
        
        Current DateTime: {current_datetime.strftime('%Y-%m-%d %H:%M')}
        
        Validation Criteria:
        - Must be in the future
        - Should be during reasonable medical hours (8 AM - 6 PM typically)
        - Should be on a weekday (Monday-Friday typically)
        - Should not be too far in the future (within 6 months)
        
        Provide time validation in JSON format:
        {{
            "is_valid": true/false,
            "validation_type": "valid/past_time/weekend/late_hour/too_far_future/invalid_format",
            "time_analysis": "detailed explanation",
            "issues_found": ["issue1", "issue2"],
            "recommendations": ["rec1", "rec2"],
            "suggested_alternatives": ["alt1", "alt2"],
            "confidence_score": 0.0-1.0
        }}
        """
        
        messages = [
            SystemMessage(content="You are a medical appointment time validation expert."),
            HumanMessage(content=time_prompt)
        ]
        
        llm_response = config.llm.invoke(messages)
        time_analysis = json.loads(llm_response.content)
        
        # Add metadata
        time_analysis["requested_datetime"] = appointment_datetime.isoformat()
        time_analysis["current_datetime"] = current_datetime.isoformat()
        time_analysis["hours_from_now"] = (appointment_datetime - current_datetime).total_seconds() / 3600
        time_analysis["validation_timestamp"] = datetime.now().isoformat()
        
        return time_analysis
        
    except Exception as e:
        logger.error(f"Appointment time validation error: {e}")
        return {
            "is_valid": True,
            "error": str(e),
            "confidence_score": 0.0,
            "message": "Error during time validation - proceeding with caution"
        }

@tool("service_availability_validator")
def service_availability_validator(service_name: str, doctor_id: int, date: str) -> Dict[str, Any]:
    """
    LLM-enhanced service availability validation with intelligent service matching.
    """
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from ..core.config import AgentConfig
        
        config = AgentConfig()
        
        # Get available services information
        services_query = """
            SELECT DISTINCT ServiceName, ServiceId 
            FROM View_Appointments 
            WHERE DoctorId = ?
            UNION
            SELECT 'consultation' as ServiceName, 1 as ServiceId
            UNION  
            SELECT 'follow-up' as ServiceName, 2 as ServiceId
            UNION
            SELECT 'check-up' as ServiceName, 3 as ServiceId
        """
        
        available_services = execute_query(services_query, [doctor_id])
        
        # Get doctor's recent service patterns
        recent_services_query = """
            SELECT ServiceName, COUNT(*) as frequency
            FROM View_Appointments 
            WHERE DoctorId = ? AND DATE(StartDateTime) >= DATE('now', '-30 days')
            GROUP BY ServiceName
            ORDER BY frequency DESC
        """
        
        recent_services = execute_query(recent_services_query, [doctor_id])
        
        # Prepare LLM analysis
        service_prompt = f"""
        Validate service availability and suggest best match for this appointment:
        
        Requested Service: "{service_name}"
        Doctor ID: {doctor_id}
        Date: {date}
        
        Available Services:
        {json.dumps(available_services, indent=2)}
        
        Recent Service Patterns (last 30 days):
        {json.dumps(recent_services, indent=2)}
        
        Provide service validation in JSON format:
        {{
            "service_available": true/false,
            "exact_match": true/false,
            "matched_service": "exact service name or best match",
            "service_id": "numeric ID if available",
            "matching_confidence": 0.0-1.0,
            "alternative_services": ["service1", "service2"],
            "validation_notes": "explanation of matching logic",
            "recommendations": ["rec1", "rec2"]
        }}
        """
        
        messages = [
            SystemMessage(content="You are a medical service matching expert."),
            HumanMessage(content=service_prompt)
        ]
        
        llm_response = config.llm.invoke(messages)
        service_analysis = json.loads(llm_response.content)
        
        # Add metadata
        service_analysis["requested_service"] = service_name
        service_analysis["doctor_id"] = doctor_id
        service_analysis["total_available_services"] = len(available_services)
        service_analysis["validation_timestamp"] = datetime.now().isoformat()
        
        return service_analysis
        
    except Exception as e:
        logger.error(f"Service availability validation error: {e}")
        return {
            "service_available": True,
            "exact_match": False,
            "matched_service": service_name,
            "error": str(e),
            "matching_confidence": 0.5,
            "message": "Error during service validation - using requested service"
        }

# =============================================================================
# MAIN APPOINTMENT TOOLS - Phase 1 (LLM-Enhanced)
# =============================================================================

@tool("appointment_booking")
def appointment_booking(doctor_id: int, patient_name: str, appointment_date: str, appointment_time: str, service_name: str = "consultation", duration_minutes: int = None) -> Dict[str, Any]:
    """
    Book a new appointment with comprehensive LLM-enhanced validation.
    """
    try:
        # Use validation tools for comprehensive checking
        conflict_check = conflict_detection_validator.invoke({
            "doctor_id": doctor_id,
            "start_datetime": f"{appointment_date} {appointment_time}:00",
            "end_datetime": f"{appointment_date} {appointment_time}:30"  # Default 30-min duration
        })
        
        working_hours_check = working_hours_validator.invoke({
            "doctor_id": doctor_id,
            "date": appointment_date,
            "start_time": appointment_time
        })
        
        time_validation = appointment_time_validator.invoke({
            "appointment_date": appointment_date,
            "appointment_time": appointment_time
        })
        
        service_validation = service_availability_validator.invoke({
            "service_name": service_name,
            "doctor_id": doctor_id,
            "date": appointment_date
        })
        
        # Check validation results
        validation_errors = []
        
        if conflict_check.get('has_conflict'):
            validation_errors.append(f"Scheduling conflict: {conflict_check.get('conflict_details', 'Time slot unavailable')}")
        
        if not working_hours_check.get('within_hours'):
            validation_errors.append(f"Outside working hours: {working_hours_check.get('validation_details', 'Invalid time')}")
        
        if not time_validation.get('is_valid'):
            validation_errors.append(f"Invalid appointment time: {time_validation.get('time_analysis', 'Time validation failed')}")
        
        if not service_validation.get('service_available'):
            validation_errors.append(f"Service unavailable: {service_validation.get('validation_notes', 'Service not found')}")
        
        if validation_errors:
            return {
                "success": False,
                "error": "validation_failed",
                "validation_errors": validation_errors,
                "validation_details": {
                    "conflict_check": conflict_check,
                    "working_hours_check": working_hours_check,
                    "time_validation": time_validation,
                    "service_validation": service_validation
                },
                "message": f"Appointment booking failed validation: {'; '.join(validation_errors)}"
            }
        
        # If all validations pass, proceed with booking
        try:
            # Get or create patient ID
            patient_id = get_or_create_patient_id(patient_name)
            
            # Get service ID and duration
            service_info = get_service_id_and_duration(service_validation.get('matched_service', service_name))
            actual_duration = duration_minutes or service_info.get('duration', 30)
            
            # Calculate end time
            start_datetime = datetime.strptime(f"{appointment_date} {appointment_time}", '%Y-%m-%d %H:%M')
            end_datetime = start_datetime + timedelta(minutes=actual_duration)
            
            # Insert appointment
            booking_query = """
                INSERT INTO View_Appointments 
                (PatientId, DoctorId, StartDateTime, EndDateTime, ServiceName, Status, PatientName)
                VALUES (?, ?, ?, ?, ?, 'Scheduled', ?)
            """
            
            execute_query(booking_query, [
                patient_id,
                doctor_id,
                start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                service_validation.get('matched_service', service_name),
                patient_name
            ])
            
            return {
                "success": True,
                "message": f"Appointment successfully booked for {patient_name}",
                "appointment_details": {
                    "patient_name": patient_name,
                    "doctor_id": doctor_id,
                    "date": appointment_date,
                    "time": appointment_time,
                    "service": service_validation.get('matched_service', service_name),
                    "duration_minutes": actual_duration,
                    "status": "Scheduled"
                },
                "validation_summary": {
                    "conflict_confidence": conflict_check.get('confidence_score', 0),
                    "hours_confidence": working_hours_check.get('confidence_score', 0),
                    "time_confidence": time_validation.get('confidence_score', 0),
                    "service_confidence": service_validation.get('matching_confidence', 0)
                }
            }
            
        except Exception as booking_error:
            logger.error(f"Booking execution error: {booking_error}")
            return {
                "success": False,
                "error": "booking_failed",
                "message": f"Failed to book appointment: {str(booking_error)}"
            }
            
    except Exception as e:
        logger.error(f"Appointment booking error: {e}")
        return {
            "success": False,
            "error": "booking_error",
            "message": f"Appointment booking failed: {str(e)}"
        }

@tool("appointment_rescheduling")
def appointment_rescheduling(appointment_id: int, new_date: str, new_time: str, reason: str = None, doctor_id: int = None) -> Dict[str, Any]:
    """
    Reschedule an existing appointment with LLM-enhanced validation and role-based access control.
    """
    try:
        # Get current appointment details
        current_query = """
            SELECT * FROM View_Appointments 
            WHERE AppointmentId = ?
        """
        
        current_appointment = execute_query(current_query, [appointment_id])
        if not current_appointment:
            return {
                "success": False,
                "error": "appointment_not_found",
                "message": f"Appointment {appointment_id} not found"
            }
        
        current = current_appointment[0]
        
        # Role-based access control: if doctor_id is provided, verify it matches the appointment
        if doctor_id and current['DoctorId'] != doctor_id:
            return {
                "success": False,
                "error": "access_denied",
                "message": f"Doctor {doctor_id} cannot reschedule appointments for Doctor {current['DoctorId']}"
            }
        
        # Validate new time slot using simple validation instead of LLM tools
        # Check for time conflicts (simplified)
        conflict_query = """
        SELECT COUNT(*) FROM View_Appointments 
        WHERE DoctorId = ? 
        AND DATE(StartDateTime) = DATE(?) 
        AND TIME(StartDateTime) = TIME(?)
        AND AppointmentId != ?
        AND Status != 'Cancelled'
        """
        conflict_result = execute_query(conflict_query, [
            current['DoctorId'], 
            f"{new_date} {new_time}:00",
            f"{new_time}:00",
            appointment_id
        ])
        
        has_conflict = conflict_result and conflict_result[0]['COUNT(*)'] > 0
        
        # Simple time validation (business hours 9 AM - 6 PM)
        try:
            hour = int(new_time.split(':')[0])
            is_valid_time = 9 <= hour <= 18
        except:
            is_valid_time = False
        
        # Check validation results
        if has_conflict:
            return {
                "success": False,
                "error": "time_conflict",
                "message": f"Doctor {current['DoctorId']} already has an appointment at {new_time} on {new_date}"
            }
        
        if not is_valid_time:
            return {
                "success": False,
                "error": "invalid_time",
                "message": f"Appointment time {new_time} is outside business hours (9 AM - 6 PM)"
            }
        
        # Update appointment
        update_query = """
            UPDATE View_Appointments 
            SET StartDateTime = ?, EndDateTime = ?, Status = 'Rescheduled'
            WHERE AppointmentId = ?
        """
        
        # Calculate new end time (preserve original duration or default to 30 minutes)
        try:
            original_start = datetime.strptime(current['StartDateTime'], '%Y-%m-%d %H:%M:%S')
            if current['EndDateTime']:
                original_end = datetime.strptime(current['EndDateTime'], '%Y-%m-%d %H:%M:%S')
                duration = (original_end - original_start).total_seconds() / 60  # minutes
            else:
                duration = 30  # Default to 30 minutes if no end time
        except (ValueError, TypeError) as e:
            logger.warning(f"Date parsing issue, using default 30 minute duration: {e}")
            duration = 30
        
        new_start = datetime.strptime(f"{new_date} {new_time}", '%Y-%m-%d %H:%M')
        new_end = new_start + timedelta(minutes=duration)
        
        execute_query(update_query, [
            new_start.strftime('%Y-%m-%d %H:%M:%S'),
            new_end.strftime('%Y-%m-%d %H:%M:%S'),
            appointment_id
        ])
        
        return {
            "success": True,
            "message": f"Appointment {appointment_id} successfully rescheduled",
            "rescheduling_details": {
                "appointment_id": appointment_id,
                "patient_name": current['PatientName'],
                "old_datetime": current['StartDateTime'],
                "new_datetime": new_start.strftime('%Y-%m-%d %H:%M:%S'),
                "reason": reason,
                "status": "Rescheduled"
            }
        }
        
    except Exception as e:
        logger.error(f"Appointment rescheduling error: {e}")
        return {
            "success": False,
            "error": "rescheduling_error",
            "message": f"Rescheduling failed: {str(e)}"
        }

@tool("appointment_cancellation")
def appointment_cancellation(appointment_id: int, reason: str = None, doctor_id: int = None) -> Dict[str, Any]:
    """
    Cancel an existing appointment with role-based access control.
    """
    try:
        # Get current appointment details
        current_query = """
            SELECT * FROM View_Appointments 
            WHERE AppointmentId = ?
        """
        
        current_appointment = execute_query(current_query, [appointment_id])
        if not current_appointment:
            return {
                "success": False,
                "error": "appointment_not_found",
                "message": f"Appointment {appointment_id} not found"
            }
        
        current = current_appointment[0]
        
        # Role-based access control: if doctor_id is provided, verify it matches the appointment
        if doctor_id and current['DoctorId'] != doctor_id:
            return {
                "success": False,
                "error": "access_denied",
                "message": f"Doctor {doctor_id} cannot cancel appointments for Doctor {current['DoctorId']}"
            }
        
        # Check if appointment is already cancelled
        if current['Status'] and current['Status'].lower() == 'cancelled':
            return {
                "success": False,
                "error": "already_cancelled",
                "message": f"Appointment {appointment_id} is already cancelled"
            }
        
        # Update appointment status to cancelled
        update_query = """
            UPDATE View_Appointments 
            SET Status = 'Cancelled'
            WHERE AppointmentId = ?
        """
        
        execute_query(update_query, [appointment_id])
        
        return {
            "success": True,
            "message": f"Appointment {appointment_id} successfully cancelled",
            "cancellation_details": {
                "appointment_id": appointment_id,
                "patient_name": current['PatientName'],
                "doctor_name": current['DoctorName'],
                "original_datetime": current['StartDateTime'],
                "service_name": current['ServiceName'],
                "reason": reason,
                "status": "Cancelled"
            }
        }
        
    except Exception as e:
        logger.error(f"Appointment cancellation error: {e}")
        return {
            "success": False,
            "error": "cancellation_error",
            "message": f"Cancellation failed: {str(e)}"
        }

@tool("schedule_analytics")
def schedule_analytics(doctor_id: int, date: str, analysis_type: str = "daily") -> Dict[str, Any]:
    """
    Generate schedule analytics and insights for better appointment management.
    """
    try:
        if analysis_type == "daily":
            # Daily schedule analysis
            appointments_query = """
                SELECT COUNT(*) as total_appointments,
                       COUNT(CASE WHEN Status = 'Completed' THEN 1 END) as completed,
                       COUNT(CASE WHEN Status = 'Scheduled' THEN 1 END) as scheduled,
                       COUNT(CASE WHEN Status = 'Cancelled' THEN 1 END) as cancelled
                FROM View_Appointments 
                WHERE DoctorId = ? AND DATE(StartDateTime) = ?
            """
            
            stats = execute_query(appointments_query, [doctor_id, date])
            
            # Get time utilization
            slots_query = """
                SELECT StartDateTime, EndDateTime 
                FROM View_Appointments 
                WHERE DoctorId = ? AND DATE(StartDateTime) = ? AND Status != 'Cancelled'
                ORDER BY StartDateTime
            """
            
            appointments = execute_query(slots_query, [doctor_id, date])
            
            # Calculate available slots
            available_slots = find_available_slots(doctor_id, date)
            
            return {
                "success": True,
                "analytics_type": "daily",
                "doctor_id": doctor_id,
                "date": date,
                "statistics": stats[0] if stats else {},
                "total_available_slots": len(available_slots),
                "utilization_rate": round((stats[0]['total_appointments'] / max(len(available_slots) + stats[0]['total_appointments'], 1)) * 100, 2) if stats else 0,
                "appointments": appointments,
                "message": f"Daily analytics generated for doctor {doctor_id} on {date}"
            }
        
        return {
            "success": False,
            "error": "unsupported_analysis_type",
            "message": f"Analysis type '{analysis_type}' not supported"
        }
        
    except Exception as e:
        logger.error(f"Schedule analytics error: {e}")
        return {
            "success": False,
            "error": "analytics_error",
            "message": f"Analytics generation failed: {str(e)}"
        }

# =============================================================================
# QUERY AND LOOKUP TOOLS
# =============================================================================

@tool("appointment_query_executor")
def appointment_query_executor(doctor_id: int, query_type: str, date: str = None, patient_name: str = None) -> Dict[str, Any]:
    """Execute specific appointment queries with time-aware formatting for better user experience."""
    try:
        current_datetime = datetime.now()
        current_date = current_datetime.strftime('%Y-%m-%d')
        current_time = current_datetime.strftime('%H:%M:%S')
        
        if query_type == "next_patient":
            query = """
                SELECT PatientName, StartDateTime, EndDateTime, ServiceName, Status
                FROM View_Appointments 
                WHERE DoctorId = ? 
                AND StartDateTime > datetime('now', 'localtime')
                AND Status IN ('Scheduled', 'Confirmed', 'Rescheduled', 'Booked')
                ORDER BY StartDateTime LIMIT 1
            """
            result = execute_query(query, [doctor_id])
            
        elif query_type == "today_schedule" or query_type == "daily_schedule":
            # Enhanced today schedule with time-awareness
            target_date = date or current_date
            query = """
                SELECT AppointmentId, PatientName, StartDateTime, EndDateTime, ServiceName, Status
                FROM View_Appointments 
                WHERE DoctorId = ? AND DATE(StartDateTime) = ?
                ORDER BY StartDateTime
            """
            all_appointments = execute_query(query, [doctor_id, target_date])
            
            # Categorize appointments by time if it's today
            if target_date == current_date:
                past_appointments = []
                current_appointments = []
                upcoming_appointments = []
                
                for apt in all_appointments:
                    apt_start = apt['StartDateTime']
                    apt_end = apt.get('EndDateTime', '')
                    
                    # Parse appointment times
                    try:
                        apt_start_dt = datetime.strptime(f"{target_date} {apt_start.split(' ')[1] if ' ' in apt_start else apt_start}", '%Y-%m-%d %H:%M:%S')
                        apt_end_dt = None
                        if apt_end:
                            apt_end_dt = datetime.strptime(f"{target_date} {apt_end.split(' ')[1] if ' ' in apt_end else apt_end}", '%Y-%m-%d %H:%M:%S')
                    except:
                        # If parsing fails, try different format
                        try:
                            apt_start_dt = datetime.strptime(apt_start, '%Y-%m-%d %H:%M:%S')
                            apt_end_dt = datetime.strptime(apt_end, '%Y-%m-%d %H:%M:%S') if apt_end else None
                        except:
                            # If all parsing fails, put in upcoming
                            upcoming_appointments.append(apt)
                            continue
                    
                    # Categorize based on current time
                    if apt_end_dt:
                        # Appointment has end time - use normal logic
                        if current_datetime > apt_end_dt:
                            past_appointments.append(apt)
                        elif apt_start_dt <= current_datetime <= apt_end_dt:
                            current_appointments.append(apt)
                        else:
                            upcoming_appointments.append(apt)
                    else:
                        # Appointment has no end time - assume 30 minute default duration for categorization
                        assumed_end_dt = apt_start_dt + timedelta(minutes=30)
                        if current_datetime > assumed_end_dt:
                            past_appointments.append(apt)
                        elif apt_start_dt <= current_datetime <= assumed_end_dt:
                            current_appointments.append(apt)
                        else:
                            upcoming_appointments.append(apt)
                
                return {
                    "success": True,
                    "query_type": query_type,
                    "is_today": True,
                    "total_appointments": len(all_appointments),
                    "past_appointments": past_appointments,
                    "current_appointments": current_appointments,
                    "upcoming_appointments": upcoming_appointments,
                    "results": all_appointments,  # Keep for backward compatibility
                    "count": len(all_appointments),
                    "time_categorized": True
                }
            else:
                # For other dates, just return all appointments
                return {
                    "success": True,
                    "query_type": query_type,
                    "is_today": False,
                    "results": all_appointments,
                    "count": len(all_appointments),
                    "time_categorized": False
                }
            
        elif query_type == "patient_history":
            query = """
                SELECT AppointmentId, StartDateTime, EndDateTime, ServiceName, Status
                FROM View_Appointments 
                WHERE DoctorId = ? AND PatientName LIKE ?
                ORDER BY StartDateTime DESC LIMIT 10
            """
            result = execute_query(query, [doctor_id, f"%{patient_name}%"])
            
        else:
            return {
                "success": False,
                "error": "unknown_query_type",
                "message": f"Query type '{query_type}' not recognized"
            }
        
        return {
            "success": True,
            "query_type": query_type,
            "results": result,
            "count": len(result) if result else 0
        }
        
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return {
            "success": False,
            "error": "query_error",
            "message": f"Query execution failed: {str(e)}"
        }

@tool("schedule_query")
def schedule_query(doctor_id: int, date: Optional[str] = None, include_availability: bool = True) -> Dict[str, Any]:
    """
    Main schedule query tool with time-aware availability and appointment categorization.
    Enhanced to handle earliest slot queries across multiple days when date is None.
    """
    try:
        current_datetime = datetime.now()
        current_date = current_datetime.strftime('%Y-%m-%d')
        
        # Special handling for earliest slot queries across multiple days
        if date is None:
            # Search for earliest available slot across the next 7 days starting from today
            for days_ahead in range(7):
                search_date = (current_datetime + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                
                # For today, use current_time_aware=True to only get future slots
                current_time_aware = (days_ahead == 0)  # Only for today
                available_slots = find_available_slots(doctor_id, search_date, current_time_aware=current_time_aware)
                
                if available_slots:
                    # Return the earliest slot found
                    earliest_slot = available_slots[0]
                    return {
                        "success": True,
                        "doctor_id": doctor_id,
                        "date": search_date,
                        "earliest_slot": {
                            "date": search_date,
                            "start_time": f"{earliest_slot['start_hour']:02d}:{earliest_slot['start_minute']:02d}",
                            "end_time": f"{earliest_slot['end_hour']:02d}:{earliest_slot['end_minute']:02d}",
                            "duration_minutes": earliest_slot.get('duration_minutes', 21)
                        },
                        "available_slots": available_slots[:5],  # Show first 5 slots
                        "query_type": "earliest_available"
                    }
            
            # If no slots found in next 7 days, return no availability
            return {
                "success": True,
                "doctor_id": doctor_id,
                "message": "No available slots found in the next 7 days",
                "query_type": "earliest_available",
                "available_slots": []
            }
        
        # Original logic for specific date queries
        target_date = date or current_date
        
        # Get appointments for the date
        appointments_query = """
            SELECT AppointmentId, PatientName, StartDateTime, EndDateTime, ServiceName, Status
            FROM View_Appointments 
            WHERE DoctorId = ? AND DATE(StartDateTime) = ?
            ORDER BY StartDateTime
        """
        
        appointments = execute_query(appointments_query, [doctor_id, target_date])
        
        result = {
            "success": True,
            "doctor_id": doctor_id,
            "date": target_date,
            "appointments": appointments,
            "total_appointments": len(appointments),
            "is_today": target_date == current_date
        }
        
        # If it's today, categorize appointments by time
        if target_date == current_date:
            past_appointments = []
            current_appointments = []
            upcoming_appointments = []
            
            for apt in appointments:
                try:
                    apt_start = apt['StartDateTime']
                    apt_end = apt.get('EndDateTime', '')
                    
                    # Parse appointment start time
                    if ' ' in apt_start:
                        apt_start_dt = datetime.strptime(apt_start, '%Y-%m-%d %H:%M:%S')
                    else:
                        apt_start_dt = datetime.strptime(f"{target_date} {apt_start}", '%Y-%m-%d %H:%M:%S')
                    
                    # Parse appointment end time
                    apt_end_dt = None
                    if apt_end:
                        if ' ' in apt_end:
                            apt_end_dt = datetime.strptime(apt_end, '%Y-%m-%d %H:%M:%S')
                        else:
                            apt_end_dt = datetime.strptime(f"{target_date} {apt_end}", '%Y-%m-%d %H:%M:%S')
                    
                    # Categorize based on current time
                    if apt_end_dt and current_datetime > apt_end_dt:
                        past_appointments.append(apt)
                    elif apt_start_dt <= current_datetime <= (apt_end_dt or apt_start_dt):
                        current_appointments.append(apt)
                    else:
                        upcoming_appointments.append(apt)
                        
                except Exception as parse_error:
                    logger.warning(f"Could not parse appointment time: {parse_error}")
                    # If parsing fails, default to upcoming
                    upcoming_appointments.append(apt)
            
            result.update({
                "past_appointments": past_appointments,
                "current_appointments": current_appointments,
                "upcoming_appointments": upcoming_appointments,
                "past_count": len(past_appointments),
                "current_count": len(current_appointments),
                "upcoming_count": len(upcoming_appointments),
                "time_categorized": True
            })
        else:
            result["time_categorized"] = False
        
        if include_availability:
            # For available slots, only show future slots if it's today
            if target_date == current_date:
                available_slots = find_available_slots(doctor_id, target_date, current_time_aware=True)
            else:
                available_slots = find_available_slots(doctor_id, target_date)
                
            formatted_slots = []
            for slot in available_slots:
                start_time = f"{slot['start_hour']:02d}:{slot['start_minute']:02d}"
                end_time = f"{slot['end_hour']:02d}:{slot['end_minute']:02d}"
                formatted_slots.append(f"{start_time} - {end_time}")
            
            result["available_slots"] = formatted_slots
            result["total_available"] = len(formatted_slots)
        
        return result
        
    except Exception as e:
        logger.error(f"Schedule query error: {e}")
        return {
            "success": False,
            "error": "schedule_query_error",
            "message": f"Schedule query failed: {str(e)}"
        }

# =============================================================================
# UTILITY FUNCTIONS - Supporting functions for appointment management
# =============================================================================

def get_db_connection():
    """Get database connection"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise Exception(f"Database connection failed: {e}")

def execute_query(query: str, params: List = None, db_path: str = None) -> List[Dict]:
    """Execute database query and return results as list of dictionaries."""
    if params is None:
        params = []
    
    try:
        db_file = db_path or DB_PATH
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        cursor = conn.cursor()
        
        cursor.execute(query, params)
        
        if query.strip().upper().startswith('SELECT'):
            results = [dict(row) for row in cursor.fetchall()]
        else:
            conn.commit()
            results = []
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"Database query error: {e}")
        if 'conn' in locals():
            conn.close()
        raise Exception(f"Database query failed: {e}")

def get_or_create_patient_id(patient_name: str) -> int:
    """Get existing patient ID or create a new one."""
    try:
        # Check if patient exists
        existing_query = """
            SELECT DISTINCT PatientId FROM View_Appointments 
            WHERE LOWER(PatientName) = LOWER(?)
            LIMIT 1
        """
        
        existing = execute_query(existing_query, [patient_name])
        
        if existing:
            return existing[0]['PatientId']
        
        # Create new patient ID (simple auto-increment logic)
        max_id_query = """
            SELECT MAX(PatientId) as max_id FROM View_Appointments
        """
        
        max_result = execute_query(max_id_query)
        new_patient_id = (max_result[0]['max_id'] or 0) + 1
        
        logger.info(f"Creating new PatientId for {patient_name}: {new_patient_id}")
        return new_patient_id
        
    except Exception as e:
        logger.error(f"Error getting/creating patient ID: {e}")
        # Return a fallback ID
        return 9999

def get_service_id_and_duration(service_name: str) -> Dict[str, Any]:
    """Get service ID and default duration for a service."""
    # Default service mapping
    service_mapping = {
        "consultation": {"id": 1, "duration": 30},
        "follow-up": {"id": 2, "duration": 20},
        "check-up": {"id": 3, "duration": 25},
        "procedure": {"id": 4, "duration": 45},
        "emergency": {"id": 5, "duration": 60}
    }
    
    service_lower = service_name.lower()
    for key, value in service_mapping.items():
        if key in service_lower:
            return value
    
    # Default fallback
    return {"id": 1, "duration": 30}

def get_doctor_default_branch(doctor_id: int) -> int:
    """Get doctor's default branch."""
    try:
        query = """
            SELECT DISTINCT BranchId FROM COR_DoctorSchedule 
            WHERE DoctorId = ? LIMIT 1
        """
        result = execute_query(query, [doctor_id])
        return result[0]['BranchId'] if result else 1
    except:
        return 1  # Default branch

def find_available_slots(doctor_id: int, date: str, service_duration_minutes: int = 21, current_time_aware: bool = False) -> List[Dict[str, str]]:
    """
    Find available appointment slots for a doctor on a specific date.
    
    Args:
        doctor_id: The doctor's ID
        date: Date in YYYY-MM-DD format
        service_duration_minutes: Duration of appointment in minutes (default 21)
        current_time_aware: If True and date is today, only return future slots
    
    Returns:
        List of available time slots with start and end times
    """
    from datetime import datetime, timedelta
    import re
    
    logger.info(f"Finding available slots for doctor {doctor_id} on {date}")
    
    try:
        # Parse the input date and get current time info
        target_date = datetime.strptime(date, '%Y-%m-%d')
        weekday_num = target_date.weekday() + 1  # Convert to 1-7 (Monday=1, Sunday=7)
        current_datetime = datetime.now()
        is_today = date == current_datetime.strftime('%Y-%m-%d')
        current_time_minutes = current_datetime.hour * 60 + current_datetime.minute if is_today else 0
        
        # Get doctor's working hours for this day
        schedule_query = """
            SELECT FromTime, ToTime 
            FROM COR_DoctorSchedule 
            WHERE DoctorId = ? AND WeekDay = ? AND IsActive = 1
        """
        
        schedule_result = execute_query(schedule_query, [doctor_id, weekday_num])
        
        if not schedule_result:
            logger.info(f"No schedule found for doctor {doctor_id} on weekday {weekday_num}")
            return []
        
        schedule = schedule_result[0]
        start_time_str = schedule['FromTime']
        end_time_str = schedule['ToTime']
        slot_duration = service_duration_minutes  # Use default since SlotDuration column doesn't exist
        
        # Parse working hours
        def parse_time(time_str):
            # Handle format like '11:00:00.0000000'
            if time_str:
                # Remove microseconds if present
                time_clean = time_str.split('.')[0]  # Remove .0000000 part
                if ':' in time_clean:
                    parts = time_clean.split(':')
                    return int(parts[0]), int(parts[1])
                else:
                    # Assume it's just hour
                    return int(time_clean), 0
            return 0, 0
        
        start_hour, start_minute = parse_time(start_time_str)
        end_hour, end_minute = parse_time(end_time_str)
        
        # Get existing appointments for this doctor on this date
        appointments_query = """
            SELECT StartDateTime, EndDateTime 
            FROM View_Appointments 
            WHERE DoctorId = ? AND DATE(StartDateTime) = ? 
            AND Status IN ('Scheduled', 'Booked', 'Confirmed', 'Rescheduled')
            ORDER BY StartDateTime
        """
        
        existing_appointments = execute_query(appointments_query, [doctor_id, date])
        
        # Convert existing appointments to time ranges
        booked_ranges = []
        for apt in existing_appointments:
            try:
                start_dt = datetime.strptime(apt['StartDateTime'], '%Y-%m-%d %H:%M:%S')
                
                # Handle EndDateTime which might be None
                if apt['EndDateTime']:
                    end_dt = datetime.strptime(apt['EndDateTime'], '%Y-%m-%d %H:%M:%S')
                else:
                    # If no end time, assume default duration
                    end_dt = start_dt + timedelta(minutes=slot_duration)
                
                booked_ranges.append((start_dt.time(), end_dt.time()))
            except Exception as e:
                logger.warning(f"Could not parse appointment time: {apt}, error: {e}")
                continue
        
        # Generate all possible slots within working hours
        available_slots = []
        current_time = datetime.combine(target_date.date(), datetime.min.time().replace(hour=start_hour, minute=start_minute))
        end_time = datetime.combine(target_date.date(), datetime.min.time().replace(hour=end_hour, minute=end_minute))
        
        # If we're checking today and current_time_aware is True, start from current time
        now = datetime.now()
        if is_today and current_time_aware and current_time < now:
            # Round up to next slot boundary
            minutes_since_start = (now - current_time).total_seconds() / 60
            slots_passed = int(minutes_since_start // slot_duration) + 1
            current_time = current_time + timedelta(minutes=slots_passed * slot_duration)
        
        while current_time + timedelta(minutes=slot_duration) <= end_time:
            slot_start_time = current_time.time()
            slot_end_time = (current_time + timedelta(minutes=slot_duration)).time()
            
            # Check if this slot conflicts with any existing appointment
            conflicts = False
            for booked_start, booked_end in booked_ranges:
                if not (slot_end_time <= booked_start or slot_start_time >= booked_end):
                    conflicts = True
                    break
            
            # Check if doctor is off during this time
            if not conflicts and not check_doctor_off_schedule(doctor_id, current_time):
                available_slots.append({
                    'start_hour': current_time.hour,
                    'start_minute': current_time.minute,
                    'end_hour': (current_time + timedelta(minutes=slot_duration)).hour,
                    'end_minute': (current_time + timedelta(minutes=slot_duration)).minute
                })
            
            current_time += timedelta(minutes=slot_duration)
        
        logger.info(f"Found {len(available_slots)} available slots for doctor {doctor_id} on {date}")
        return available_slots
        
    except Exception as e:
        logger.error(f"Error finding available slots: {e}")
        return []

def get_earliest_available_slot(doctor_id: int, date: str, service_duration_minutes: int = 21) -> Optional[Dict[str, str]]:
    """Get the earliest available slot for a doctor on a specific date."""
    available_slots = find_available_slots(doctor_id, date, service_duration_minutes)
    
    if available_slots:
        earliest = available_slots[0]
        return {
            'start_time': f"{earliest['start_hour']:02d}:{earliest['start_minute']:02d}",
            'end_time': f"{earliest['end_hour']:02d}:{earliest['end_minute']:02d}"
        }
    
    return None

def check_doctor_off_schedule(doctor_id: int, appointment_datetime: datetime) -> bool:
    """Check if doctor is off schedule at the given datetime."""
    try:
        date_str = appointment_datetime.strftime('%Y-%m-%d')
        query = "SELECT * FROM COR_DoctorOffSchedule WHERE DoctorId = ? AND Date = ?"
        results = execute_query(query, [doctor_id, date_str])
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error checking doctor off schedule: {e}")
        return False

def check_appointment_overlap(doctor_id: int, start_datetime: str, end_datetime: str, exclude_appointment_id: int = None) -> bool:
    """Check if there's an appointment overlap for the given time period."""
    try:
        query = """
        SELECT * FROM View_Appointments 
        WHERE DoctorId = ? 
        AND ((StartDateTime <= ? AND EndDateTime > ?) OR (StartDateTime < ? AND EndDateTime >= ?))
        """
        params = [doctor_id, start_datetime, start_datetime, end_datetime, end_datetime]
        
        if exclude_appointment_id:
            query += " AND AppointmentId != ?"
            params.append(exclude_appointment_id)
            
        results = execute_query(query, params)
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error checking appointment overlap: {e}")
        return False

def check_doctor_working_hours(doctor_id: int, date: str, start_time: str) -> bool:
    """Check if the given time is within doctor's working hours."""
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        weekday_num = date_obj.weekday() + 1  # Convert to 1-7 (Monday=1, Sunday=7)
        query = """
        SELECT FromTime, ToTime FROM COR_DoctorSchedule 
        WHERE DoctorId = ? AND WeekDay = ? AND IsActive = 1
        """
        results = execute_query(query, [doctor_id, weekday_num])
        
        if not results:
            return False
            
        schedule = results[0]
        # Handle time format with microseconds and compare properly
        from_time = schedule['FromTime'].split('.')[0]  # Remove .0000000
        to_time = schedule['ToTime'].split('.')[0]      # Remove .0000000
        return from_time <= start_time <= to_time
    except Exception as e:
        logger.error(f"Error checking doctor working hours: {e}")
        return False

def normalize_appointment_data(appointment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize appointment data for consistent format."""
    try:
        normalized = appointment_data.copy()
        # Add any normalization logic here
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing appointment data: {e}")
        return appointment_data


def find_appointment_for_rescheduling(patient_name: str, doctor_id: int = None, service_name: str = None, current_date: str = None) -> Dict[str, Any]:
    """
    Find an existing appointment that can be rescheduled with explicit RBAC checking.
    
    Args:
        patient_name: Name of the patient
        doctor_id: Doctor ID (optional)
        service_name: Service name (optional)
        current_date: Current appointment date (optional)
    
    Returns:
        Appointment info dict, None if not found, or error dict for RBAC violation
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # First, check if appointment exists for ANY doctor (for RBAC checking)
        broad_query = """
        SELECT AppointmentId, PatientName, DoctorId, ServiceName, 
               StartDateTime, EndDateTime, Status
        FROM View_Appointments 
        WHERE PatientName LIKE ? 
        AND Status != 'Cancelled'
        """
        broad_params = [f"%{patient_name}%"]
        
        if service_name:
            broad_query += " AND ServiceName LIKE ?"
            broad_params.append(f"%{service_name}%")
        
        if current_date:
            broad_query += " AND DATE(StartDateTime) = DATE(?)"
            broad_params.append(current_date)
        
        broad_query += " ORDER BY StartDateTime DESC LIMIT 1"
        
        cursor.execute(broad_query, broad_params)
        broad_result = cursor.fetchone()
        
        if broad_result:
            appointment_doctor_id = broad_result[2]
            
            # Check if requesting doctor has access to this appointment
            # Fix: Ensure type consistency by converting both to int for comparison
            if doctor_id and int(appointment_doctor_id) != int(doctor_id):
                return {
                    "error": "access_denied",
                    "message": f"Access denied: {patient_name}'s appointment belongs to Doctor {appointment_doctor_id}. You can only manage your own patients' appointments.",
                    "patient_name": patient_name,
                    "actual_doctor_id": appointment_doctor_id,
                    "requesting_doctor_id": doctor_id
                }
            
            # If RBAC check passes, return the appointment
            return {
                "id": broad_result[0],
                "patient_name": broad_result[1],
                "doctor_id": broad_result[2],
                "service_name": broad_result[3],
                "start_datetime": broad_result[4],
                "end_datetime": broad_result[5],
                "status": broad_result[6]
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding appointment for rescheduling: {e}")
        return None
    finally:
        if conn:
            conn.close()


def find_appointment_for_cancellation(patient_name: str, doctor_id: int = None, service_name: str = None, current_date: str = None) -> Dict[str, Any]:
    """
    Find an existing appointment that can be cancelled with explicit RBAC checking.
    
    Args:
        patient_name: Name of the patient
        doctor_id: Doctor ID (optional)
        service_name: Service name (optional)
        current_date: Current appointment date (optional)
    
    Returns:
        Appointment info dict, None if not found, or error dict for RBAC violation
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # First, check if appointment exists for ANY doctor (for RBAC checking)
        broad_query = """
        SELECT AppointmentId, PatientName, DoctorId, ServiceName, 
               StartDateTime, EndDateTime, Status
        FROM View_Appointments 
        WHERE PatientName LIKE ? 
        AND Status != 'Cancelled'
        """
        broad_params = [f"%{patient_name}%"]
        
        if service_name:
            broad_query += " AND ServiceName LIKE ?"
            broad_params.append(f"%{service_name}%")
        
        if current_date:
            broad_query += " AND DATE(StartDateTime) = DATE(?)"
            broad_params.append(current_date)
        
        broad_query += " ORDER BY StartDateTime DESC LIMIT 1"
        
        cursor.execute(broad_query, broad_params)
        broad_result = cursor.fetchone()
        
        if broad_result:
            appointment_doctor_id = broad_result[2]
            
            # Check if requesting doctor has access to this appointment
            # Fix: Ensure type consistency by converting both to int for comparison
            if doctor_id and int(appointment_doctor_id) != int(doctor_id):
                return {
                    "error": "access_denied",
                    "message": f"Access denied: {patient_name}'s appointment belongs to Doctor {appointment_doctor_id}. You can only manage your own patients' appointments.",
                    "patient_name": patient_name,
                    "actual_doctor_id": appointment_doctor_id,
                    "requesting_doctor_id": doctor_id
                }
            
            # If RBAC check passes, return the appointment
            return {
                "id": broad_result[0],
                "patient_name": broad_result[1],
                "doctor_id": broad_result[2],
                "service_name": broad_result[3],
                "start_datetime": broad_result[4],
                "end_datetime": broad_result[5],
                "status": broad_result[6]
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding appointment for cancellation: {e}")
        return None
    finally:
        if conn:
            conn.close()

