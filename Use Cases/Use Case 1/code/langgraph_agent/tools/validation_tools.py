"""
Validation tools that the LLM can actively choose to call during booking process.
These tools provide intelligent validation capabilities that the LLM can use based on context.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pytz

def validate_booking_conflicts_tool(
    start_datetime: str, 
    doctor_id: str, 
    duration_minutes: int = 21
) -> Dict[str, Any]:
    """
    Tool for the LLM to check if a proposed appointment time conflicts with existing bookings.
    
    Args:
        start_datetime: Appointment start time in 'YYYY-MM-DD HH:MM:SS' format
        doctor_id: Doctor's ID to check conflicts for
        duration_minutes: Expected appointment duration (default 21 minutes)
        
    Returns:
        dict: Validation result with conflict details
    """
    try:
        from ..tools.database import execute_query
        
        # Calculate end time
        start_dt = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        end_datetime = end_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Check for conflicts using database query
        conflict_query = """
        SELECT PatientName, StartDateTime, EndDateTime, ServiceName
        FROM View_Appointments 
        WHERE DoctorId = ? 
        AND StartDateTime <= ? 
        AND EndDateTime > ?
        AND StatusId = 1
        """
        
        try:
            conflicts = execute_query(
                conflict_query, 
                [doctor_id, end_datetime, start_datetime]
            )
        except Exception as db_error:
            return {
                "valid": False,
                "error_type": "database_error",
                "message": f"Database error checking conflicts: {str(db_error)}",
                "tool_used": "validate_booking_conflicts_tool"
            }
        
        if conflicts:
            return {
                "valid": False,
                "error_type": "booking_conflict",
                "conflicts": [
                    {
                        "PatientName": conf[0],
                        "StartDateTime": conf[1], 
                        "EndDateTime": conf[2],
                        "ServiceName": conf[3]
                    } for conf in conflicts
                ],
                "message": f"Time slot {start_dt.strftime('%I:%M %p')} is already booked",
                "tool_used": "validate_booking_conflicts_tool"
            }
        
        return {
            "valid": True,
            "message": "No booking conflicts found",
            "tool_used": "validate_booking_conflicts_tool"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error_type": "validation_error",
            "message": f"Error checking booking conflicts: {str(e)}",
            "tool_used": "validate_booking_conflicts_tool"
        }


def validate_working_hours_tool(start_datetime: str, doctor_id: str) -> Dict[str, Any]:
    """
    Tool for the LLM to validate if appointment is within doctor's working hours.
    
    Args:
        start_datetime: Appointment start time in 'YYYY-MM-DD HH:MM:SS' format
        doctor_id: Doctor's ID to check working hours for
        
    Returns:
        dict: Validation result with working hours details
    """
    try:
        from ..tools.database import execute_query
        
        # Parse the datetime
        appointment_dt = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        weekday = appointment_dt.weekday() + 1  # Convert to 1-7 (Monday=1)
        appointment_time = appointment_dt.strftime('%H:%M:%S')
        
        # Query doctor's schedule for this weekday
        schedule_query = """
        SELECT FromTime, ToTime 
        FROM COR_DoctorSchedule 
        WHERE DoctorId = ? AND WeekDay = ?
        """
        
        schedule = execute_query(schedule_query, [doctor_id, weekday])
        
        if not schedule:
            return {
                "valid": False,
                "error_type": "no_schedule",
                "weekday": appointment_dt.strftime('%A'),
                "message": f"No working hours found for {appointment_dt.strftime('%A')}",
                "tool_used": "validate_working_hours_tool"
            }
        
        from_time, to_time = schedule[0]
        
        # Convert times for comparison - handle both time and datetime formats
        try:
            # Try parsing as time first (HH:MM:SS format)
            from_dt = datetime.strptime(f"{appointment_dt.date()} {from_time}", '%Y-%m-%d %H:%M:%S')
            to_dt = datetime.strptime(f"{appointment_dt.date()} {to_time}", '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # If that fails, try parsing as just time (HH:MM format)
            try:
                from_dt = datetime.strptime(f"{appointment_dt.date()} {from_time}", '%Y-%m-%d %H:%M')
                to_dt = datetime.strptime(f"{appointment_dt.date()} {to_time}", '%Y-%m-%d %H:%M')
            except ValueError:
                # Log the actual values for debugging
                return {
                    "valid": False,
                    "error_type": "time_format_error",
                    "message": f"Could not parse working hours format. FromTime: '{from_time}', ToTime: '{to_time}'",
                    "tool_used": "validate_working_hours_tool"
                }
        
        if appointment_dt < from_dt or appointment_dt >= to_dt:
            return {
                "valid": False,
                "error_type": "outside_hours",
                "working_hours": {
                    "start": from_dt.strftime('%I:%M %p'),
                    "end": to_dt.strftime('%I:%M %p')
                },
                "requested_time": appointment_dt.strftime('%I:%M %p'),
                "weekday": appointment_dt.strftime('%A'),
                "message": f"Appointment time {appointment_dt.strftime('%I:%M %p')} is outside working hours",
                "tool_used": "validate_working_hours_tool"
            }
        
        return {
            "valid": True,
            "working_hours": {
                "start": from_dt.strftime('%I:%M %p'),
                "end": to_dt.strftime('%I:%M %p')
            },
            "message": "Appointment is within working hours",
            "tool_used": "validate_working_hours_tool"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error_type": "validation_error",
            "message": f"Error validating working hours: {str(e)}",
            "tool_used": "validate_working_hours_tool"
        }


def validate_appointment_timing_tool(start_datetime: str) -> Dict[str, Any]:
    """
    Tool for the LLM to validate if appointment is scheduled for a future date/time.
    
    Args:
        start_datetime: Appointment start time in 'YYYY-MM-DD HH:MM:SS' format
        
    Returns:
        dict: Validation result with timing details
    """
    try:
        appointment_dt = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        current_dt = datetime.now()
        
        if appointment_dt <= current_dt:
            return {
                "valid": False,
                "error_type": "past_appointment",
                "appointment_time": appointment_dt.strftime('%Y-%m-%d %I:%M %p'),
                "current_time": current_dt.strftime('%Y-%m-%d %I:%M %p'),
                "message": "Cannot book appointments in the past",
                "tool_used": "validate_appointment_timing_tool"
            }
        
        # Check if appointment is too far in the future (optional business rule)
        max_advance_days = 365  # 1 year advance booking limit
        max_future_dt = current_dt + timedelta(days=max_advance_days)
        
        if appointment_dt > max_future_dt:
            return {
                "valid": False,
                "error_type": "too_far_future",
                "appointment_time": appointment_dt.strftime('%Y-%m-%d %I:%M %p'),
                "max_advance_date": max_future_dt.strftime('%Y-%m-%d'),
                "message": f"Cannot book appointments more than {max_advance_days} days in advance",
                "tool_used": "validate_appointment_timing_tool"
            }
        
        return {
            "valid": True,
            "appointment_time": appointment_dt.strftime('%Y-%m-%d %I:%M %p'),
            "message": "Appointment timing is valid",
            "tool_used": "validate_appointment_timing_tool"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error_type": "validation_error",
            "message": f"Error validating appointment timing: {str(e)}",
            "tool_used": "validate_appointment_timing_tool"
        }


def validate_service_availability_tool(service_name: str, doctor_id: str) -> Dict[str, Any]:
    """
    Tool for the LLM to validate if requested service is available with the doctor.
    
    Args:
        service_name: Name of the requested service
        doctor_id: Doctor's ID to check service availability for
        
    Returns:
        dict: Validation result with service availability details
    """
    try:
        from ..tools.database import execute_query
        
        # Check if service exists in the system
        service_query = """
        SELECT DISTINCT ServiceName 
        FROM View_Appointments 
        WHERE UPPER(ServiceName) LIKE UPPER(?) 
        LIMIT 10
        """
        
        available_services = execute_query(
            service_query, 
            [f'%{service_name}%']
        )
        
        if not available_services:
            # Get all available services for suggestions
            all_services_query = """
            SELECT DISTINCT ServiceName 
            FROM View_Appointments 
            ORDER BY ServiceName
            """
            
            all_services = execute_query(all_services_query, [])
            
            return {
                "valid": False,
                "error_type": "service_not_found",
                "requested_service": service_name,
                "available_services": [service[0] for service in all_services[:10]],
                "message": f"Service '{service_name}' not found in system",
                "tool_used": "validate_service_availability_tool"
            }
        
        # Service exists, check if doctor provides this service
        doctor_service_query = """
        SELECT COUNT(*) 
        FROM View_Appointments 
        WHERE DoctorId = ? 
        AND UPPER(ServiceName) LIKE UPPER(?)
        """
        
        try:
            doctor_provides_service = execute_query(
                doctor_service_query, 
                [doctor_id, f'%{service_name}%']
            )
            
            service_count = doctor_provides_service[0][0] if doctor_provides_service else 0
            
            if service_count == 0:
                return {
                    "valid": False,
                    "error_type": "service_not_provided_by_doctor",
                    "requested_service": service_name,
                    "doctor_id": doctor_id,
                    "message": f"Doctor does not provide '{service_name}' service",
                    "tool_used": "validate_service_availability_tool"
                }
        except Exception as db_error:
            return {
                "valid": False,
                "error_type": "database_error",
                "message": f"Database error checking doctor services: {str(db_error)}",
                "tool_used": "validate_service_availability_tool"
            }
        
        # Find the exact service name match
        try:
            exact_service = available_services[0][0]  # Take the first match
        except (IndexError, TypeError) as e:
            return {
                "valid": False,
                "error_type": "data_access_error",
                "message": f"Error accessing service data: {str(e)}",
                "tool_used": "validate_service_availability_tool"
            }
        
        return {
            "valid": True,
            "requested_service": service_name,
            "matched_service": exact_service,
            "message": f"Service '{exact_service}' is available",
            "tool_used": "validate_service_availability_tool"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error_type": "validation_error",
            "message": f"Error validating service availability: {str(e)}",
            "tool_used": "validate_service_availability_tool"
        }


def check_doctor_off_schedule_tool(start_datetime: str, doctor_id: str) -> Dict[str, Any]:
    """
    Tool for the LLM to check if doctor is off/unavailable on the requested date.
    
    Args:
        start_datetime: Appointment start time in 'YYYY-MM-DD HH:MM:SS' format
        doctor_id: Doctor's ID to check off-schedule for
        
    Returns:
        dict: Validation result with off-schedule details
    """
    try:
        from ..tools.database import execute_query
        
        appointment_dt = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        appointment_date = appointment_dt.strftime('%Y-%m-%d')
        
        # Check if doctor is off on this date
        off_schedule_query = """
        SELECT Date, Comment 
        FROM COR_DoctorOffSchedule 
        WHERE DoctorId = ? 
        AND Date = ?
        """
        
        off_schedule = execute_query(
            off_schedule_query, 
            [doctor_id, appointment_date]
        )
        
        if off_schedule:
            off_date, comment = off_schedule[0]
            return {
                "valid": False,
                "error_type": "doctor_unavailable",
                "off_date": off_date,
                "reason": comment or "Doctor is not available",
                "message": f"Doctor is not available on {appointment_dt.strftime('%B %d, %Y')}",
                "tool_used": "check_doctor_off_schedule_tool"
            }
        
        return {
            "valid": True,
            "message": "Doctor is available on requested date",
            "tool_used": "check_doctor_off_schedule_tool"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error_type": "validation_error",
            "message": f"Error checking doctor availability: {str(e)}",
            "tool_used": "check_doctor_off_schedule_tool"
        }


# Tool registry for LLM access
VALIDATION_TOOLS = {
    "validate_booking_conflicts_tool": {
        "function": validate_booking_conflicts_tool,
        "description": "Check if proposed appointment time conflicts with existing bookings",
        "parameters": {
            "start_datetime": "Appointment start time (YYYY-MM-DD HH:MM:SS)",
            "doctor_id": "Doctor's ID",
            "duration_minutes": "Appointment duration in minutes (optional, default 21)"
        }
    },
    "validate_working_hours_tool": {
        "function": validate_working_hours_tool,
        "description": "Validate if appointment is within doctor's working hours",
        "parameters": {
            "start_datetime": "Appointment start time (YYYY-MM-DD HH:MM:SS)",
            "doctor_id": "Doctor's ID"
        }
    },
    "validate_appointment_timing_tool": {
        "function": validate_appointment_timing_tool,
        "description": "Validate if appointment is scheduled for a valid future date/time",
        "parameters": {
            "start_datetime": "Appointment start time (YYYY-MM-DD HH:MM:SS)"
        }
    },
    "validate_service_availability_tool": {
        "function": validate_service_availability_tool,
        "description": "Validate if requested service is available with the doctor",
        "parameters": {
            "service_name": "Name of the requested service",
            "doctor_id": "Doctor's ID"
        }
    },
    "check_doctor_off_schedule_tool": {
        "function": check_doctor_off_schedule_tool,
        "description": "Check if doctor is off/unavailable on the requested date",
        "parameters": {
            "start_datetime": "Appointment start time (YYYY-MM-DD HH:MM:SS)",
            "doctor_id": "Doctor's ID"
        }
    }
}
