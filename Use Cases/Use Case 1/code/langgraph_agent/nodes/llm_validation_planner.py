"""
Intelligent validation planning system where LLM decides which validations to run.
This allows the LLM to strategically choose validation tools based on appointment context.
"""

from typing import Dict, Any, List
from ..tools.validation_tools import VALIDATION_TOOLS
import json
import logging

logger = logging.getLogger(__name__)


def llm_validation_planner_node(state: Dict[str, Any], config=None) -> Dict[str, Any]:
    """
    LLM-driven validation planning node that intelligently selects which validations to run.
    
    The LLM analyzes the appointment booking context and decides:
    1. Which validation tools are needed
    2. In what order to run them
    3. How to interpret the results
    """
    
    tool_parameters = state.get('tool_parameters', {})
    
    # Extract validation context
    start_datetime = tool_parameters.get('StartDateTime')
    doctor_id = tool_parameters.get('doctor_id')
    service_name = tool_parameters.get('service_name')
    
    if not start_datetime or not doctor_id:
        logger.warning("⚠️ Insufficient data for LLM validation planning")
        return {
            **state,
            'validation_status': 'skipped',
            'validation_results': {
                'valid': True,
                'message': 'Validation skipped due to insufficient data'
            }
        }
    
    # Let LLM create a validation strategy using the existing config.llm
    if not config or not hasattr(config, 'llm'):
        logger.error("❌ No LLM configuration available for validation planning")
        return fallback_validation(state, start_datetime, doctor_id, service_name)
    
    validation_prompt = f"""
You are an intelligent appointment validation planner. Analyze this booking request and decide which validations to run:

BOOKING CONTEXT:
- Appointment Time: {start_datetime}
- Doctor ID: {doctor_id}
- Service: {service_name or 'Not specified'}

AVAILABLE VALIDATION TOOLS:
1. validate_booking_conflicts_tool - Check for scheduling conflicts
2. validate_working_hours_tool - Verify appointment is within working hours  
3. validate_appointment_timing_tool - Ensure appointment is not in the past
4. validate_service_availability_tool - Check if service exists
5. check_doctor_off_schedule_tool - Verify doctor is not off on that day

TASK: Create a validation plan by selecting relevant tools.

Respond ONLY with valid JSON:
{{
    "validation_plan": [
        {{
            "tool": "validate_appointment_timing_tool",
            "priority": 1,
            "reasoning": "Check if appointment is in future"
        }}
    ],
    "strategy": "Brief explanation of validation approach"
}}"""
    
    try:
        from langchain.schema import HumanMessage, SystemMessage
        
        messages = [
            SystemMessage(content="You are an expert medical appointment validation planner."),
            HumanMessage(content=validation_prompt)
        ]
        
        response = config.llm.invoke(messages)
        
        # Debug the LLM response
        logger.info(f"🔍 LLM response type: {type(response)}")
        logger.info(f"🔍 LLM response content: '{response.content}'")
        
        if not response.content or not response.content.strip():
            logger.error("❌ LLM returned empty response")
            raise ValueError("Empty LLM response")
            
        validation_plan = json.loads(response.content)
        logger.info(f"🧠 LLM created validation plan: {validation_plan['strategy']}")
        
        # Execute the LLM's validation plan
        validation_results = execute_validation_plan(
            validation_plan['validation_plan'],
            {
                'start_datetime': start_datetime,
                'doctor_id': doctor_id,
                'service_name': service_name
            }
        )
        
        return {
            **state,
            'validation_status': 'completed',
            'validation_plan': validation_plan,
            'validation_results': validation_results,
            'llm_planned_validation': True
        }
        
    except Exception as e:
        logger.error(f"❌ LLM validation planning failed: {str(e)}")
        # Fallback to basic validation
        return fallback_validation(state, start_datetime, doctor_id, service_name)


def execute_validation_plan(plan: List[Dict], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the LLM's validation plan by calling the selected tools in order.
    """
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'tool_results': {},
        'execution_order': []
    }
    
    # Sort by priority
    sorted_plan = sorted(plan, key=lambda x: x.get('priority', 999))
    
    for step in sorted_plan:
        tool_name = step['tool']
        
        if tool_name not in VALIDATION_TOOLS:
            logger.warning(f"⚠️ Unknown validation tool: {tool_name}")
            continue
            
        logger.info(f"🔧 LLM executing validation tool: {tool_name}")
        logger.info(f"📋 Reasoning: {step.get('reasoning', 'No reasoning provided')}")
        
        # Get the tool function
        tool_function = VALIDATION_TOOLS[tool_name]['function']
        
        # Prepare parameters based on tool requirements
        tool_params = prepare_tool_parameters(tool_name, context)
        
        try:
            # Execute the validation tool
            result = tool_function(**tool_params)
            
            validation_results['tool_results'][tool_name] = result
            validation_results['execution_order'].append(tool_name)
            
            # Check if validation failed
            if not result.get('valid', True):
                validation_results['valid'] = False
                validation_results['errors'].append(result)
                
                # Log LLM's tool execution
                logger.warning(f"❌ LLM validation tool {tool_name} failed: {result.get('message')}")
            else:
                logger.info(f"✅ LLM validation tool {tool_name} passed")
                
        except Exception as e:
            logger.error(f"❌ Error executing LLM validation tool {tool_name}: {str(e)}")
            validation_results['valid'] = False
            validation_results['errors'].append({
                'valid': False,
                'error_type': 'tool_execution_error',
                'tool': tool_name,
                'message': f"Error executing {tool_name}: {str(e)}"
            })
    
    return validation_results


def prepare_tool_parameters(tool_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare parameters for validation tools based on their specific requirements.
    """
    
    # Map each tool to its required parameters
    tool_param_mapping = {
        'validate_booking_conflicts_tool': {
            'start_datetime': context.get('start_datetime'),
            'doctor_id': context.get('doctor_id'),
            'duration_minutes': 21  # default duration
        },
        'validate_working_hours_tool': {
            'start_datetime': context.get('start_datetime'),
            'doctor_id': context.get('doctor_id')
        },
        'validate_appointment_timing_tool': {
            'start_datetime': context.get('start_datetime')
        },
        'validate_service_availability_tool': {
            'service_name': context.get('service_name'),
            'doctor_id': context.get('doctor_id')
        },
        'check_doctor_off_schedule_tool': {
            'start_datetime': context.get('start_datetime'),
            'doctor_id': context.get('doctor_id')
        }
    }
    
    # Return the appropriate parameters for this tool
    return tool_param_mapping.get(tool_name, {})


def fallback_validation(state: Dict[str, Any], start_datetime: str, doctor_id: str, service_name: str) -> Dict[str, Any]:
    """
    Fallback validation when LLM planning fails - run essential validations.
    """
    
    logger.info("🔄 Running fallback validation (LLM planning failed)")
    
    essential_validations = [
        ('validate_appointment_timing', {'start_datetime': start_datetime}),
        ('validate_working_hours', {'start_datetime': start_datetime, 'doctor_id': doctor_id}),
        ('validate_booking_conflicts', {'start_datetime': start_datetime, 'doctor_id': doctor_id})
    ]
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'tool_results': {},
        'fallback_mode': True
    }
    
    for tool_name, params in essential_validations:
        try:
            tool_function = VALIDATION_TOOLS[tool_name]['function']
            result = tool_function(**params)
            
            validation_results['tool_results'][tool_name] = result
            
            if not result.get('valid', True):
                validation_results['valid'] = False
                validation_results['errors'].append(result)
                
        except Exception as e:
            logger.error(f"❌ Fallback validation error for {tool_name}: {str(e)}")
    
    return {
        **state,
        'validation_status': 'fallback_completed',
        'validation_results': validation_results,
        'llm_planned_validation': False
    }


def format_llm_validation_response(validation_results: Dict[str, Any]) -> str:
    """
    Format validation results for LLM response generation.
    """
    
    if validation_results.get('valid', True):
        return "All LLM-planned validations passed successfully."
    
    # Format errors in a user-friendly way
    errors = validation_results.get('errors', [])
    if not errors:
        return "Validation completed with unknown status."
    
    # Group errors by type
    error_messages = []
    for error in errors:
        tool_used = error.get('tool_used', 'Unknown tool')
        message = error.get('message', 'Validation failed')
        error_messages.append(f"• {message} (checked by {tool_used})")
    
    return f"Appointment validation failed:\n" + "\n".join(error_messages)
