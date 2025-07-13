"""
Multi-Turn Medical Assistant Processing Nodes

This module contains the core processing nodes for the LangGraph medical assistant
that handles context resolution, tool selection, SQL generation, and response formatting
with enhanced MCP (Model Context Protocol) integration for superior context preservation.
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph_agent.core.config import AgentConfig
from langgraph_agent.core.state import AgentState, update_patient_context, add_to_conversation_memory
from langgraph_agent.tools.database import (
    execute_query, resolve_doctor_uuid_to_id, resolve_doctor_name_from_uuid,
    get_next_appointment, get_patient_history, get_doctor_schedule
)
from langgraph_agent.tools.mcp_context_manager import mcp_context_manager

logger = logging.getLogger(__name__)


def context_resolver_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Enhanced context resolver with MCP (Model Context Protocol) integration.
    
    This node analyzes the current query against conversation memory and 
    existing context to resolve references like "next patient", "she", etc.
    Enhanced with MCP for standardized context management.
    """
    logger.info(f"Context resolver processing: {state['current_query']}")
    
    session_id = state.get("session_id", "default_session")
    
    # First, try MCP context resolution for better reference handling
    mcp_resolved_refs = {}
    query_lower = state["current_query"].lower()
    
    # Common reference patterns
    reference_patterns = ["next patient", "her", "him", "that appointment", "this patient", "my schedule"]
    
    for pattern in reference_patterns:
        if pattern in query_lower:
            resolved = mcp_context_manager.resolve_reference(
                reference=pattern,
                session_id=session_id
            )
            if resolved:
                mcp_resolved_refs[pattern] = resolved
                logger.info(f"MCP resolved '{pattern}' -> {resolved.get('patient_name', 'context')}")
    
    system_prompt = config.get_system_prompt("context_resolver")
    
    # Enhanced context information including MCP context
    mcp_context_summary = mcp_context_manager.get_context_summary(session_id)
    
    context_info = {
        "current_query": state["current_query"],
        "user_role": state["user_role"],
        "doctor_id": state.get("doctor_id"),
        "patient_context": state.get("patient_context"),
        "doctor_context": state.get("doctor_context"),
        "conversation_memory": state["conversation_memory"],
        "recent_messages": [msg.content for msg in state["messages"][-3:] if hasattr(msg, 'content')],
        "mcp_context_summary": mcp_context_summary,
        "mcp_resolved_references": mcp_resolved_refs
    }
    
    prompt = f"""
{system_prompt}

Current context (enhanced with MCP):
{json.dumps(context_info, indent=2, default=str)}

Analyze the query "{state['current_query']}" and provide:
1. query_intent: The main intent (next_patient, patient_history, schedule, etc.)
2. resolved_references: Dictionary mapping implicit references to explicit values
3. context_updates: Any updates needed to patient or doctor context

Use MCP-resolved references when available for better accuracy.

Respond in JSON format:
{{
    "query_intent": "...",
    "resolved_references": {{}},
    "context_updates": {{}},
    "reasoning": "..."
}}
"""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = config.llm.invoke(messages)
        
        # Debug the response content
        logger.info(f"Context resolver OpenAI response content: '{response.content}'")
        
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from OpenAI in context resolver")
            raise ValueError("Empty response from OpenAI")
        
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error in context resolver: {je}")
            logger.error(f"Response content was: '{response.content}'")
            # Use fallback logic if JSON parsing fails
            raise je
        
        # Update state with resolved information
        state["query_intent"] = result.get("query_intent")
        
        # Merge MCP-resolved references with LLM-resolved references
        llm_resolved = result.get("resolved_references", {})
        final_resolved = {**mcp_resolved_refs, **llm_resolved}
        state["resolved_references"] = final_resolved
        
        # Apply context updates
        context_updates = result.get("context_updates", {})
        if context_updates.get("patient_context"):
            state = update_patient_context(state, **context_updates["patient_context"])
        
        logger.info(f"Context resolved - Intent: {state['query_intent']}, References: {state['resolved_references']}")
        
    except Exception as e:
        logger.error(f"Context resolution error: {e}")
        state["errors"].append(f"Context resolution failed: {e}")
        state["has_errors"] = True
        # Fallback to basic intent detection
        query_lower = state["current_query"].lower()
        if any(word in query_lower for word in ["next", "upcoming"]):
            state["query_intent"] = "next_patient"
        elif any(word in query_lower for word in ["history", "past"]):
            state["query_intent"] = "patient_history"
        elif any(word in query_lower for word in ["schedule", "calendar"]):
            state["query_intent"] = "schedule"
        else:
            state["query_intent"] = "general_query"
        
        # Use MCP-resolved references even if LLM fails
        if mcp_resolved_refs:
            state["resolved_references"] = mcp_resolved_refs
            logger.info(f"Using MCP fallback references: {mcp_resolved_refs}")
    
    return state


def tool_selector_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Select the appropriate tool and parameters based on resolved context.
    """
    logger.info(f"Tool selector processing intent: {state['query_intent']}")
    
    system_prompt = config.get_system_prompt("tool_selector")
    
    # Get allowed tools for user role
    allowed_tools = config.get_role_permissions(state["user_role"])
    
    selection_context = {
        "query_intent": state["query_intent"],
        "resolved_references": state["resolved_references"],
        "user_role": state["user_role"],
        "doctor_id": state.get("doctor_id"),
        "allowed_tools": allowed_tools,
        "patient_context": state.get("patient_context"),
        "doctor_context": state.get("doctor_context")
    }
    
    prompt = f"""
{system_prompt}

Selection context:
{json.dumps(selection_context, indent=2, default=str)}

Select the best tool and parameters. Respond in JSON format:
{{
    "selected_tool": "tool_name",
    "tool_parameters": {{}},
    "reasoning": "..."
}}
"""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = config.llm.invoke(messages)
        
        # Debug the response content
        logger.info(f"Tool selector OpenAI response content: '{response.content}'")
        
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from OpenAI in tool selector")
            raise ValueError("Empty response from OpenAI")
        
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error in tool selector: {je}")
            logger.error(f"Response content was: '{response.content}'")
            # Use fallback logic if JSON parsing fails
            raise je
        
        state["selected_tool"] = result.get("selected_tool")
        state["tool_parameters"] = result.get("tool_parameters", {})
        
        logger.info(f"Tool selected: {state['selected_tool']} with params: {state['tool_parameters']}")
        
    except Exception as e:
        logger.error(f"Tool selection error: {e}")
        state["errors"].append(f"Tool selection failed: {e}")
        state["has_errors"] = True
        # Fallback tool selection
        if state["query_intent"] == "next_patient":
            state["selected_tool"] = "appointment_lookup"
            state["tool_parameters"] = {"type": "next_appointment", "doctor_id": state.get("doctor_id")}
        else:
            state["selected_tool"] = "schedule_query"
            state["tool_parameters"] = {"doctor_id": state.get("doctor_id")}
    
    return state


def sql_generator_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Generate SQL query based on selected tool and parameters.
    """
    logger.info(f"SQL generator processing tool: {state['selected_tool']}")
    
    # Handle next_patient queries directly with get_next_appointment function
    if state["query_intent"] == "next_patient":
        doctor_uuid = state.get("doctor_id")
        if doctor_uuid:
            logger.info(f"Using get_next_appointment for doctor UUID: {doctor_uuid}")
            state["tool_results"] = get_next_appointment(doctor_uuid)
            logger.info(f"Next appointment query results: {len(state['tool_results']) if state['tool_results'] else 0} rows")
            return state
    
    system_prompt = config.get_system_prompt("sql_generator")
    
    # Map doctor UUID to integer DoctorId if needed
    doctor_uuid = state.get("doctor_id")
    doctor_id_mapped = None
    if doctor_uuid:
        doctor_id_mapped = resolve_doctor_uuid_to_id(doctor_uuid)
        logger.info(f"Mapped doctor UUID {doctor_uuid} to DoctorId {doctor_id_mapped}")
    
    generation_context = {
        "selected_tool": state["selected_tool"],
        "tool_parameters": state["tool_parameters"],
        "resolved_references": state["resolved_references"],
        "doctor_uuid": doctor_uuid,
        "doctor_id_mapped": doctor_id_mapped,
        "patient_context": state.get("patient_context")
    }
    
    prompt = f"""
{system_prompt}

IMPORTANT: The View_Appointments table uses integer DoctorId, not UUID. 
Doctor UUID {doctor_uuid} maps to DoctorId {doctor_id_mapped}.

Generation context:
{json.dumps(generation_context, indent=2, default=str)}

Generate the appropriate SQL query using the mapped DoctorId. Respond in JSON format:
{{
    "sql_query": "SELECT ...",
    "query_parameters": [],
    "reasoning": "..."
}}
"""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = config.llm.invoke(messages)
        
        # Debug the response content
        logger.info(f"OpenAI response content: '{response.content}'")
        
        if not response.content or response.content.strip() == "":
            logger.error("Empty response from OpenAI")
            raise ValueError("Empty response from OpenAI")
        
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as je:
            logger.error(f"JSON decode error: {je}")
            logger.error(f"Response content was: '{response.content}'")
            # Try to extract SQL from non-JSON response
            content = response.content.strip()
            if "SELECT" in content.upper():
                # Extract SQL query from the response
                lines = content.split('\n')
                sql_line = None
                for line in lines:
                    if "SELECT" in line.upper():
                        sql_line = line.strip()
                        break
                if sql_line:
                    logger.info(f"Extracted SQL from non-JSON response: {sql_line}")
                    result = {"sql_query": sql_line, "query_parameters": []}
                else:
                    raise je
            else:
                raise je
        
        state["sql_query"] = result.get("sql_query")
        query_params = result.get("query_parameters", [])
        
        # Replace doctor UUID with mapped integer DoctorId in parameters
        final_params = []
        for param in query_params:
            if param == doctor_uuid and doctor_id_mapped is not None:
                final_params.append(doctor_id_mapped)
            else:
                final_params.append(param)
        
        # Execute the query
        state["tool_results"] = execute_query(state["sql_query"], tuple(final_params))
        
        logger.info(f"SQL executed: {state['sql_query']}, Params: {final_params}, Results: {len(state['tool_results']) if state['tool_results'] else 0} rows")
        
    except Exception as e:
        logger.error(f"SQL generation/execution error: {e}")
        state["errors"].append(f"SQL execution failed: {e}")
        state["has_errors"] = True
        state["tool_results"] = []
    
    return state


def response_formatter_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Format the tool results into a natural language response.
    """
    logger.info("Response formatter processing results")
    
    system_prompt = config.get_system_prompt("response_formatter")
    
    # Resolve doctor names in results if needed
    results = state["tool_results"]
    if results and isinstance(results, list):
        for result in results:
            if isinstance(result, dict) and "DoctorID" in result and result["DoctorID"]:
                doctor_name = resolve_doctor_name_from_uuid(result["DoctorID"])
                if doctor_name:
                    result["DoctorName"] = doctor_name
    
    formatting_context = {
        "query_intent": state["query_intent"],
        "original_query": state["current_query"],
        "tool_results": results,
        "user_role": state["user_role"],
        "patient_context": state.get("patient_context"),
        "resolved_references": state["resolved_references"],
        "conversation_flow": state["conversation_memory"]["conversation_flow"]
    }
    
    prompt = f"""
{system_prompt}

Formatting context:
{json.dumps(formatting_context, indent=2, default=str)}

Format a helpful, natural response. Include relevant context and suggest follow-up actions if appropriate.
Respond in JSON format:
{{
    "formatted_response": "...",
    "response_metadata": {{}},
    "suggested_followups": []
}}
"""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        response = config.llm.invoke(messages)
        result = json.loads(response.content)
        
        state["formatted_response"] = result.get("formatted_response", "I processed your request successfully.")
        state["response_metadata"] = result.get("response_metadata", {})
        
        # Add processing metadata
        state["response_metadata"].update({
            "intent": state["query_intent"],
            "tool_used": state["selected_tool"],
            "has_errors": state["has_errors"],
            "context_resolved": bool(state["resolved_references"])
        })
        
        logger.info("Response formatted successfully")
        
    except Exception as e:
        logger.error(f"Response formatting error: {e}")
        state["errors"].append(f"Response formatting failed: {e}")
        state["has_errors"] = True
        # Fallback response
        if state["tool_results"]:
            state["formatted_response"] = f"I found {len(state['tool_results'])} results for your query."
        else:
            state["formatted_response"] = "I couldn't find any results for your query."
    
    return state


def memory_manager_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Enhanced memory manager with MCP context storage.
    
    Updates conversation memory and context for future reference using
    standardized MCP (Model Context Protocol) for better context preservation.
    """
    logger.info("Memory manager updating conversation state with MCP integration")
    
    session_id = state.get("session_id", "default_session")
    
    try:
        # Add current interaction to traditional memory
        state = add_to_conversation_memory(
            state,
            query=state["current_query"],
            result=state["tool_results"],
            flow_step=f"{state['query_intent']}:{state['selected_tool']}"
        )
        
        # Enhanced MCP context storage based on query intent
        if state["query_intent"] == "next_patient" and state["tool_results"]:
            # Store next patient information in MCP context
            next_appointment = state["tool_results"][0] if state["tool_results"] else None
            if next_appointment and isinstance(next_appointment, dict):
                
                # Add to traditional memory (backward compatibility)
                state["conversation_memory"]["implicit_references"]["current_patient"] = {
                    "name": next_appointment.get("PatientName"),
                    "id": next_appointment.get("PatientID"),
                    "appointment_id": next_appointment.get("AppointmentID"),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add to MCP context manager
                patient_context_id = mcp_context_manager.add_patient_context(
                    patient_name=next_appointment.get("PatientName", "Unknown"),
                    patient_id=str(next_appointment.get("PatientID", "")),
                    appointment_details={
                        "appointment_id": next_appointment.get("AppointmentID"),
                        "start_time": next_appointment.get("StartDateTime"),
                        "end_time": next_appointment.get("EndDateTime"),
                        "appointment_type": next_appointment.get("AppointmentType"),
                        "status": next_appointment.get("Status")
                    },
                    session_id=session_id
                )
                
                # Also add appointment context
                appointment_context_id = mcp_context_manager.add_appointment_context(
                    query_intent="next_patient",
                    appointments=state["tool_results"],
                    session_id=session_id
                )
                
                logger.info(f"Added MCP contexts: patient={patient_context_id}, appointment={appointment_context_id}")
                
                # Update patient context in state
                state = update_patient_context(
                    state,
                    patient_id=next_appointment.get("PatientID"),
                    patient_name=next_appointment.get("PatientName"),
                    appointment_id=next_appointment.get("AppointmentID"),
                    appointment_date=next_appointment.get("StartDateTime")
                )
        
        elif state["query_intent"] == "schedule" and state["tool_results"]:
            # Store schedule information in MCP context
            schedule_context_id = mcp_context_manager.add_schedule_context(
                schedule_data=state["tool_results"],
                date=datetime.now().strftime("%Y-%m-%d"),
                session_id=session_id
            )
            logger.info(f"Added MCP schedule context: {schedule_context_id}")
        
        elif state["query_intent"] == "patient_history" and state["tool_results"]:
            # Store patient history context
            if state["resolved_references"].get("patient_name"):
                patient_name = state["resolved_references"]["patient_name"]
                patient_context_id = mcp_context_manager.add_patient_context(
                    patient_name=patient_name,
                    patient_id=state["resolved_references"].get("patient_id", ""),
                    appointment_details={
                        "history_query": True,
                        "results_count": len(state["tool_results"]),
                        "query_time": datetime.now().isoformat()
                    },
                    session_id=session_id
                )
                logger.info(f"Added MCP patient history context: {patient_context_id}")
        
        # Update doctor context with recent activity
        if state.get("doctor_context") and state["tool_results"]:
            state["doctor_context"]["last_queried_date"] = datetime.now().strftime("%Y-%m-%d")
            if state["query_intent"] in ["schedule", "next_patient"]:
                state["doctor_context"]["current_appointments"] = state["tool_results"][:5]  # Keep recent appointments
        
        # Add response to message history for LangGraph
        if state["formatted_response"]:
            from langchain_core.messages import AIMessage
            state["messages"].append(AIMessage(content=state["formatted_response"]))
        
        # Log MCP context summary for debugging
        mcp_summary = mcp_context_manager.get_context_summary(session_id)
        logger.info(f"MCP context summary: {mcp_summary['total_items']} items, types: {mcp_summary['context_types']}")
        
        logger.info("Memory updated successfully with MCP integration")
        
    except Exception as e:
        logger.error(f"Memory management error: {e}")
        state["errors"].append(f"Memory update failed: {e}")
    
    return state
