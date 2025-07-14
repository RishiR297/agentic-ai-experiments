"""
Multi-Turn Medical Assistant Processing Nodes

This module contains the core processing nodes for the LangGraph medical assistant
that handles context resolution, tool selection, SQL generation, and response formatting
with enhanced MCP (Model Context Protocol) integration for superior context preservation.
"""

import json
import logging
import re
from typing import Dict, Any
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph_agent.core.config import AgentConfig
from langgraph_agent.core.state import AgentState, update_patient_context, add_to_conversation_memory
from langgraph_agent.tools.database import (
    execute_query, resolve_doctor_uuid_to_id, resolve_doctor_name_from_uuid,
    get_next_appointment, get_patient_history, get_doctor_schedule
)
from langgraph_agent.tools.mcp_context_manager import mcp_context_manager

logger = logging.getLogger(__name__)


def clean_json_response(response_content: str) -> str:
    """
    Clean OpenAI response content to extract valid JSON.
    Handles responses wrapped in ```json code blocks and removes comments.
    """
    # Remove code block markers
    content = response_content.strip()
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    if content.startswith("```"):
        content = content[3:]   # Remove ```
    if content.endswith("```"):
        content = content[:-3]  # Remove closing ```
    
    # Remove single-line comments (// comments)
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove // comments but preserve the rest of the line
        if '//' in line:
            comment_pos = line.find('//')
            # Check if // is inside a string
            before_comment = line[:comment_pos]
            quote_count = before_comment.count('"') - before_comment.count('\\"')
            if quote_count % 2 == 0:  # Even number of quotes means // is outside string
                line = line[:comment_pos].rstrip()
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def preprocess_date_references(query: str) -> Dict[str, str]:
    """
    Preprocess date references like 'tomorrow', 'today', 'next week' etc.
    Returns a mapping of date references to actual dates.
    """
    date_mappings = {}
    current_date = datetime.now()
    
    query_lower = query.lower()
    
    # Handle today
    if 'today' in query_lower:
        date_mappings['today'] = current_date.strftime('%Y-%m-%d')
    
    # Handle tomorrow
    if 'tomorrow' in query_lower:
        tomorrow_date = current_date + timedelta(days=1)
        date_mappings['tomorrow'] = tomorrow_date.strftime('%Y-%m-%d')
    
    # Handle yesterday
    if 'yesterday' in query_lower:
        yesterday_date = current_date - timedelta(days=1)
        date_mappings['yesterday'] = yesterday_date.strftime('%Y-%m-%d')
    
    # Handle next week
    if 'next week' in query_lower:
        next_week_date = current_date + timedelta(days=7)
        date_mappings['next week'] = next_week_date.strftime('%Y-%m-%d')
    
    # Handle this week
    if 'this week' in query_lower:
        # Find the start of this week (Monday)
        days_since_monday = current_date.weekday()
        week_start = current_date - timedelta(days=days_since_monday)
        date_mappings['this week'] = week_start.strftime('%Y-%m-%d')
    
    return date_mappings


def safe_json_parse(response_content: str, node_name: str) -> Dict[str, Any]:
    """
    Safely parse JSON response with cleaning and error handling.
    """
    try:
        cleaned_content = clean_json_response(response_content)
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {node_name}: {e}")
        logger.error(f"Response content was: {response_content}")
        
        # Try to extract JSON pattern manually
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, cleaned_content, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except:
                pass
        
        # Return a safe default
        return {
            "error": f"Failed to parse JSON in {node_name}",
            "raw_content": response_content[:200] + "..." if len(response_content) > 200 else response_content
        }


def context_resolver_node(state: AgentState, config: AgentConfig) -> AgentState:
    """
    Enhanced context resolver with MCP (Model Context Protocol) integration.
    
    This node analyzes the current query against conversation memory and 
    existing context to resolve references like "next patient", "she", etc.
    Enhanced with MCP for standardized context management.
    """
    logger.info(f"Context resolver processing: {state['current_query']}")
    
    session_id = state.get("session_id", "default_session")
    
    # Preprocess date references before other processing
    date_mappings = preprocess_date_references(state["current_query"])
    logger.info(f"Date mappings found: {date_mappings}")
    
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
    
    # Enhanced context information including MCP context and date mappings
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
        "mcp_resolved_references": mcp_resolved_refs,
        "date_mappings": date_mappings,
        "current_date": datetime.now().strftime('%Y-%m-%d')
    }
    
    prompt = f"""
{system_prompt}

Current context (enhanced with MCP and date preprocessing):
{json.dumps(context_info, indent=2, default=str)}

IMPORTANT DATE AND TIME PROCESSING:
- Current date: {datetime.now().strftime('%Y-%m-%d')}
- Date mappings: {date_mappings}
- When the query mentions "tomorrow", use date: {date_mappings.get('tomorrow', 'N/A')}
- When the query mentions "today", use date: {date_mappings.get('today', 'N/A')}

TIME-SPECIFIC QUERY DETECTION:
- If query contains specific times (e.g., "2 PM", "10:30", "at 3", "9 AM"), classify as "time_specific_lookup"
- If query asks about "next patient" or "who's next", classify as "next_patient"
- If query asks about general schedule ("my appointments", "schedule today"), classify as "schedule"
- Time patterns to detect: [time][AM/PM], [hour]:[minute], "at [time]", "who's at [time]"

Analyze the query "{state['current_query']}" and provide:
1. query_intent: Choose from (time_specific_lookup, next_patient, schedule, patient_history, availability)
   - Use "time_specific_lookup" for queries with specific times like "Who's at 2 PM?"
   - Use "next_patient" only for queries asking about the chronologically next patient
   - Use "schedule" for general schedule viewing requests
2. resolved_references: Dictionary mapping implicit references to explicit values (MUST include actual dates for temporal references)
3. context_updates: Any updates needed to patient or doctor context
4. reasoning: Explain why this intent was chosen and how time references were interpreted

Use MCP-resolved references when available for better accuracy.
Use the preprocessed date mappings for temporal references like "tomorrow", "today", etc.

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
            result = safe_json_parse(response.content, "context resolver")
        except Exception as je:
            logger.error(f"Context resolution error: {je}")
            # Use fallback logic if JSON parsing fails
            raise je
        
        # Update state with resolved information
        state["query_intent"] = result.get("query_intent")
        
        # Merge date mappings, MCP-resolved references, and LLM-resolved references
        llm_resolved = result.get("resolved_references", {})
        final_resolved = {**date_mappings, **mcp_resolved_refs, **llm_resolved}
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

TOOL SELECTION GUIDELINES:
- For "time_specific_lookup" intent: Use "schedule_query" with specific time filtering
- For "next_patient" intent: Use "appointment_lookup" to find chronologically next appointment
- For "schedule" intent: Use "schedule_query" for general schedule viewing
- For "patient_history" intent: Use "patient_lookup" or "history_query"

TIME-SPECIFIC HANDLING:
- If resolved_references contains specific times (e.g., "2 PM", "14:00"), this indicates a time-specific lookup
- Generate parameters that will filter by the specific time mentioned
- For time-specific queries, include both date and time constraints in parameters

Select the best tool and parameters. Respond in JSON format:
{{
    "selected_tool": "tool_name",
    "tool_parameters": {{}},
    "reasoning": "Explain tool choice and parameter generation, especially for time-specific queries"
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
            result = safe_json_parse(response.content, "tool selector")
        except Exception as je:
            logger.error(f"Tool selection error: {je}")
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
    ALL queries are generated by LLM for proper evaluation of SQL generation accuracy.
    """
    logger.info(f"SQL generator processing tool: {state['selected_tool']}")
    
    system_prompt = config.get_system_prompt("sql_generator")
    
    # Map doctor UUID to integer DoctorId if needed
    doctor_uuid = state.get("doctor_id")
    doctor_id_mapped = None
    if doctor_uuid:
        # Check if doctor_id is already an integer (direct DoctorId)
        try:
            # If it's already an integer, use it directly
            doctor_id_mapped = int(doctor_uuid)
            logger.info(f"Doctor ID {doctor_uuid} is already an integer DoctorId: {doctor_id_mapped}")
        except ValueError:
            # If it's not an integer, try to resolve UUID to DoctorId
            doctor_id_mapped = resolve_doctor_uuid_to_id(doctor_uuid)
            logger.info(f"Mapped doctor UUID {doctor_uuid} to DoctorId {doctor_id_mapped}")
    
    generation_context = {
        "selected_tool": state["selected_tool"],
        "tool_parameters": state["tool_parameters"],
        "resolved_references": state["resolved_references"],
        "query_intent": state["query_intent"],
        "original_query": state["current_query"],
        "doctor_uuid": doctor_uuid,
        "doctor_id_mapped": doctor_id_mapped,
        "patient_context": state.get("patient_context"),
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_conversion_guide": {
            "12_hour_to_24_hour": {
                "12 AM": "00:00", "1 AM": "01:00", "2 AM": "02:00", "3 AM": "03:00",
                "4 AM": "04:00", "5 AM": "05:00", "6 AM": "06:00", "7 AM": "07:00", 
                "8 AM": "08:00", "9 AM": "09:00", "10 AM": "10:00", "11 AM": "11:00",
                "12 PM": "12:00", "1 PM": "13:00", "2 PM": "14:00", "3 PM": "15:00",
                "4 PM": "16:00", "5 PM": "17:00", "6 PM": "18:00", "7 PM": "19:00",
                "8 PM": "20:00", "9 PM": "21:00", "10 PM": "22:00", "11 PM": "23:00"
            },
            "time_patterns": ["at", "PM", "AM", ":", "o'clock", "sharp", "exactly"]
        },
        "resolved_time_references": state.get("resolved_references", {})
    }
    
    prompt = f"""
{system_prompt}

CRITICAL: Generate SQL queries for ALL requests. Do NOT use hardcoded functions.

DATABASE SCHEMA INFORMATION:
- View_Appointments table: AppointmentId, PatientId, PatientName, DoctorId, DoctorName, BranchId, BranchName, CategoryId, CategoryName, ServiceId, ServiceName, MachineId, MachineName, StartDateTime, EndDateTime, StatusId, Status
- COR_Doctor table: UserId (UUID), SpecialtyId, FirstName, LastName, DisplayName, Phone, Email, DefaultBranchId, IsActive
- Current DateTime: {generation_context['current_datetime']}
- Doctor UUID {doctor_uuid} maps to DoctorId {doctor_id_mapped}

QUERY TYPE PATTERNS:
1. NEXT_PATIENT: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND StartDateTime > datetime('now') ORDER BY StartDateTime ASC LIMIT 1"
2. TIME_SPECIFIC_LOOKUP: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND DATE(StartDateTime) = ? AND strftime('%H:%M', StartDateTime) = ? ORDER BY StartDateTime"
3. PATIENT_HISTORY: "SELECT * FROM View_Appointments WHERE PatientName LIKE ? OR PatientId = ? ORDER BY StartDateTime DESC"
4. DOCTOR_SCHEDULE: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND DATE(StartDateTime) = ? ORDER BY StartDateTime"
5. TODAY_APPOINTMENTS: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND DATE(StartDateTime) = DATE('now') ORDER BY StartDateTime"
6. TOMORROW_APPOINTMENTS: "SELECT * FROM View_Appointments WHERE DoctorId = ? AND DATE(StartDateTime) = DATE('now', '+1 day') ORDER BY StartDateTime"

TIME-SPECIFIC QUERY HANDLING:
- For queries like "Who's at 2 PM?", use TIME_SPECIFIC_LOOKUP pattern
- Convert time references: "2 PM" → "14:00", "9 AM" → "09:00", "10:30" → "10:30"
- Time-specific queries should filter by BOTH date AND time
- Example: "Who's at 2 PM today?" → WHERE DoctorId = ? AND DATE(StartDateTime) = DATE('now') AND strftime('%H:%M', StartDateTime) = '14:00'

IMPORTANT MAPPINGS:
- Use DoctorId {doctor_id_mapped} for doctor queries (NOT the UUID)
- For date references, use the resolved dates from context: {generation_context.get('resolved_references', {})}
- Patient references should use LIKE '%name%' for partial matching
- Time references should be converted to 24-hour format (HH:MM)

Generation context:
{json.dumps(generation_context, indent=2, default=str)}

Based on the query intent "{state['query_intent']}" and original query "{state['current_query']}", generate the appropriate SQL query.

Respond in JSON format:
{{
    "sql_query": "SELECT ...",
    "query_parameters": [],
    "reasoning": "Explanation of why this query matches the intent and handles the user's request",
    "query_type": "next_patient|patient_history|schedule|general"
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
            result = safe_json_parse(response.content, "SQL generator")
        except Exception as je:
            logger.error(f"JSON decode error: {je}")
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
                    result = {
                        "sql_query": sql_line, 
                        "query_parameters": [],
                        "reasoning": "Extracted from non-JSON response",
                        "query_type": "extracted"
                    }
                else:
                    raise je
            else:
                raise je
        
        state["sql_query"] = result.get("sql_query")
        query_params = result.get("query_parameters", [])
        query_type = result.get("query_type", "unknown")
        reasoning = result.get("reasoning", "No reasoning provided")
        
        # Replace doctor UUID with mapped integer DoctorId in parameters
        final_params = []
        for param in query_params:
            if param == doctor_uuid and doctor_id_mapped is not None:
                final_params.append(doctor_id_mapped)
            else:
                final_params.append(param)
        
        # Execute the query and add observability metadata
        state["tool_results"] = execute_query(state["sql_query"], tuple(final_params))
        
        # Enhanced SQL logging for observability (to stdout)
        print("=" * 80)
        print("🗄️  LLM-GENERATED SQL EVALUATION - LangGraph Medical Agent")
        print("=" * 80)
        print(f"📊 Tool: {state.get('selected_tool', 'unknown')}")
        print(f"🎯 Query Intent: {state.get('query_intent', 'unknown')}")
        print(f"🔐 User Role: {state.get('user_role', 'unknown')}")
        print(f"👤 Doctor ID: {state.get('doctor_id', 'N/A')} -> {doctor_id_mapped}")
        print(f"🆔 Session: {state.get('session_id', 'N/A')}")
        print(f"📝 Original Query: {state.get('current_query', 'N/A')}")
        print(f"🧠 LLM Reasoning: {reasoning}")
        print(f"� Query Type: {query_type}")
        print(f"�🗄️  Generated SQL: {state['sql_query']}")
        print(f"📊 Parameters: {final_params}")
        print(f"📈 Result Count: {len(state['tool_results']) if state['tool_results'] else 0} rows")
        print(f"🎯 Context References: {state.get('resolved_references', {})}")
        print(f"⏰ Timestamp: {datetime.now().isoformat()}")
        print("=" * 80)
        
        # Add SQL metadata to state for response inclusion
        state["sql_metadata"] = {
            "raw_query": state["sql_query"],
            "parameters": final_params,
            "parameter_mapping": {
                "original_params": query_params,
                "mapped_params": final_params,
                "doctor_uuid_mapping": f"{doctor_uuid} -> {doctor_id_mapped}" if doctor_uuid and doctor_id_mapped else None
            },
            "result_count": len(state["tool_results"]) if state["tool_results"] else 0,
            "generated_at": datetime.now().isoformat(),
            "tool_name": state.get("selected_tool", "unknown"),
            "query_type": query_type,
            "llm_reasoning": reasoning,
            "execution_method": "llm_generated_sql",
            "user_context": {
                "role": state.get("user_role", "unknown"),
                "doctor_id": state.get("doctor_id"),
                "session_id": state.get("session_id")
            },
            "query_evaluation": {
                "intent": state["query_intent"],
                "original_query": state["current_query"],
                "resolved_references": state["resolved_references"],
                "context_used": bool(state.get("patient_context") or state.get("doctor_context"))
            }
        }
        
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

CRITICAL INSTRUCTION: 
- When formatting a response for a specific date query (like "tomorrow" or "today"), ONLY use the tool_results data.
- Do NOT mix cached context data with current query results.
- The tool_results contain the EXACT data requested for the specific query.
- Original query: "{state['current_query']}"
- Query intent: {state['query_intent']}
- Resolved date references: {state.get('resolved_references', {}).get('tomorrow', 'N/A')} / {state.get('resolved_references', {}).get('today', 'N/A')}

Format a helpful, natural response based ONLY on the tool_results. Include relevant context and suggest follow-up actions if appropriate.
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
        logger.info(f"Response formatter LLM output: {response.content}")
        result = safe_json_parse(response.content, "response formatter")
        logger.info(f"Parsed result: {result}")
        
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
