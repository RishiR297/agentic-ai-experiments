# ==============================================
# File: streamlit_app.py
# Purpose: Streamlit UI to demo LangGraph appointment agent (via FastAPI MCP server)
# ==============================================

import streamlit as st
import json
import requests
from collections.abc import Mapping
from pydantic import BaseModel
import traceback

# -----------------------------
# Config
# -----------------------------
BACKEND_URL = "http://localhost:8003/invoke"  # Make sure FastAPI server is running here

st.set_page_config(page_title="Doctor Appointment Agent", page_icon="🤖")

# -----------------------------
# Init State
# -----------------------------
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {
        "chat_history": [],
        "identity": "user_001"
    }

st.title("🤖 Doctor Appointment Agent")

# Display current role if authenticated
current_role = st.session_state["agent_state"].get("user_role")
if current_role:
    if current_role == "doctor":
        doctor_name = st.session_state["agent_state"].get("doctor_authenticated_name", "Unknown")
        st.success(f"👨‍⚕️ **Logged in as:** Dr. {doctor_name}")
    else:
        st.info(f"👤 **Current Mode:** Patient")
else:
    st.warning("🔍 **Please identify yourself** - say 'patient' or 'doctor' to get started")

st.markdown("Ask anything about appointments, doctors, or bookings.")

def safe_post(url, payload, retries=3):
    for attempt in range(retries):
        try:
            return requests.post(url, json=payload, timeout=5)
        except requests.exceptions.ConnectionError as e:
            if attempt < retries - 1:
                import time
                time.sleep(1)  # wait a second before retry
            else:
                raise e

# Initialize on first load only
if not st.session_state["agent_state"].get("chat_history") and not st.session_state["agent_state"].get("user_role"):
    try:
        with st.spinner("Initializing..."):
            response = safe_post(BACKEND_URL, st.session_state["agent_state"])

            response.raise_for_status()
            result = response.json()

            # ✅ Ensure chat_history exists
            if "chat_history" not in result:
                result["chat_history"] = []

            # ✅ Add welcome message if available
            if result.get("final_answer"):
                result["chat_history"].append({
                    "type": "ai",
                    "content": result["final_answer"]
                })

            # ✅ Save back to session
            st.session_state["agent_state"] = result

    except Exception as e:
        st.error("Failed to load welcome message")
        st.text(traceback.format_exc())


# -----------------------------
# Display Chat History
# -----------------------------
st.markdown("---")
for msg in st.session_state["agent_state"].get("chat_history", []):
    with st.chat_message("user" if msg["type"] == "human" else "assistant"):
        st.markdown(msg["content"])

# Dynamic placeholder based on current role
current_role = st.session_state["agent_state"].get("user_role")
if current_role == "patient":
    placeholder = "Ask about doctors, book appointments, or check availability..."
elif current_role == "doctor":
    placeholder = "Check your schedule, view appointments, or manage availability..."
else:
    placeholder = "Start by saying 'I am a patient' or 'I am a doctor'..."

user_input = st.chat_input(placeholder)
if user_input:
    st.chat_message("user").markdown(user_input)  # Show immediately


# -----------------------------
# Serializer
# -----------------------------
def serialize_result(result):
    try:
        if isinstance(result, BaseModel):
            return result.model_dump()
        elif isinstance(result, Mapping):
            return {k: serialize_result(v) for k, v in result.items()}
        elif isinstance(result, list):
            return [serialize_result(v) for v in result]
        elif isinstance(result, (str, int, float, type(None), bool)):
            return result
        else:
            return str(result)
    except Exception as e:
        print("Serialization error:", e)
        print("Offending object type:", type(result))
        traceback.print_exc()
        raise

# -----------------------------
# Helper function to detect role switch
# -----------------------------
def detect_role_switch(user_input, current_state):
    """Detect if user is trying to switch roles"""
    user_lower = user_input.lower().strip()
    current_role = current_state.get("user_role")
    
    print(f"[DEBUG ROLE SWITCH] Input: '{user_input}', Current role: {current_role}")
    
    # More specific role switch phrases to avoid false positives
    role_switch_phrases = [
        "i am a doctor", "i'm a doctor", "i am doctor", 
        "i am a patient", "i'm a patient", "i am patient",
        "switch to doctor", "switch to patient", "change role",
        "log in as doctor", "login as doctor"
    ]
    
    # Check if input matches role switch patterns
    for phrase in role_switch_phrases:
        if phrase in user_lower:
            print(f"[DEBUG ROLE SWITCH] Found phrase: '{phrase}'")
            # If currently a patient and trying to be doctor, or vice versa
            if current_role == "patient" and ("doctor" in user_lower):
                print(f"[DEBUG ROLE SWITCH] Patient->Doctor switch detected")
                return True
            elif current_role == "doctor" and ("patient" in user_lower):
                print(f"[DEBUG ROLE SWITCH] Doctor->Patient switch detected")
                return True
            elif not current_role:  # No role set yet
                print(f"[DEBUG ROLE SWITCH] No current role, letting normal flow handle")
                return False  # Let normal flow handle it
    
    # Only detect role switch for doctor login if there's already a different role
    if current_role == "patient" and user_lower.startswith("doctor ") and " id " in user_lower:
        print(f"[DEBUG ROLE SWITCH] Patient->Doctor login detected")
        return True
    
    print(f"[DEBUG ROLE SWITCH] No role switch detected")
    return False

# -----------------------------
# Run Agent if Input
# -----------------------------
if user_input:
    # Check for role switch
    if detect_role_switch(user_input, st.session_state["agent_state"]):
        # Clear everything and start fresh
        st.session_state["agent_state"] = {
            "chat_history": [],
            "identity": "user_001"
        }
        # Show role switch message
        with st.chat_message("assistant"):
            st.markdown("🔄 **Role switch detected!** Starting fresh. Please identify yourself.")
        st.rerun()
    
    with st.spinner("Running LangGraph agent via backend..."):
        try:
            state = st.session_state["agent_state"]
            state["user_input"] = user_input

            # 🔁 POST to FastAPI
            response = requests.post(BACKEND_URL, json=state)
            response.raise_for_status()
            result = response.json()

            # Ensure chat_history exists
            if "chat_history" not in result:
                result["chat_history"] = []

            # Add user message to chat history if not already there
            user_msg = {"type": "human", "content": user_input}
            if not result["chat_history"] or result["chat_history"][-1] != user_msg:
                result["chat_history"].append(user_msg)

            # Add assistant response to chat history
            final_answer = result.get("final_answer", "🤖 No answer produced.")
            ai_msg = {"type": "ai", "content": final_answer}
            result["chat_history"].append(ai_msg)

            # Update session state
            st.session_state["agent_state"] = result
            result = serialize_result(result)

            # Show assistant reply
            with st.chat_message("assistant"):
                st.markdown(final_answer)

            if result.get("appointments_output"):
                if st.session_state["agent_state"].get("user_role") == "doctor":
                    st.subheader("👨‍⚕️ Your Appointments")
                else:
                    st.subheader("📅 Appointment Query Result")
                st.code(
                    json.dumps(result["appointments_output"], indent=2)
                    if isinstance(result["appointments_output"], (dict, list))
                    else str(result["appointments_output"]),
                    language="json",
                )

            if result.get("booking_confirmation"):
                st.subheader("✅ Booking Confirmation")
                st.code(
                    json.dumps(result["booking_confirmation"], indent=2)
                    if isinstance(result["booking_confirmation"], (dict, list))
                    else str(result["booking_confirmation"]),
                    language="json",
                )

        except Exception as e:
            st.error(f"Agent failed: {e}")
            st.text("🔍 Traceback:")
            st.text(traceback.format_exc())
        
        # Force rerun to update the chat history display
        st.rerun()

# -----------------------------
# Control Buttons
# -----------------------------
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state["agent_state"] = {
            "chat_history": [],
            "identity": "user_001"
        }
        st.rerun()

with col2:
    current_role = st.session_state["agent_state"].get("user_role")
    if current_role:
        button_text = f"🔄 Switch from {current_role.title()}"
    else:
        button_text = "🔄 Start Fresh"
    
    if st.button(button_text):
        # Complete reset for role switching
        st.session_state["agent_state"] = {
            "chat_history": [],
            "identity": "user_001"
        }
        st.rerun()

with col3:
    if st.button("ℹ️ Help"):
        help_text = """
        **How to use this agent:**
        
        **As a Patient:**
        - Say "I am a patient" or just "patient"
        - Ask to book appointments, find doctors, or check availability
        
        **As a Doctor:**
        - Say "I am a doctor" or "doctor"
        - Provide your name to log in
        - View your schedule and appointments
        
        **Tips:**
        - You can specify doctors by name or ask for doctors by service
        - The agent will guide you through the booking process step by step
        - Use the buttons below to clear chat or switch roles
        """
        st.info(help_text)

# -----------------------------
# Debug Info (Optional)
# -----------------------------
if st.checkbox("🔧 Show Debug Info"):
    st.subheader("Current Agent State")
    debug_state = st.session_state["agent_state"].copy()
    # Remove chat history for cleaner debug view
    if "chat_history" in debug_state:
        debug_state["chat_history"] = f"[{len(debug_state['chat_history'])} messages]"
    st.json(debug_state)
