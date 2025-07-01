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
BACKEND_URL = "http://localhost:8000/invoke"  # Make sure FastAPI server is running here

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

if not st.session_state["agent_state"].get("chat_history"):
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
for msg in st.session_state["agent_state"].get("chat_history", []):
    with st.chat_message("user" if msg["type"] == "human" else "assistant"):
        st.markdown(msg["content"])

user_input = st.chat_input("How can I help you today?")
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
# Run Agent if Input
# -----------------------------
if user_input:
    with st.spinner("Running LangGraph agent via backend..."):
        try:
            state = st.session_state["agent_state"]
            state["user_input"] = user_input

            # 🔁 POST to FastAPI
            response = requests.post(BACKEND_URL, json=state)
            response.raise_for_status()
            result = response.json()

            st.session_state["agent_state"] = result
            result = serialize_result(result)

            final_answer = result.get("final_answer", "🤖 No answer produced.")

            pass

            # Show assistant reply
            with st.chat_message("assistant"):
                st.markdown(final_answer)

            if result.get("appointments_output"):
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

# -----------------------------
# Reset Button
# -----------------------------
if st.button("🧹 Reset Agent"):
    st.session_state["agent_state"] = {
        "chat_history": [],
        "identity": "user_001"
    }
    st.rerun()
    st.success("Agent memory reset!")
