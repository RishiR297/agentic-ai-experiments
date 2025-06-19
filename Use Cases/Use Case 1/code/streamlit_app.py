# ==============================================
# File: streamlit_app.py
# Purpose: Streamlit UI to demo LangGraph appointment agent (MCP style)
# ==============================================

import streamlit as st
from agent.graph import doctor_agent_executor
from pydantic import BaseModel
from collections.abc import Mapping
import json
import traceback

st.set_page_config(page_title="Doctor Appointment Agent", page_icon="🤖")

st.title("🤖 Doctor Appointment Agent")
st.markdown("Ask anything about appointments, doctors, or bookings.")

# --- Text input ---
user_input = st.text_input("What would you like to do?", placeholder="e.g., I want to book with Dr. Patel next Monday")


# --- Recursive serializer with error trap ---
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
            return str(result)  # fallback for unserializable types
    except Exception as e:
        print("❌ Serialization error:", e)
        print("Offending object type:", type(result))
        traceback.print_exc()
        raise


# --- Run Agent Button ---
if st.button("Run Agent") and user_input:
    with st.spinner("Running LangGraph agent..."):

        try:
            # 👇 Prepare initial state (you might have a more elaborate one)
            state = {"user_input": user_input}

            # 💉 Trap agent invoke & serialization
            result = doctor_agent_executor.invoke(state)

            # 🧪 Log raw agent output BEFORE serialization
            print("🧪 Raw agent output before serialization:")
            print(result)
            print("🧪 Type of result:", type(result))

            result = serialize_result(result)

        except Exception as e:
            st.error(f"Agent failed: {e}")
            st.text("🔍 Traceback:")
            st.text(traceback.format_exc())  # ← full traceback in UI
            raise  # still shows in dev console

        try:
            # --- Display Final Answer ---
            st.subheader("🧠 Final Answer")
            st.success(result.get("final_answer", "No answer produced."))

            # --- Tool Output Display ---
            if result.get("appointments_output"):
                st.subheader("📅 Appointment Query Result")
                st.code(json.dumps(result["appointments_output"], indent=2) if isinstance(result["appointments_output"], (dict, list)) else str(result["appointments_output"]), language="json")


            if result.get("booking_confirmation"):
                st.subheader("✅ Booking Confirmation")
                st.code(json.dumps(result["booking_confirmation"], indent=2) if isinstance(result["booking_confirmation"], (dict, list)) else str(result["booking_confirmation"]), language="json")

            # --- Optional Debug Output ---
            # st.subheader("🛠 Final State")
            # st.json(result)

        except Exception as e:
            st.error(f"Agent display logic failed: {e}")
            st.text("🔍 Traceback:")
            st.text(traceback.format_exc())
