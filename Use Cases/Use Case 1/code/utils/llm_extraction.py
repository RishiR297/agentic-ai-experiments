# ==============================================
# File: utils/llm_extraction.py
# Purpose: LLM-based field extraction utilities for appointment booking
#
# This module provides intelligent extraction of structured information
# from natural language user inputs. It uses Azure OpenAI's LLM to identify
# and parse appointment-related fields such as:
# - Doctor names (with normalization)
# - Patient names
# - Service types
# - Date and time preferences
# - Branch preferences
#
# The extraction is robust to various natural language formats and
# handles ambiguous inputs gracefully by returning None for unclear fields.
# ==============================================

import json
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from tools.doctor import process_doctor_name


def extract_fields_from_user_input(user_input: str, llm_basic) -> dict:
    """
    Extract structured fields from user input using LLM.
    Requires an LLM instance to avoid circular imports.
    """
    prompt = f"""
    You are a JSON extractor. Given a user's message, extract the following fields if present:
    - doctor_name
    - patient_name
    - branch_id
    - service_name
    - start_time
    - end_time
    - weekday (0 for Monday to 6 for Sunday)
    
    Special instructions:
    - If the user mentions "today", extract weekday as the current day of the week
    - Current date is {datetime.now().strftime('%A, %B %d, %Y')} (weekday {datetime.now().weekday()})

    Return a JSON object ONLY. No explanation.

    Example:
    {{
        "doctor_name": "Dr. Antonella",
        "weekday": 1
    }}

    User: "{user_input}"
    """

    try:
        response = llm_basic.invoke([HumanMessage(content=prompt)])
        extracted = json.loads(response.content)

        if isinstance(extracted, dict):
            print("Extracted fields:", extracted)

            # Normalize doctor name if present
            if "doctor_name" in extracted and extracted["doctor_name"]:
                extracted["doctor_name"] = process_doctor_name(extracted["doctor_name"], for_display=True)
                print(f"Normalized doctor name to: {extracted['doctor_name']}")

            # Handle "today" mentions - override any extracted weekday with current day
            if "today" in user_input.lower():
                current_weekday = datetime.now().weekday()
                extracted["weekday"] = current_weekday
                print(f"Detected 'today' in input, setting weekday to: {current_weekday}")

            # Handle weekday → ISO after (with safety check)
            if "weekday" in extracted and extracted["weekday"] is not None:
                now = datetime.now()
                weekday = extracted["weekday"]
                
                # If it's today's weekday, use today's date
                if weekday == now.weekday():
                    extracted["after"] = now.isoformat()
                    print("Using today's date for 'after':", now.date())
                else:
                    # Calculate next occurrence of this weekday
                    days_ahead = (weekday - now.weekday() + 7) % 7
                    if days_ahead == 0:
                        days_ahead = 7  # If it's the same weekday, go to next week
                    next_date = now + timedelta(days=days_ahead)
                    extracted["after"] = next_date.isoformat()
                    print("Interpreted weekday as:", next_date.date())

            return extracted
        return {}
    except Exception as e:
        print("extract_fields_from_user_input failed:", e)
        return {}
