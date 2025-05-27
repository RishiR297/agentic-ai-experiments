from datetime import datetime
from agent.tools.appointment import get_appointments

today = "2025-03-11"
print("Running standalone appointment test...")
result = get_appointments.invoke({"doctor_name": "Antonella", "after": today})
print(result)

