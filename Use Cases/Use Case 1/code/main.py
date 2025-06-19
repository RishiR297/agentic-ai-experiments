# ==============================================
# File: main.py
# Purpose: Entry point to run the doctor appointment agent loop
# ==============================================

# -----------------------------
# Imports
# -----------------------------
import os
from agent.graph import doctor_agent_executor  # LangGraph flow
from agent.state import AgentState


# -----------------------------
# Function: run_agent_loop
# -----------------------------
def run_agent_loop():
    """
    Runs a command-line loop for chatting with the doctor appointment agent.
    Accepts user input, feeds it into the agent, and prints the response.
    """
    print("🤖 Doctor Appointment Agent (powered by Azure OpenAI)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        # Construct AgentState
        state: AgentState = {
            "user_input": user_input
        }

        try:
            # Run LangGraph agent executor
            result = doctor_agent_executor.invoke(state)
            print("\n🧠 Agent:", result.get("final_answer", "No answer generated."))
            print("-" * 40)
        except Exception as e:
            print("❌ Error:", e)


# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        print(f"❌ Missing Azure OpenAI env vars: {missing}")
    else:
        run_agent_loop()