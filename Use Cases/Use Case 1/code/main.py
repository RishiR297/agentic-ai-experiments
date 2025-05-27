import os
from agent.graph import doctor_agent_executor  # your LangGraph flow
from agent.state import AgentState

def run_agent_loop():
    print("🤖 Doctor Appointment Agent (powered by Azure OpenAI)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        # Create initial state
        state = AgentState(user_input=user_input)

        try:
            # Run the LangGraph agent
            # Run the LangGraph agent
            result = doctor_agent_executor.invoke(state)
            # print("DEBUG result:", result)  # Optional, for debugging
            print("\n🧠 Agent:", result["final_answer"])
            print("-" * 40)
        except Exception as e:
            print("❌ Error:", e)

if __name__ == "__main__":
    # Check for required Azure env vars
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    missing = [var for var in required_vars if var not in os.environ]
    if missing:
        print(f"❌ Missing Azure OpenAI env vars: {missing}")
    else:
        run_agent_loop()
