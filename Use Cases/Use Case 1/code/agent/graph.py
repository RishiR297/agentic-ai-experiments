from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import get_user_input, query_appointments, generate_final_answer

# Create the graph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("input", get_user_input)
builder.add_node("appointments", query_appointments)
builder.add_node("final", generate_final_answer)
builder.add_edge("appointments", "final")
builder.add_edge("final", END)


# Define edges
builder.set_entry_point("input")
builder.add_edge("input", "appointments")
builder.add_edge("appointments", "final")
builder.add_edge("final", END)

# Compile
graph = builder.compile()
doctor_agent_executor = graph
