# agent/graph.py

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from agent.nodes import (
    welcome_node,
    planner_node,
    call_tool_node,
    generate_final_answer,
    ask_for_missing_fields_node,
    respond_naturally_node
)

# -----------------------------
# Router Function
# -----------------------------
def route_decision(state: dict) -> str:
    """
    Determine the next step based on the planner output.
    """
    return state.get("next", "answer")  # Fallback to 'answer'

path_fn = RunnableLambda(route_decision)
path_fn.name = "router_decision"

# -----------------------------
# Step 1: Build the graph
# -----------------------------
graph = StateGraph(dict)

# 1. Add nodes
graph.add_node("welcome", welcome_node)
graph.add_node("planner", planner_node)
graph.add_node("tool", call_tool_node)
graph.add_node("respond", respond_naturally_node)
graph.add_node("ask_missing_info", ask_for_missing_fields_node)
graph.add_node("answer", generate_final_answer)

# 2. Define entry
graph.set_entry_point("planner")

# 3. Conditional routing from planner
graph.add_conditional_edges(
    source="planner",
    path=path_fn,
    path_map={
        "welcome": "welcome",
        "tool": "tool",
        "ask_missing_info": "ask_missing_info", 
        "answer": "answer"
    }
)

# 4. Welcome goes to answer (for displaying the welcome message)
graph.add_edge("welcome", "answer")

# 5. Tool flows to natural response  
graph.add_edge("tool", "respond")
graph.add_edge("respond", "answer")

# 6. Ask_missing_info also ends at answer
graph.add_edge("ask_missing_info", "answer")

# 7. Final output
graph.set_finish_point("answer")

# -----------------------------
# Step 2: Compile the graph
# -----------------------------
doctor_agent_executor = graph.compile()
