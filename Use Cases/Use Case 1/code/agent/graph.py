# agent/graph.py

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from agent.nodes import (
    welcome_node,
    planner_node,
    route_node,
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
graph.add_node("router", route_node)
graph.add_node("tool", call_tool_node)
graph.add_node("respond", respond_naturally_node)
graph.add_node("ask_missing_info", ask_for_missing_fields_node)
graph.add_node("answer", generate_final_answer)

# 2. Define entry
graph.set_entry_point("welcome")
graph.add_edge("welcome", "planner")
graph.add_edge("planner", "router")

# 3. Conditional routing from router
graph.add_conditional_edges(
    source="router",
    path=path_fn,
    path_map={
        "tool": "tool",
        "ask_missing_info": "ask_missing_info",
        "answer": "answer"
    }
)

# 4. Tool flows to natural response
graph.add_edge("tool", "respond")
graph.add_edge("respond", "answer")

# 5. Ask_missing_info also ends at answer
graph.add_edge("ask_missing_info", "answer")

# 6. Final output
graph.set_finish_point("answer")

# -----------------------------
# Step 2: Compile the graph
# -----------------------------
doctor_agent_executor = graph.compile()
