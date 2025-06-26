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
    respond_naturally_node  # ✅ New node added
)

# Helper for routing
path_fn = RunnableLambda(lambda state: state["next"])
path_fn.name = "router_decision"

# -----------------------------
# Step 1: Build the graph
# -----------------------------
graph = StateGraph(dict)

# Define all nodes
graph.add_node("welcome", welcome_node)
graph.add_node("planner", planner_node)                     # LLM decides next action
graph.add_node("router", route_node)                        # Chooses branch: ask, tool, or finish
graph.add_node("respond", respond_naturally_node)           # ✅ Gives natural response before tool
graph.add_node("tool", call_tool_node)                      # Executes the tool
graph.add_node("answer", generate_final_answer)             # Formats final output
graph.add_node("ask_missing_info", ask_for_missing_fields_node)

# Entry point
graph.set_entry_point("welcome")
graph.add_edge("welcome", "planner")
graph.add_edge("planner", "router")

# Router logic: decide next step
graph.add_conditional_edges(
    source="router",
    path=path_fn,
    path_map={
        "tool": "tool",                # ✅ New: respond naturally before calling tool
        "answer": "answer",
        "ask_missing_info": "ask_missing_info"
    }
)

# ✅ New edge: respond first → then tool
graph.add_edge("tool", "respond")    # ✅ goes from tool → respond

graph.add_edge("respond", "answer")  # ✅ This is the correct final transition

# Also go to answer after asking for missing fields
graph.add_edge("ask_missing_info", "answer")

# Finish node
graph.set_finish_point("answer")

# -----------------------------
# Step 2: Compile the graph
# -----------------------------
doctor_agent_executor = graph.compile()
