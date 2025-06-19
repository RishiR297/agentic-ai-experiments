# agent/graph.py
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from agent.nodes import (
    planner_node,
    route_node,
    call_tool_node,
    generate_final_answer,
)

path_fn = RunnableLambda(lambda state: state["next"])
path_fn.name = "router_decision" 

# -----------------------------
# Step 1: Build the graph
# -----------------------------
graph = StateGraph(dict)

# Define all nodes
graph.add_node("planner", planner_node)              # Uses LLM + MCP to choose tool
graph.add_node("router", route_node)                 # Checks for tool_calls in response
graph.add_node("tool", call_tool_node)               # Executes selected tool
graph.add_node("answer", generate_final_answer)      # Generates final answer

# Entry and flow control
graph.set_entry_point("planner")
graph.add_edge("planner", "router")

# Conditional transition from router
graph.add_conditional_edges(
    source="router",
    path=path_fn,
    path_map={
        "tool": "tool",
        "answer": "answer"
    }
)




# After executing tool, always generate final answer
graph.add_edge("tool", "answer")

# Final node
graph.set_finish_point("answer")

# -----------------------------
# Step 2: Compile the graph into a runnable executor
# -----------------------------
doctor_agent_executor = graph.compile()
