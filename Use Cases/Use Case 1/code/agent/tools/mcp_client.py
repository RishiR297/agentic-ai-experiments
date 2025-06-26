# # ==============================================
# # File: mcp_client.py
# # Purpose: 
# # ==============================================

import requests

MCP_SERVER_URL = "http://localhost:8001/tools"

def call_mcp_tool(tool_name: str, input_dict: dict) -> str:
    """
    Calls an MCP tool via HTTP POST and returns the result string.
    """
    try:
        response = requests.post(f"{MCP_SERVER_URL}/{tool_name}", json=input_dict)
        response.raise_for_status()
        result = response.json()
        return result if isinstance(result, str) else str(result)
    except Exception as e:
        return f"❌ MCP tool '{tool_name}' failed: {e}"
