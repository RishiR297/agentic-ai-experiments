# ==============================================
# File: main.py
# Purpose: FastAPI server to expose the doctor appointment agent via HTTP
# ==============================================

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from agent.graph import doctor_agent_executor
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# -----------------------------
# FastAPI App Setup
# -----------------------------
app = FastAPI(title="Doctor Appointment Agent", version="1.0")

# CORS (adjust allow_origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request Model
# -----------------------------
class AgentStateRequest(BaseModel):
    state: Dict[str, Any]

# -----------------------------
# Agent Invoke Endpoint
# -----------------------------
# Backend receive handler
@app.post("/invoke")
async def invoke(state: dict):
    if "user_input" not in state:
        print("Received state missing user_input")
        return {"error": "Missing 'user_input' in agent state"}
    
    result = doctor_agent_executor.invoke(state)
    return result


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# -----------------------------
# Script Entry Point
# -----------------------------
if __name__ == "__main__":
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
    missing = [var for var in required_vars if not os.environ.get(var)]

    if missing:
        print(f"Missing Azure OpenAI env vars: {missing}")
    else:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
