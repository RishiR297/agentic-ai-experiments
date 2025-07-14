# LangGraph Medical Assistant API & Setup Guide

## 🚀 Overview
A multi-turn conversational medical assistant built with LangGraph, featuring advanced context preservation, LLM-powered SQL, and a RESTful API for seamless integration.

---

## 📦 Project Structure (Key Files)
```
code/
├── agent/                  # LangGraph agent logic (state, nodes, tools)
├── main.py                 # Entry point
├── streamlit_app.py        # (Optional) Streamlit UI
├── requirements.txt        # Python dependencies
├── package.json            # NPM scripts (optional)
├── .env                    # Environment variables
├── src/
│   ├── api/
│   │   └── langgraph_server.py   # FastAPI API server
│   └── langgraph_agent/         # Core agent logic
└── db/                    # SQLite database
```

---

## ⚡ Quickstart

### 1. Prerequisites
- Python 3.8+
- SQLite database (with `View_Appointments`, `COR_Doctor` tables)
- Azure OpenAI API access

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file:
```
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### 4. Database
Ensure your SQLite DB is at `./db/output.db` with the required schema.

---

## 🌐 Running the API

### Local Development
```bash
cd code
python -m uvicorn src.api.langgraph_server:app --host 0.0.0.0 --port 8502 --reload
```
- Access docs: http://localhost:8502/docs

### Public Access (ngrok)
1. Start API server (see above)
2. In a new terminal:
```bash
ngrok http 8502
```
3. Share the HTTPS URL (e.g., `https://<your-ngrok-domain>`) with your team.

#### Testing
```bash
curl -X GET "https://<your-ngrok-domain>/health"
curl -X POST "https://<your-ngrok-domain>/chat" \
  -H "X-User-Role: doctor" \
  -H "X-Doctor-ID: 1" \
  -H "Content-Type: application/json" \
  -d '{"message": "Who is my next patient?", "session_id": "public_test"}'
```

#### Troubleshooting
- 502 Bad Gateway: API server not running or wrong port
- Connection refused: Use `0.0.0.0` as host, not `localhost`
- ngrok URL changes: Free tier gets new URL each time

---

## 🔑 Authentication & Headers
- `X-User-Role`: doctor or assistant (required)
- `X-Doctor-ID`: Doctor identifier (required for doctor role)
- `X-User-ID`: Optional user identifier

---

## 📋 API Endpoints

### 1. POST `/chat` — Main Chat
Processes user queries via the LangGraph agent.
```bash
curl -X POST "https://<your-ngrok-domain>/chat" \
  -H "X-User-Role: doctor" \
  -H "X-Doctor-ID: 1" \
  -H "Content-Type: application/json" \
  -d '{"message": "Who is my next patient?", "session_id": "doctor_1_20250714"}'
```

### 2. GET `/context/{session_id}` — Session Context
Returns conversation context and memory for a session.

### 3. GET `/tools` — Available Tools
Lists all tools available to the current user role.

### 4. GET `/health` — Health Check
Returns API/server status.

### 5. GET `/mcp/summary` — MCP System Summary
Returns Model Context Protocol system info.

### 6. GET `/` — API Info
Root endpoint with API documentation.

### 7. GET `/demo/test` — Demo Endpoint
Sample request/response for testing.

---

## 🧪 Example Queries
- "Who's my next patient?"
- "Who's at 2 PM?"
- "What are my appointments today?"
- "Show me John Smith's history"
- "When am I available tomorrow?"

---

## 🔒 Security Notes
- ngrok URLs are public — do not share real patient data through public tunnels
- Use ngrok authentication for sensitive testing
- Do not commit real static ngrok domains to public repos

---

## 🛠️ Troubleshooting & Monitoring
- API logs: shown in terminal
- ngrok dashboard: http://127.0.0.1:4040
- Common issues: port conflicts, firewall, .env misconfiguration

---

## 📚 Further Info
- For full API details, see the `/docs` endpoint after running the server.
- For advanced deployment (Docker, scripts), see deployment scripts in `code/`.

---

**This guide replaces all previous markdown documentation for this use case.**
