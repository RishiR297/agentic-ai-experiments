# 🏥 Medical Assistant Agent - Hosting & API Guide

This guide explains how to host the Medical Assistant Agent and interact with its API endpoints.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js and npm
- ngrok account (for public hosting)

### 1. Start the Agent
```bash
cd "Use Case 1/code"
npm run start-full
```

This starts all services:
- **LangGraph API**: `http://localhost:8001`
- **MCP Server**: `http://localhost:8002` 
- **Streamlit UI**: `http://localhost:8504`

### 2. View API Documentation
Once running, access the interactive API docs:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## 🌐 Public Hosting with ngrok

### Setup ngrok
1. Install ngrok: https://ngrok.com/download
2. Authenticate: `ngrok authtoken YOUR_TOKEN`
3. Start tunnel: `ngrok http 8001`

### Example ngrok Output
```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:8001
```

Your public API is now available at: `https://abc123.ngrok-free.app`

## 📚 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Main conversation endpoint |
| `GET` | `/health` | Health check |
| `GET` | `/tools` | Available tools list |
| `GET` | `/context/{session_id}` | Session context |
| `GET` | `/docs` | Swagger UI documentation |
| `GET` | `/redoc` | ReDoc documentation |

## 💬 Chat API Usage

### Request Format
```json
{
  "message": "Your message here",
  "session_id": "unique_session_identifier", 
  "user_role": "doctor" | "assistant",
  "doctor_id": "14" (for doctor role)
}
```

### Response Format
```json
{
  "success": true,
  "result": "Agent response text",
  "metadata": {
    "intent": "detected_intent",
    "tool_used": "tool_name",
    "has_errors": false
  },
  "session_id": "session_identifier",
  "sql_metadata": {
    "query_type": "booking|query",
    "parameters": [...],
    "execution": "status"
  }
}
```

## 🧪 Example API Calls

### 1. Health Check
```bash
# Local
curl http://localhost:8001/health

# Public (replace with your ngrok URL)
curl https://abc123.ngrok-free.app/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-27T20:20:20.198236",
  "agent_status": "operational", 
  "active_sessions": 0
}
```

### 2. Check Schedule
```bash
# Local
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me my schedule for tomorrow",
    "session_id": "demo_session",
    "user_role": "doctor",
    "doctor_id": "14"
  }'

# Public
curl -X POST https://abc123.ngrok-free.app/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me my schedule for tomorrow", 
    "session_id": "demo_session",
    "user_role": "doctor",
    "doctor_id": "14"
  }'
```

### 3. Book Appointment
```bash
curl -X POST https://abc123.ngrok-free.app/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Book an appointment tomorrow for John Doe at the earliest slot for a consultation",
    "session_id": "booking_session",
    "user_role": "doctor", 
    "doctor_id": "14"
  }'
```

### 4. Find Available Slots
```bash
curl -X POST https://abc123.ngrok-free.app/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What appointment slots are available tomorrow?",
    "session_id": "schedule_session",
    "user_role": "doctor",
    "doctor_id": "14"
  }'
```

### 5. Reschedule Appointment
```bash
curl -X POST https://abc123.ngrok-free.app/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Reschedule John Doe appointment to 2 PM tomorrow",
    "session_id": "reschedule_session", 
    "user_role": "doctor",
    "doctor_id": "14"
  }'
```

### 6. Assistant Role Query
```bash
curl -X POST https://abc123.ngrok-free.app/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me Dr. Smith schedule for today", 
    "session_id": "assistant_session",
    "user_role": "assistant"
  }'
```

## 🔧 Advanced Usage

### Session Management
- Use consistent `session_id` to maintain conversation context
- Each session preserves conversation history and references
- Session context available at: `GET /context/{session_id}`

### Role-Based Access
- **Doctor Role**: Can manage own appointments (requires `doctor_id`)
- **Assistant Role**: Can view/manage appointments across doctors

### Natural Language Processing
The agent understands natural language including:
- **Dates**: "tomorrow", "next Monday", "August 28th"
- **Times**: "earliest slot", "2 PM", "morning appointment" 
- **Services**: "consultation", "follow up", "botox", etc.
- **Actions**: "book", "reschedule", "cancel", "show schedule"

## 🛠 Development & Testing

### Local Development
```bash
# Start individual services
python api/langgraph_server.py          # Port 8001
python api/mcp_server.py               # Port 8002  
streamlit run apps/streamlit_app.py    # Port 8504

# Or start all together
npm run start-full
```

### Testing Tools
```bash
# Test health
curl http://localhost:8001/health

# View available tools
curl http://localhost:8001/tools

# Interactive docs
open http://localhost:8001/docs
```

## 🔍 Troubleshooting

### Common Issues

1. **Port conflicts**: Kill existing processes with `taskkill /IM python.exe /F`
2. **ngrok tunnel issues**: Restart ngrok tunnel
3. **API not responding**: Check if all services are running
4. **Session errors**: Use fresh session IDs

### Logs & Debugging
- Server logs show detailed request/response information
- Check `sql_metadata` in responses for database query details
- Use `/health` endpoint to verify system status

## 📊 Service Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  LangGraph API   │    │   MCP Server    │
│   Port 8504     │◄──►│   Port 8001      │◄──►│   Port 8002     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  SQLite Database │
                       │  Medical Records │
                       └──────────────────┘
```

## 🚀 Production Deployment

For production deployment:
1. Use a proper reverse proxy (nginx)
2. Set up SSL certificates
3. Configure environment variables
4. Use production WSGI server (gunicorn)
5. Set up monitoring and logging

---

## 📞 Support

For issues or questions:
- Check the `/docs` endpoint for interactive API documentation
- Review server logs for detailed error information
- Test with simple health check first: `curl YOUR_URL/health`
