# LangGraph Medical Assistant

A sophisticated multi-turn conversational agent for medical appointment management, built with LangGraph and FastAPI.

## 🏗️ Architecture Overview

This application has been reorganized from a simple LLM-powered tool into a comprehensive LangGraph multi-turn agent with the following architecture:

```
code/
├── agent/                    # LangGraph Agent Implementation
│   ├── state.py             # Agent state management & conversation history
│   ├── nodes.py             # Processing nodes (intent, query, response)
│   ├── graph.py             # LangGraph workflow definition
│   └── tools/               # Agent tools
│       ├── appointment_tools.py  # Appointment management
│       └── query_tools.py        # SQL query generation
├── core/                    # Core Services
│   ├── config.py           # Configuration management
│   ├── database.py         # Database operations
│   └── llm_service.py      # LLM interactions
├── api/                     # REST API Layer
│   ├── app.py              # FastAPI application
│   └── routes.py           # API endpoints
├── ui/                      # User Interface (Future)
├── main.py                 # Application entry point
├── streamlit_app.py        # Streamlit web interface
└── requirements.txt        # Dependencies
```

## 🤖 Agent Capabilities

### Multi-Turn Conversations
- **State Management**: Maintains conversation history and context across turns
- **Intent Recognition**: Understands user intentions using LLM-based analysis
- **Entity Extraction**: Extracts relevant information from user queries
- **Clarification Handling**: Asks for clarification when intent is unclear

### Appointment Management
- **Natural Language Queries**: "Show me upcoming appointments", "Find patient Smith"
- **Date/Time Filtering**: "appointments today", "this week", "tomorrow"
- **Patient Search**: Search by patient ID or name
- **Status Filtering**: Filter by appointment status (scheduled, cancelled, etc.)
- **Doctor-Specific Views**: Filter appointments by doctor UUID

### Query Processing
- **LLM-Generated SQL**: Dynamic SQL query generation from natural language
- **Fallback Mechanisms**: Hardcoded query patterns when LLM fails
- **Query Validation**: Safety checks to prevent dangerous SQL operations
- **Multiple Strategies**: Intent-based routing to appropriate query methods

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- SQLite database with medical appointment data
- Azure OpenAI API access

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   Create a `.env` file with:
   ```env
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
   AZURE_OPENAI_API_VERSION=2024-02-15-preview
   
   # Optional configurations
   API_HOST=127.0.0.1
   API_PORT=8001
   DEBUG=true
   USE_LLM_QUERY_GENERATION=true
   FALLBACK_TO_HARDCODED_ON_LLM_FAILURE=true
   ENABLE_DEBUG_LOGGING=true
   ```

3. **Database Setup**
   Ensure your SQLite database is available at `./db/output.db` with:
   - `View_Appointments` view/table
   - `COR_Doctor` table
   - Proper schema as expected by the application

### Running the Application

#### Option 1: Run Both Servers Concurrently (Recommended)

**Using Python:**
```bash
cd code
python run_servers.py
```

**Using NPM:**
```bash
cd code
npm install  # First time only
npm start    # Runs both LangGraph (8001) and MCP (8002) servers
```

**Using PowerShell (Windows):**
```powershell
cd code
.\start_servers.ps1
```

#### Option 2: Individual Servers

**LangGraph Server Only (Port 8001):**
```bash
cd code
npm run start-langgraph
# OR
cd code/src && python api/langgraph_server.py
```

**MCP Server Only (Port 8002):**
```bash
cd code
npm run start-mcp
# OR
cd code/src && python api/mcp_server.py
```

#### Option 3: Legacy Interface (Archived)
```bash
cd code
npm run start-legacy
# OR
cd code/legacy && python streamlit_app.py
```

### Server URLs
- **LangGraph API:** `http://localhost:8001` (Main medical agent)
- **MCP API:** `http://localhost:8002` (Enhanced context version)
- **API Documentation:** `http://localhost:8001/docs` or `http://localhost:8002/docs`

### NPM Commands Reference

```bash
npm start        # Run both servers concurrently (LangGraph + MCP)
npm run dev      # Same as npm start
npm run start-both        # Alternative: use Python script
npm run start-langgraph   # Run only LangGraph server (Port 8001)
npm run start-mcp         # Run only MCP server (Port 8002)
npm run start-legacy      # Run legacy Streamlit app
npm test         # Run tests (placeholder)
```

### Testing the APIs

**Test LangGraph Server:**
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need to book an appointment with a cardiologist"}'
```

**Test MCP Server:**
```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I need to book an appointment with a cardiologist"}'
```

### Environment Check
```bash
python main.py --check-env
```
Validates configuration and component health.

- **View_Appointments**: Main appointments table with all booking data
- **View_Appointments_Setup**: Reference table for doctor-service combinations
- **COR_Doctor**: Doctor information (names, IDs, specialties)
- **COR_DoctorSchedule**: Doctor working hours and availability
- **COR_DoctorOffSchedule**: Doctor off days and exceptions

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js (for running Streamlit with npm)
- SQLite database with appointment system data

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment
```

### Running the Application

1. Start the tool server:
```bash
python tool_server.py
```

2. Run the Streamlit app:
```bash
npm run dev
```

## Key Components

### Agent Workflow (`agent/nodes.py`)
- **planner_node**: Main planning logic with field extraction
- **ask_for_missing_fields_node**: Collects missing information
- **tool_node**: Executes appointment booking tools
- **answer_node**: Provides responses to users

### Appointment System (`agent/tools/appointment.py`)
- **book_appointment_tool**: Complete booking workflow
- **suggest_appointment_slots**: Find available time slots
- **is_doctor_available**: Availability validation
- **create_appointment**: Database insertion logic

### Doctor Management (`tools/doctor.py`)
- **process_doctor_name**: Normalize doctor names for lookup
- **get_services_for_doctor**: Available services validation
- **suggest_doctor_for_service**: Find doctors by service

## Usage Examples

The agent can handle natural language requests like:

- "I want to book an appointment with Dr. Antonella"
- "Is Dr. Smith available today?"
- "Book me for laser treatment on July 10th"
- "What services does Dr. Jones offer?"

## API Endpoints

The tool server exposes the following endpoints:

- `POST /tools/book_appointment_tool` - Book a new appointment
- `POST /tools/get_appointments` - Retrieve appointments
- `POST /tools/suggest_appointment_slots` - Get available slots
- `POST /tools/get_earliest_available_slot` - Find earliest slot
- `POST /tools/get_next_client_info` - Next patient info
- `POST /tools/summarize_calendar_today` - Today's schedule

## Error Handling

The system includes comprehensive error handling for:
- Invalid doctor names
- Scheduling conflicts
- Missing required information
- Database connection issues
- Malformed time inputs

## Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings to all functions
3. Include error handling for edge cases
4. Test booking flows end-to-end before committing

## License

This project is part of an internship program for agentic AI experiments.
