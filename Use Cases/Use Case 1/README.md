# Doctor Appointment Booking Agent

A conversational AI agent for booking doctor appointments using LangGraph, built with a robust appointment booking flow that handles natural language inputs and integrates with a SQLite database.

## Quick Start

**Environment Variables Required**: Set up necessary environment variables for the agent to run.

**Run the Application**:
```bash
cd '.\Use Cases\Use Case 1\code\'
npm run dev
```

## Features

- **Natural Language Processing**: Understands user requests in natural language
- **Doctor Name Normalization**: Handles various formats of doctor names ("Dr.", "doctor", etc.)
- **Intelligent Slot Selection**: Uses LLM to interpret user time preferences
- **Availability Checking**: Validates doctor schedules and existing appointments
- **Robust State Management**: Handles missing information collection
- **Database Integration**: Works with existing hospital appointment systems

## Architecture

```
code/
├── agent/                  # LangGraph agent implementation
│   ├── graph.py           # Agent workflow graph definition
│   ├── nodes.py           # Core agent logic nodes
│   ├── state.py           # Agent state management
│   └── tools/             # Agent-specific tools
│       ├── appointment.py # Appointment booking logic
│       └── mcp_client.py  # MCP client integration
├── tools/                 # Shared utility tools
│   ├── doctor.py          # Doctor lookup and validation
│   └── branch.py          # Branch management utilities
├── utils/                 # Core utilities
│   ├── db.py              # Database connection management
│   ├── llm_extraction.py  # LLM-based field extraction
│   └── time_parser.py     # Time and date parsing utilities
├── main.py                # Main application entry point
├── streamlit_app.py       # Streamlit web interface
└── tool_server.py         # FastAPI tool server (MCP-style)
```

## Database Schema

The system works with the following key tables:

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
