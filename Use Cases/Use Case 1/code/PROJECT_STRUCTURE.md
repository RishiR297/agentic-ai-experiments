# LangGraph Medical Assistant - Project Structure

## 📁 **Clean Codebase Organization**

```
code/
├── 📄 Core Application Files
│   ├── streamlit_app.py              # Main Streamlit frontend interface
│   ├── requirements.txt              # Python dependencies
│   ├── package.json                  # Node.js dependencies (for additional tools)
│   └── .env                         # Environment variables (Azure OpenAI config)
│
|
│
├── 🚀 API & Backend
│   └── src/
│       ├── api/
│       │   └── langgraph_server.py   # FastAPI REST API server
│       ├── langgraph_agent/          # Core LangGraph agent implementation
│       │   ├── core/
│       │   │   ├── config.py         # Agent configuration and system prompts
│       │   │   ├── graph.py          # LangGraph workflow definition
│       │   │   └── state.py          # Agent state management
│       │   ├── nodes/
│       │   │   └── processing.py     # LangGraph processing nodes
│       │   ├── tools/
│       │   │   ├── appointment.py    # Appointment management tools
│       │   │   └── mcp_client.py     # Model Context Protocol client
│       │   └── mcp/
│       │       └── mcp_agent.py      # MCP-enhanced agent
│       └── db/
│           └── database.py           # Database connection and utilities
│
├── 📚 Documentation
│   ├── README.md                     # Project overview and setup guide
│   ├── API_DEPLOYMENT_GUIDE.md       # Complete API documentation
│   └── LangGraph_Medical_Assistant_API.postman_collection.json
│
└── 🛠️ Deployment Scripts
    ├── deploy_api.ps1                # Windows deployment script
    └── deploy_api.sh                 # Linux/Mac deployment script
```

## 🔧 **Core Components**

### **1. Frontend (Streamlit)**
- **File**: `streamlit_app.py`
- **Purpose**: 4-tab diagnostic interface with real-time query processing
- **Features**: Processing Flow, MCP Context, SQL Details, Technical Metadata

### **2. Backend API (FastAPI)**
- **File**: `src/api/langgraph_server.py`
- **Purpose**: RESTful API with comprehensive endpoints
- **Endpoints**: `/chat`, `/health`, `/tools`, `/context/{session_id}`, `/mcp/summary`

### **3. LangGraph Agent**
- **Directory**: `src/langgraph_agent/`
- **Purpose**: Multi-turn conversation processing with context preservation
- **Features**: Intent classification, tool selection, SQL generation, response formatting

### **4. Database Layer**
- **File**: `medical_appointments.db`
- **Schema**: View_Appointments, COR_Doctor tables with medical appointment data
- **Purpose**: Sample data for testing and demonstration

## 🚀 **Quick Start Commands**

### **Start API Server:**
```bash
# Windows
.\deploy_api.ps1

# Linux/Mac
./deploy_api.sh

# Manual
python -m uvicorn src.api.langgraph_server:app --host 0.0.0.0 --port 8502 --reload
```

### **Start Streamlit Interface:**
```bash
streamlit run streamlit_app.py --server.port=8501
```

### **Install Dependencies:**
```bash
pip install -r requirements.txt
```

## 🌐 **Access Points**

- **Streamlit UI**: http://localhost:8501
- **FastAPI Server**: http://localhost:8502
- **API Documentation**: http://localhost:8502/docs
- **API Reference**: http://localhost:8502/redoc

## 📋 **Configuration**

### **Environment Variables (.env):**
```
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
```

## 🧪 **Testing**

- **Postman Collection**: Import `LangGraph_Medical_Assistant_API.postman_collection.json`
- **Sample Queries**: "Who's my next patient?", "Who's at 2 PM?", "What are my appointments today?"
- **Interactive Testing**: Use Swagger UI at `/docs`

## 🔍 **Key Features**

1. **Time-Specific Query Handling**: Proper classification of "Who's at 2 PM?" vs "Who's my next patient?"
2. **LLM SQL Generation**: All queries processed through LLM for transparency
3. **Complete Observability**: 4-tab diagnostics showing processing pipeline
4. **Context Preservation**: Multi-turn conversation memory
5. **Role-Based Access**: Doctor vs Assistant permissions
6. **MCP Integration**: Enhanced context management

## 📊 **Project Status**

✅ **Complete and Functional:**
- LangGraph agent with enhanced time-specific query logic
- FastAPI REST API with all required endpoints
- Streamlit interface with comprehensive diagnostics
- Database integration with sample medical data
- Complete documentation and deployment scripts
- Postman collection for API testing

The codebase is now clean, organized, and ready for production deployment! 🎉
