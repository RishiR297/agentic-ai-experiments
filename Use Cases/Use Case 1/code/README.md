# ✅ Medical Appointment Agent

✅ **Multiple Ways to Run** - Python, NPM, PowerShell options  
✅ **Concurrent Servers** - Both LangGraph and MCP running together  

## 🚀 How to Run (3 Easy Ways)

### 1. NPM (Recommended - Most Professional)
```bash
cd code
npm install  # First time only
npm start    # Runs both servers
```

### 2. Python Script
```bash
cd code
python run_servers.py
```

### 3. Individual Servers
```bash
cd code
npm run start-langgraph  # Port 8001 only
npm run start-mcp        # Port 8002 only
```

## ✅ Verified Working

Both servers are now running and responding:
- **LangGraph Server:** ✅ http://localhost:8001 (healthy)
- **MCP Server:** ✅ http://localhost:8002 (healthy)

## 📁 Final Clean Structure

```
Use Case 1/
├── README.md                # 📖 Complete documentation
├── doctor_appointment_agent_mermaid_diagram.png
└── code/                    # 🎯 All you need
    ├── run_servers.py      # 🐍 Python way
    ├── package.json        # 📦 NPM way  
    ├── requirements.txt    # 🔧 Dependencies
    ├── src/                # 💻 Source code
    │   ├── langgraph_agent/
    │   └── api/
    ├── legacy/             # 🗄️ Old files (archived)
    └── data/               # 💾 Databases
```

## 🎯 Key Benefits

- **No More Confusion:** One README, clear instructions
- **Multiple Options:** Choose Python, NPM, or PowerShell
- **Professional Setup:** NPM scripts for easy management
- **Concurrent Servers:** Both implementations running together
- **Clean Structure:** Organized, GitHub-friendly layout

## 🔧 Available NPM Commands

```bash
npm start              # Both servers (LangGraph + MCP)
npm run start-langgraph # LangGraph only (Port 8001)
npm run start-mcp       # MCP only (Port 8002)
npm run start-legacy    # Legacy Streamlit app
```


The medical appointment agent is now properly organized and easy to run. You can:
1. Use `npm start` for the most professional experience
2. Test both implementations side by side
3. Focus on building features instead of figuring out how to run things

Both servers handle the same medical appointment tasks - choose based on your context preservation needs!
