#!/usr/bin/env python3
"""
Main Entry Point for Medical Assistant System

This script provides multiple ways to run the medical assistant:
1. LangGraph multi-turn agent (recommended)
2. Legacy single-turn system
3. Individual components
"""

import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def run_langgraph_system():
    """Run the full LangGraph multi-turn system."""
    print("🚀 Starting LangGraph Multi-Turn Medical Assistant...")
    print("📡 Backend: http://127.0.0.1:8001")
    print("🌐 Frontend: http://localhost:8502")
    print("=" * 50)
    
    try:
        # Use npm start to run both components with concurrently
        subprocess.run(["npm", "start"], check=True)
    except subprocess.CalledProcessError:
        print("❌ npm start failed. Trying direct Python execution...")
        run_langgraph_manual()
    except FileNotFoundError:
        print("❌ npm not found. Running with Python directly...")
        run_langgraph_manual()

def run_langgraph_manual():
    """Run LangGraph system manually with Python."""
    import threading
    import os
    
    def start_api():
        os.system("python api/langgraph_server.py")
    
    def start_ui():
        time.sleep(3)  # Wait for API to start
        os.system("streamlit run ui/streamlit_app.py --server.port 8502")
    
    print("🔧 Starting components manually...")
    
    # Start API server in background thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    # Start UI (this will block)
    start_ui()

def run_legacy_system():
    """Run the legacy single-turn system."""
    print("🔄 Starting Legacy Medical Assistant...")
    print("📡 Backend: http://127.0.0.1:8001")
    print("🌐 Frontend: http://localhost:8502")
    print("=" * 50)
    
    try:
        subprocess.run(["npm", "run", "start-legacy"], check=True)
    except subprocess.CalledProcessError:
        print("❌ npm failed. Running legacy system manually...")
        run_legacy_manual()

def run_legacy_manual():
    """Run legacy system manually."""
    import threading
    import os
    
    def start_legacy_api():
        os.system("python llm_tool_server.py")
    
    def start_legacy_ui():
        time.sleep(3)
        os.system("streamlit run streamlit_app.py --server.port 8502")
    
    # Start legacy API server
    api_thread = threading.Thread(target=start_legacy_api, daemon=True)
    api_thread.start()
    
    # Start legacy UI
    start_legacy_ui()

def run_api_only():
    """Run only the LangGraph API server."""
    print("🔧 Starting LangGraph API Server only...")
    print("📡 API Server: http://127.0.0.1:8001")
    subprocess.run(["python", "api/langgraph_server.py"])

def run_ui_only():
    """Run only the Streamlit UI."""
    print("🌐 Starting Streamlit UI only...")
    print("🌐 UI Server: http://localhost:8502")
    subprocess.run(["streamlit", "run", "ui/streamlit_app.py", "--server.port", "8502"])

def show_help():
    """Show usage information."""
    print("""
🏥 Medical Assistant System - Usage Guide

Available Commands:
  python main.py                    - Run LangGraph multi-turn system (default)
  python main.py langgraph         - Run LangGraph multi-turn system
  python main.py legacy            - Run legacy single-turn system
  python main.py api               - Run only the LangGraph API server
  python main.py ui                - Run only the Streamlit UI
  python main.py help              - Show this help message

🎯 Recommended: Use 'python main.py' for the full multi-turn experience

🔧 Alternative npm commands:
  npm start                        - Full LangGraph system
  npm run start-legacy            - Legacy system
  npm run start-langgraph         - LangGraph API only
  npm run start-ui                - UI only

📖 Features:
  ✅ Multi-turn conversations with context
  ✅ Implicit reference resolution ("her", "next patient")
  ✅ Session management and memory
  ✅ Role-based access (doctor/assistant)
  ✅ Natural follow-up questions

🌐 Access Points:
  - Web UI: http://localhost:8502
  - API: http://127.0.0.1:8001
  - Health Check: http://127.0.0.1:8001/health
""")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        command = "langgraph"  # Default to LangGraph system
    else:
        command = sys.argv[1].lower()
    
    if command in ["help", "-h", "--help"]:
        show_help()
    elif command in ["langgraph", "lg", "multi", "new"]:
        run_langgraph_system()
    elif command in ["legacy", "old", "single"]:
        run_legacy_system()
    elif command in ["api", "server", "backend"]:
        run_api_only()
    elif command in ["ui", "frontend", "streamlit"]:
        run_ui_only()
    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Use 'python main.py help' for usage information")
        sys.exit(1)

if __name__ == "__main__":
    main()
