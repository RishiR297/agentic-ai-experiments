#!/usr/bin/env python3
"""
Simple script to run both LangGraph and MCP servers concurrently
"""
import subprocess
import sys
import os
import time
from pathlib import Path

def main():
    print("🏥 Medical Appointment Agent - Starting Both Servers")
    print("=" * 55)
    
    # Add src to Python path
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    sys.path.insert(0, str(src_dir))
    
    # Change to the src directory for imports to work
    os.chdir(src_dir)
    
    print("📁 Working from:", os.getcwd())
    print("🚀 Starting both servers concurrently...")
    print()
    
    try:
        # Start LangGraph server (Port 8001)
        print("🟢 Starting LangGraph Server on Port 8001...")
        langgraph_process = subprocess.Popen([
            sys.executable, "api/langgraph_server.py"
        ], cwd=src_dir)
        
        # Give it a moment to start
        time.sleep(2)
        
        # Start MCP server (Port 8002)
        print("🟡 Starting MCP Server on Port 8002...")
        mcp_process = subprocess.Popen([
            sys.executable, "api/mcp_server.py"
        ], cwd=src_dir)
        
        print()
        print("✅ Both servers are starting up!")
        print("📊 Server URLs:")
        print("   🔗 LangGraph API: http://localhost:8001")
        print("   🔗 MCP API: http://localhost:8002")
        print()
        print("Press Ctrl+C to stop both servers")
        print("-" * 50)
        
        # Wait for both processes
        langgraph_process.wait()
        mcp_process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        
        # Terminate both processes
        if 'langgraph_process' in locals():
            langgraph_process.terminate()
        if 'mcp_process' in locals():
            mcp_process.terminate()
            
        print("✅ Servers stopped successfully")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
