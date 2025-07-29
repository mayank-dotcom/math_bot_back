import subprocess

# Start uvicorn server
subprocess.Popen(["cmd", "/k", "cd /d D:\\fastapi\\backend && uvicorn main:app --port 8000"])

# Start the mcp wrapper
subprocess.Popen(["cmd", "/k", "cd /d D:\\fastapi && python mcp_http_wrapper.py"])
