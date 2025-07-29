import subprocess
#jfnvkflvnklnk
# Start uvicorn (Linux-style path)
subprocess.Popen(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])

# Start MCP wrapper
subprocess.Popen(["python", "mcp_http_wrapper.py"])
