

# from fastapi import FastAPI
# from socketio_instance import sio
# from routes.route import router
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import os
# import subprocess
# import socketio

# load_dotenv()

# # Create FastAPI app
# fastapi_app = FastAPI()

# fastapi_app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include your API routes with no prefix (we'll mount under /api)
# fastapi_app.include_router(router)

# # Launch mcp_http_wrapper.py on startup
# @fastapi_app.on_event("startup")
# async def run_wrapper():
#     subprocess.Popen(["python", "mcp_http_wrapper.py"])

# # Now mount the FastAPI app under '/api'
# app = socketio.ASGIApp(
#     sio,
#     socketio_path="/socket.io",
#     other_asgi_app=FastAPI()
# )

# # Mount /api onto the main socketio app
# app.other_asgi_app.mount("/api", fastapi_app)

from fastapi import FastAPI
from socketio_instance import sio
from routes.route import router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import subprocess
import socketio

load_dotenv()

# Create the main FastAPI app
fastapi_app = FastAPI()

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or set to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic health check route (optional but useful for Render)
@fastapi_app.get("/")
def health_check():
    return {"status": "ok"}

# Include your API routes under /api
fastapi_app.include_router(router, prefix="/api")

# Launch mcp_http_wrapper.py on startup
@fastapi_app.on_event("startup")
async def run_wrapper():
    subprocess.Popen(["python", "mcp_http_wrapper.py"])

# Mount Socket.IO with FastAPI app
app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=fastapi_app,
    socketio_path="/socket.io"
)
