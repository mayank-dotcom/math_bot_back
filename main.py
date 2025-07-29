

# from fastapi import FastAPI
# from socketio_instance import sio
# import socketio
# from routes.route import router
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Create FastAPI app
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Include your API routes
# app.include_router(router, prefix="/api")

# # Mount Socket.IO app
# socket_app = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="/socket.io")

# # Export the socket_app as the main app
# app = socket_app


from fastapi import FastAPI
from socketio_instance import sio
import socketio
from routes.route import router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import subprocess

load_dotenv()

# Create FastAPI app
base_app = FastAPI()

base_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your API routes
base_app.include_router(router, prefix="/api")

# Launch mcp_http_wrapper.py on startup
@base_app.on_event("startup")
async def run_wrapper():
    subprocess.Popen(["python", "mcp_http_wrapper.py"])

# Mount Socket.IO app
app = socketio.ASGIApp(sio, other_asgi_app=base_app, socketio_path="/socket.io")
