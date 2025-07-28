

from fastapi import FastAPI
from socketio_instance import sio
import socketio
from routes.route import router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your API routes
app.include_router(router, prefix="/api")

# Mount Socket.IO app
socket_app = socketio.ASGIApp(sio, other_asgi_app=app, socketio_path="/socket.io")

# Export the socket_app as the main app
app = socket_app