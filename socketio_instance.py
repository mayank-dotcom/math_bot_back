# from socketio import AsyncServer

# sio = AsyncServer(async_mode='asgi', cors_allowed_origins='*') 

import socketio

# Create Socket.IO server with proper configuration
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    logger=True,  # Enable logging for debugging
    engineio_logger=True  # Enable engine.io logging
)

# Optional: Add connection event handlers for debugging
@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")

@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")