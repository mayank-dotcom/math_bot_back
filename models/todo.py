from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Message(BaseModel):
    role: str
    content: str

class Todo(BaseModel):
    thread_id: str
    messages: List[Message] 