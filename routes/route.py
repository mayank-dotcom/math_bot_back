



from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
from config.database import collection_name
import asyncio
import os
import numpy as np
from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.database import db
import datetime
# Import sio for socketio events
from socketio_instance import sio
import re
from web_search import DuckDuckGoSearchTool
import json

load_dotenv()

router = APIRouter()

# Initialize model (you might want to move this to a separate config file)
model = init_chat_model("gpt-4o-mini", model_provider="openai")
memory = MemorySaver()

# Define structure of each message
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

# Define input body (just messages)
class MessageList(BaseModel):
    messages: List[Message]

# Define query input
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    limit: int = 10

# Define response model
class AIResponse(BaseModel):
    response: str
    session_id: str
    status: str

def get_relevant_syllabus(query, top_k=3):
    """Get relevant syllabus context using embeddings"""
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        query_embedding = np.array(response.data[0].embedding)

        # Fetch all syllabus docs
        syllabus_collection = db["Syllabus"]
        syllabus_docs = list(syllabus_collection.find({}, {"text": 1, "embedding": 1}))

        # Compute cosine similarity
        similarities = []
        for doc in syllabus_docs:
            emb = np.array(doc["embedding"])
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append((sim, doc["text"]))
        similarities.sort(reverse=True)
        return [text for _, text in similarities[:top_k]]
    except Exception as e:
        print(f"Error getting syllabus context: {e}")
        return []

def load_conversation_history(session_id="default", limit=10):
    """Load conversation history for a specific session"""
    try:
        all_messages = []
        # Fetch documents for the specific session
        session_docs = collection_name.find({"session_id": session_id}).sort("timestamp", 1)
        
        for doc in session_docs:
            for msg in doc.get("messages", []):
                all_messages.append(msg)
        
        if not all_messages:
            return []
        
        # Keep only last n messages
        recent = all_messages[-limit:]
        
        # Convert to langchain message objects
        history = []
        for msg in recent:
            if msg['role'] == 'user':
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == 'assistant':
                history.append(AIMessage(content=msg['content']))
        
        return history
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        return []

def insert_conversation(user_query, assistant_response, session_id="default"):
    """Insert conversation into database"""
    try:
        conversation = {
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": assistant_response}
            ],
            "timestamp": datetime.datetime.now()
        }
        collection_name.insert_one(conversation)
        return "conversation inserted successfully"
    except Exception as e:
        return f"error occurred while conversation insertion: {e}"

# Add this configuration near the top of your file
DDG_MCP_SERVER_URL = os.environ.get("DDG_MCP_SERVER_URL", "http://localhost:8080/mcp")

def setup_workflow(syllabus_context):
    """Setup the LangGraph workflow with syllabus context and web search capability"""
    
    # Create the web search tool
    web_search_tool = DuckDuckGoSearchTool(mcp_server_url=DDG_MCP_SERVER_URL)
    
    # Bind tools to the model
    tools = [web_search_tool]
    model_with_tools = model.bind_tools(tools)
    
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
            f"You are a mathematics professor who solves only mathemtaics questions step by step using the following context:\n\n{syllabus_context}\n\n"
     "Always mention the formulas used when solving mathematical problems and don;t answer questions that don't involve maths.\n\n"
     "If a question cannot be answered using the provided syllabus context, you can use the web_search tool to find information online. "
     "Only use web search when absolutely necessary and the question cannot be answered with the syllabus context."
     "ALways refrain from answering questions that are not related to mathematics"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # Define the function that calls the model
    def call_model(state: MessagesState):
        try:
            # Format the prompt with the current messages
            formatted_prompt = prompt_template.invoke({"messages": state["messages"]})
            
            # Call the model with tools
            response = model_with_tools.invoke(formatted_prompt.to_messages())
            
            # Return the response by adding it to messages
            return {"messages": [response]}
        except Exception as e:
            print(f"Error in call_model: {e}")
            # Return an error message if something goes wrong
            error_response = AIMessage(content=f"I apologize, but I encountered an error while processing your request: {str(e)}")
            return {"messages": [error_response]}

    # Create the workflow
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    # Compile the workflow
    app = workflow.compile(checkpointer=memory)
    return app

async def process_ai_query(query: str, session_id: str = "default", limit: int = 10):
    """Process AI query and return response"""
    try:
        # Get relevant syllabus context
        syllabus_contexts = get_relevant_syllabus(query, top_k=3)
        syllabus_context = "\n".join(syllabus_contexts) if syllabus_contexts else "No specific syllabus context found."
        
        print(f"Syllabus context retrieved: {len(syllabus_contexts)} contexts")
        
        # Setup workflow
        app = setup_workflow(syllabus_context)
        
        # Load conversation history
        chat_history = load_conversation_history(session_id=session_id, limit=limit)
        print(f"Chat history loaded: {len(chat_history)} messages")
        
        # Add new user query
        chat_history.append(HumanMessage(content=query))
        
        # Configure thread
        config = {"configurable": {"thread_id": session_id}}
        
        print(f"Invoking LangGraph app with query: {query[:100]}...")
        
        # Call the LangGraph app
        output = app.invoke({"messages": chat_history}, config=config)
        
        # Get the response (last message that's not a tool message)
        response_messages = output["messages"]
        ai_response = None
        
        # Find the last AI message that's not a tool call
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                ai_response = msg
                break
        
        if not ai_response:
            # Fallback: get the last message
            ai_response = response_messages[-1]
        
        response_content = ai_response.content

        # Automatically convert LaTeX delimiters for KaTeX/remark-math
        # Inline math: \( ... \) -> $...$
        response_content = re.sub(r"\\\((.*?)\\\)", r"$\1$", response_content, flags=re.DOTALL)
        # Block math: \[ ... \] -> $$...$$
        response_content = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", response_content, flags=re.DOTALL)

        print(f"AI response generated successfully: {len(response_content)} characters")

        # Insert conversation into database
        insert_result = insert_conversation(query, response_content, session_id=session_id)
        print(f"Database insertion: {insert_result}")
        
        return response_content
        
    except Exception as e:
        print(f"Error in process_ai_query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing AI query: {str(e)}")

@router.get("/")
async def root():
    return {"message": "Hello"}

@router.post("/add-messages")
async def insert_messages(data: MessageList):
    try:
        # Convert to dict format and insert
        doc = {"messages": [msg.dict() for msg in data.messages]}
        result = collection_name.insert_one(doc)
        return {"status": "success", "inserted_id": str(result.inserted_id)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask-ai", response_model=AIResponse)
async def ask_ai(request: QueryRequest):
    """
    Process a query through the AI system and return the response
    """
    try:
        print(f"Received query: {request.query}")
        print(f"Session ID: {request.session_id}")
        
        # Validate that query is not empty
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process the query through AI
        ai_response = await process_ai_query(
            query=request.query,
            session_id=request.session_id,
            limit=request.limit
        )
        
        print(f"AI response ready, emitting to socket...")
        
        # Emit the new assistant message to all connected clients
        await sio.emit('new_message', {
            'role': 'assistant',
            'content': ai_response,
            'session_id': request.session_id
        })
        
        return AIResponse(
            response=ai_response,
            session_id=request.session_id,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in ask_ai: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/conversation-history/{session_id}")
async def get_conversation_history(session_id: str, limit: int = 10):
    """
    Get conversation history for a specific session
    """
    try:
        history = load_conversation_history(session_id=session_id, limit=limit)
        
        # Convert LangChain messages back to dict format
        formatted_history = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                formatted_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_history.append({"role": "assistant", "content": msg.content})
        
        return {
            "session_id": session_id,
            "messages": formatted_history,
            "count": len(formatted_history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")