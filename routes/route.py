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
from langgraph.prebuilt import tools_condition
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config.database import db
import datetime
from socketio_instance import sio
import re
from web_search import DuckDuckGoSearchTool
import json

load_dotenv()

router = APIRouter()
model = init_chat_model("gpt-4o-mini", model_provider="openai")
memory = MemorySaver()

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class MessageList(BaseModel):
    messages: List[Message]

class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    limit: int = 10

class AIResponse(BaseModel):
    response: str
    session_id: str
    status: str

def get_relevant_syllabus(query, top_k=3):
    try:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(input=query, model="text-embedding-3-large")
        query_embedding = np.array(response.data[0].embedding)
        syllabus_docs = list(db["Syllabus"].find({}, {"text": 1, "embedding": 1}))
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
    try:
        all_messages = []
        session_docs = collection_name.find({"session_id": session_id}).sort("timestamp", 1)
        for doc in session_docs:
            all_messages.extend(doc.get("messages", []))
        recent = all_messages[-limit:]
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

def setup_workflow(syllabus_context):
    web_search_tool = DuckDuckGoSearchTool()
    tools = [web_search_tool]
    model_with_tools = model.bind_tools(tools)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
            f"You are a mathematics professor who solves only mathematics questions step by step using the following context:\n\n{syllabus_context}\n\n"
            "Always mention the formulas used when solving mathematical problems.\n\n"
            "If the topic of the maths question is out of the provided syllabus context, you can use the web_search tool to find information online. "
            "Use web_search tool when a maths query cannot be answered with the syllabus context. "
            "Always refrain from answering questions that are not related to mathematics. "
            "IMPORTANT: If you encounter any errors with the web search tool, please include the full error message in your response so we can debug it."
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

    def call_model(state: MessagesState):
        try:
            print(f"call_model called with {len(state['messages'])} messages")
            formatted_prompt = prompt_template.invoke({"messages": state["messages"]})
            print("Calling model with tools...")
            response = model_with_tools.invoke(formatted_prompt.to_messages())
            print(f"Model response: {response}")
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Model wants to use {len(response.tool_calls)} tools")
                for i, tool_call in enumerate(response.tool_calls):
                    print(f"Tool call {i}: {tool_call}")
            
            return {"messages": [response]}
        except Exception as e:
            print(f"Error in call_model: {e}")
            import traceback
            traceback.print_exc()
            error_response = AIMessage(content=f"I apologize, but I encountered an error while processing your request: {str(e)}")
            return {"messages": [error_response]}

    async def async_tool_node(state: MessagesState):
        print("Async tool node called!")
        print(f"Current state has {len(state['messages'])} messages")
        
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print(f"Processing {len(last_message.tool_calls)} tool calls")
            
            tool_messages = []
            for tool_call in last_message.tool_calls:
                print(f"Executing tool call: {tool_call}")
                try:
                    tool_name = tool_call.get("name")
                    tool_id = tool_call.get("id", "unknown_tool_id")
                    args = tool_call.get("args", {})
                    
                    if not isinstance(args, dict) or "query" not in args:
                        raise ValueError(f"Malformed tool args: {args}")
                    
                    if tool_name == "web_search":
                        query = args["query"]
                        print(f"Calling web_search with query: {query}")
                        result = await web_search_tool._arun(query)
                        print(f"Web search result: {result[:200]}...")
                    else:
                        print(f"Unknown tool: {tool_name}")
                        result = f"Unknown tool: {tool_name}"
                    
                    tool_message = ToolMessage(
                        content=result,
                        tool_call_id=tool_id
                    )
                    tool_messages.append(tool_message)
                        
                except Exception as e:
                    print(f"Error executing tool {tool_call.get('name', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    error_message = ToolMessage(
                        content=f"Error executing {tool_call.get('name', 'unknown')}: {str(e)}",
                        tool_call_id=tool_call.get("id", "unknown")
                    )
                    tool_messages.append(error_message)
            
            return {"messages": tool_messages}
        else:
            print("No tool calls found in last message")
            return {"messages": []}

    def tool_node_wrapper(state: MessagesState):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(async_tool_node(state)))
                    return future.result()
            else:
                return loop.run_until_complete(async_tool_node(state))
        except RuntimeError:
            return asyncio.run(async_tool_node(state))

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node_wrapper)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=memory)

async def process_ai_query(query: str, session_id: str = "default", limit: int = 10):
    try:
        print(f"Processing AI query: {query[:100]}...")
        
        syllabus_contexts = get_relevant_syllabus(query, top_k=3)
        syllabus_context = "\n".join(syllabus_contexts) if syllabus_contexts else "No specific syllabus context found."
        
        print(f"Syllabus context retrieved: {len(syllabus_contexts)} contexts")
        
        app = setup_workflow(syllabus_context)
        chat_history = load_conversation_history(session_id=session_id, limit=limit)
        
        print(f"Chat history loaded: {len(chat_history)} messages")
        
        chat_history.append(HumanMessage(content=query))
        config = {"configurable": {"thread_id": session_id}}
        
        print(f"Invoking LangGraph app with query: {query[:100]}...")
        
        output = app.invoke({"messages": chat_history}, config=config)
        response_messages = output["messages"]
        
        # Find the last AI message that's not a tool call
        ai_response = None
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                ai_response = msg
                break
        
        if not ai_response:
            ai_response = response_messages[-1]
        
        response_content = ai_response.content

        # Convert LaTeX delimiters for KaTeX/remark-math
        response_content = re.sub(r"\\\((.*?)\\\)", r"$\1$", response_content, flags=re.DOTALL)
        response_content = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", response_content, flags=re.DOTALL)

        print(f"AI response generated successfully: {len(response_content)} characters")

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
        doc = {"messages": [msg.dict() for msg in data.messages]}
        result = collection_name.insert_one(doc)
        return {"status": "success", "inserted_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask-ai", response_model=AIResponse)
async def ask_ai(request: QueryRequest):
    try:
        print(f"Received query: {request.query}")
        print(f"Session ID: {request.session_id}")
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        ai_response = await process_ai_query(
            query=request.query,
            session_id=request.session_id,
            limit=request.limit
        )
        
        print(f"AI response ready, emitting to socket...")
        
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
    try:
        history = load_conversation_history(session_id=session_id, limit=limit)
        formatted_history = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                formatted_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_history.append({"role": "assistant", "content": msg.content})
        return {"session_id": session_id, "messages": formatted_history, "count": len(formatted_history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")