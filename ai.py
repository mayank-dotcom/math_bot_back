import getpass
import os
import numpy as np
from openai import OpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from config.database import collection_name, db
import datetime
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

model = init_chat_model("gpt-4o-mini", model_provider="openai")

def get_relevant_syllabus(query, top_k=3):
    # Generate embedding for the query using OpenAI API
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

def load_last_n_messages(limit=10):
   all_messages = []
   #fetch all documents(chats)
   all_docs = collection_name.find()
   for doc in all_docs:
      for msg in doc.get("messages",[]):
         all_messages.append(msg)
          # ❗️If no messages exist, return an empty list
   if not all_messages:
        return []
         #keep only last n messages
   recent  = all_messages[-limit:]
         #convert to langchain message objects
   history = []
   for msg in recent:
            if msg['role'] == 'user':
               history.append(HumanMessage(content = msg["content"]))
            elif msg["role"] == 'assistant':
               history.append(AIMessage(content=msg['content']))
   return history

def load_conversation_history(session_id="default", limit=10):
    """Load conversation history for a specific session"""
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

# query = "Evaluate the integral:∫ (3x² + 2x + 1) dx"
query = "Prove: sin(theta) / cos(theta) = tan(theta)"
# query = "hey how are you ?"
# Get relevant syllabus context
syllabus_contexts = get_relevant_syllabus(query, top_k=3)
syllabus_context = "\n".join(syllabus_contexts)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are a mathematics professor who solves questions step by step using the following context strictly:\n{syllabus_context}\n.Always mention the formulas used.If any question is not solvable by using the given forumulas and require some extra formulas, simply tell I can't solve the question.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc345"}}

# Load last 10 messages from DB
chat_history = load_conversation_history(session_id="default", limit=10)

# Add new user query
chat_history.append(HumanMessage(content=query))

# ✅ FIX: Add thread_id config
config = {"configurable": {"thread_id": "default"}}

# Call the LangGraph app
output = app.invoke({"messages": chat_history}, config=config)

# Print the reply
response = output["messages"][-1]
output["messages"][-1].pretty_print()

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

# Insert the complete conversation
message = insert_conversation(query, response.content, session_id="default")
print(message)
