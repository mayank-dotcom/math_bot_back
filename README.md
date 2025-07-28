# FastAPI Backend Setup

## Environment Variables Setup

### Method 1: Using .env file (Recommended)

1. Create a `.env` file in the backend directory:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=todo_db
COLLECTION_NAME=todo_collection

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

2. Replace the placeholder values with your actual credentials.

### Method 2: Set environment variables directly

#### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
$env:MONGODB_URI="your_mongodb_connection_string"
$env:DATABASE_NAME="todo_db"
$env:COLLECTION_NAME="todo_collection"
```

#### Windows (Command Prompt):
```cmd
set OPENAI_API_KEY=your_openai_api_key_here
set MONGODB_URI=your_mongodb_connection_string
set DATABASE_NAME=todo_db
set COLLECTION_NAME=todo_collection
```

#### Linux/Mac:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export MONGODB_URI="your_mongodb_connection_string"
export DATABASE_NAME="todo_db"
export COLLECTION_NAME="todo_collection"
```

### Method 3: Using a startup script

Create a `start.bat` (Windows) or `start.sh` (Linux/Mac) file:

**start.bat (Windows):**
```batch
@echo off
set OPENAI_API_KEY=your_openai_api_key_here
set MONGODB_URI=your_mongodb_connection_string
set DATABASE_NAME=todo_db
set COLLECTION_NAME=todo_collection
python -m uvicorn main:api --host 0.0.0.0 --port 8000 --reload
```

**start.sh (Linux/Mac):**
```bash
#!/bin/bash
export OPENAI_API_KEY="your_openai_api_key_here"
export MONGODB_URI="your_mongodb_connection_string"
export DATABASE_NAME="todo_db"
export COLLECTION_NAME="todo_collection"
python -m uvicorn main:api --host 0.0.0.0 --port 8000 --reload
```

## Installation and Running

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables using one of the methods above.

3. Run the application:
```bash
python -m uvicorn main:api --host 0.0.0.0 --port 8000 --reload
```

## Security Notes

- Never commit your `.env` file to version control
- Add `.env` to your `.gitignore` file
- Use strong, unique passwords for your database
- Consider using a secrets management service for production deployments 

## Real-time Chat Support

To enable real-time chat updates, install the following dependency:

```bash
pip install 'python-socketio[asyncio]'
``` 