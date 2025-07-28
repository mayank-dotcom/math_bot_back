from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get database configuration from environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Create MongoDB client
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]
collection_name = db[COLLECTION_NAME]
