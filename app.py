import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pymongo import MongoClient
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
mongo_uri = os.environ["MONGODB_URI"]

client = MongoClient(mongo_uri)
db = client["StudyBro"]
collection = db["users"]

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    human_message: str

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
    allow_credentials=True
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a Study Assistant that can answer study-related questions named StudyBro."),
    ("placeholder", "{history}"),
    ("user", "{human_message}")
])

llm = ChatGroq(model = "llama-3.3-70b-versatile", api_key= groq_api_key)
chain = prompt_template | llm

user_id = "user123"

def get_history(user_id):
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []

    for chat in chats:
        if chat["role"] == "user":
            history.append(HumanMessage(content=chat["message"]))
        elif chat["role"] == "assistant":
            history.append(AIMessage(content=chat["message"]))

    return history

@app.get("/")
def home():
    return {"message": "Welcome to the StudyBro API"}

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_history(request.user_id)

    response = chain.invoke({"history" : history,"human_message" : request.human_message})

    collection.insert_one({
        "user_id": request.user_id,
        "role": "user",
        "message": request.human_message,
        "timestamp": datetime.utcnow()
    })

    collection.insert_one({
        "user_id": request.user_id,
        "role": "assistant",
        "message": response.content,
        "timestamp": datetime.utcnow()
    })

    return {"response": response.content}


