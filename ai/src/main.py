from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes import chat

app = FastAPI(
    title="Document Management System", 
    description="Intelligent document management system with LLM-powered agents"
)

origins = [
    "http://localhost",
    "http://localhost:3000",  # React frontend
    "http://localhost:8000",  # FastAPI backend
    "http://localhost:8080",  # Alternative frontend
    "http://localhost:3001",  # Alternative frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, tags=["chat"])

@app.get("/")
async def root():
    return {
        "message": "Document Management System API",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
