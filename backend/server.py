from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import SecondBrainAgent
from app.memory import LocalMemoryStore
import uvicorn
import os

app = FastAPI(title="Second Brain Agent")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent and Memory
# In a real app, you might want this to persist to disk
memory_store = LocalMemoryStore()
agent = SecondBrainAgent(memory_store=memory_store)

class ChatRequest(BaseModel):
    message: str

# Simple in-memory history for the session
chat_history = [] 

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Pass history to agent
        response = agent.process_input(request.message, chat_history)
        
        # Update history
        chat_history.append(f"User: {request.message}")
        chat_history.append(f"AI: {response}")
        
        # Keep last 10 turns
        if len(chat_history) > 20:
             chat_history.pop(0)
             chat_history.pop(0)
             
        return {"response": response}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
async def get_graph():
    """Visualize the current knowledge graph"""
    return memory_store.get_graph_data()

@app.get("/summary/{entity_id}")
async def get_entity_summary(entity_id: str):
    try:
        data = agent.summarize_entity(entity_id)
        return {"entity": entity_id, **data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
