from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.services.rag_service import get_rag_service, RAGService, update_repository
from app.core.config import settings
import asyncio
from pathlib import Path
import time
import subprocess
from typing import AsyncGenerator, Union, List, Literal, Optional, Dict
import json
from pydantic import BaseModel, Field, ValidationError

# --- MOVED Pydantic Model Definitions HERE ---
# Model for complex content parts (like text)
class ContentPart(BaseModel):
    type: Literal['text', 'image_url']
    text: Optional[str] = None

class VercelChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'function']
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None

class VercelChatRequest(BaseModel):
    messages: List[VercelChatMessage]
# --- End Moved Definitions ---

print(f"Backend starting with settings:")
print(f"- Litepaper Source: {settings.litepaper_src_dir}")
print(f"- Vector DB: {settings.vector_db_dir}")
print(f"- Embedding Model: {settings.embedding_model_name}")
print(f"- LLM: {settings.openai_chat_model}")

app = FastAPI(
    title="ChaosChain Litepaper RAG API",
    description="API for querying the ChaosChain Litepaper using RAG.",
    version="0.1.0"
)

# CORS Configuration
print(f"--- Configuring CORS --- Allowed Origins Raw: {settings.cors_origins}")
allowed_origins_list = settings.cors_origins.split(",")
print(f"--- Configuring CORS --- Allowed Origins List: {allowed_origins_list}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list, # Use the pre-split list
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)

async def update_repository_task():
    """Background task to update the repository periodically."""
    while True:
        try:
            if settings.is_github_repo:
                repo_name = settings.repo_name
                if repo_name:
                    repo_dir = Path("data/repos") / repo_name
                    if repo_dir.exists():
                        print("Running periodic repository update...")
                        update_repository(repo_dir)
                        print("Repository update completed.")
                else:
                    print("Warning: Could not determine repository name for periodic update.")
        except Exception as e:
            print(f"Error in repository update task: {e}")
        
        # Wait for 1 hour before next update
        await asyncio.sleep(3600)

@app.on_event("startup")
async def startup_event():
    # Start the background task
    asyncio.create_task(update_repository_task())

# Async generator for streaming the response
async def stream_answer(query: str, rag_service: RAGService) -> AsyncGenerator[str, None]:
    print("--- ENTERED stream_answer --- ")
    print(f"[Stream] Generating answer for query: {query}")
    try:
        result = await rag_service.get_answer(query)
        answer = result.get("answer", "Sorry, an error occurred while generating the answer.")
        sources = result.get("sources", [])
        print(f"[Stream] RAG Service returned: Answer=\"{answer}\", Sources Count={len(sources)}")

        # --- Stream Text Chunks --- 
        # Simple word-by-word streaming for demonstration
        words = answer.split(' ')
        for i, word in enumerate(words):
            chunk = word + (' ' if i < len(words) - 1 else '') # Add space back
            yield f"0:{json.dumps(chunk)}\n" # Prefix with 0: and JSON encode
            print(f"[Stream] Yielded TEXT chunk: 0:{json.dumps(chunk)}")
            await asyncio.sleep(0.01) # Small delay to simulate streaming
        # --------------------------

        # --- Stream Sources Annotation --- 
        if sources:
            # Ensure sources are JSON serializable (convert Path objects if needed)
            serializable_sources = [
                {
                    "name": src.get("name", "Unknown Source"), 
                    "url": src.get("url", "#"), 
                    "page_content": src.get("page_content", "") 
                } 
                for src in sources
            ]
            sources_json_str = json.dumps(serializable_sources)
            yield f"2:{sources_json_str}\n" # Prefix with 2:
            print(f"[Stream] Yielded SOURCES annotation: 2:{sources_json_str}")
         # --------------------------

    except Exception as e:
        print(f"[Stream] Error during answer generation: {e}")
        error_message = f"Sorry, an internal error occurred: {e}"
        yield f"0:{json.dumps(error_message)}\n"
        print(f"[Stream] Yielded ERROR chunk: 0:{json.dumps(error_message)}")
    finally:
        print("--- EXITING stream_answer --- ")
        print("[Stream] Finished streaming.")

@app.post("/api/chat")
async def handle_chat_request(
    raw_request: Request,
    rag_service_instance: RAGService = Depends(get_rag_service)
):
    print("--- ENTERED handle_chat_request --- ")
    # Manually parse and validate the request body
    try:
        body = await raw_request.json()
        print(f"[API] Received raw request body: {body}")
        request = VercelChatRequest.model_validate(body)
        print(f"[API] Successfully validated request payload: {request.dict()}")
    except ValidationError as e:
        print(f"[API] Pydantic Validation Error: {e.errors()}")
        raise HTTPException(status_code=422, detail=e.errors())
    except json.JSONDecodeError:
        print("[API] Error: Invalid JSON received")
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    if not request.messages or request.messages[-1].role != 'user':
        print("[API] Validation Error: Last message not from user")
        raise HTTPException(status_code=400, detail="Invalid request: Last message must be from user.")
    
    last_message = request.messages[-1]
    query = ""
    # Extract query based on content type
    if isinstance(last_message.content, str):
        query = last_message.content
    elif isinstance(last_message.content, list):
        text_part = next((part for part in last_message.content if part.type == 'text' and part.text is not None), None)
        if text_part:
            query = text_part.text
    
    print(f"[API] Extracted query: {query}")
    if not query:
        print("[API] Validation Error: Query is empty or invalid content structure")
        raise HTTPException(status_code=400, detail="Query cannot be empty or content structure is invalid")

    print("--- BEFORE StreamingResponse --- ")
    print("[API] Returning StreamingResponse formatted for Vercel AI SDK")
    return StreamingResponse(
        stream_answer(query, rag_service_instance),
        media_type="text/plain"
    )

@app.get("/api/info")
async def get_info():
    """Endpoint to get repository information, like last update time."""
    last_update_time = "N/A"
    try:
        if settings.is_github_repo and settings.repo_name:
            repo_dir = Path("data/repos") / settings.repo_name
            if repo_dir.exists():
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%cd", "--date=iso"],
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                last_update_time = result.stdout.strip()
    except Exception as e:
        print(f"Error getting last commit date: {e}")
        # Keep last_update_time as "N/A" or handle error differently
        pass  # Or raise HTTPException if this is critical

    return {"last_update_time": last_update_time}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 