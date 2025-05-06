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
# --- LLM Preprocessing Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
# --- End LLM Imports ---
import uuid
import traceback
# --- Add context manager for lifespan ---
from contextlib import asynccontextmanager

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

# --- Lifespan Management for Startup Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code here runs on startup
    print("--- Application Startup ---")
    print("Initializing RAG Service...")
    start_time = time.time()
    get_rag_service() # Initialize the singleton instance
    end_time = time.time()
    print(f"RAG Service initialized in {end_time - start_time:.2f} seconds.")
    print("--- Application Ready ---")
    yield
    # Code here runs on shutdown (if needed)
    print("--- Application Shutdown ---")

# Initialize FastAPI app with lifespan
app = FastAPI(title="ChaosChain Litepaper RAG API", lifespan=lifespan)

# Configure CORS
print(f"--- Configuring CORS --- Allowed Origins Raw: {settings.cors_origins}")
allowed_origins_list = settings.cors_origins.split(",")
print(f"--- Configuring CORS --- Allowed Origins List: {allowed_origins_list}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list, # Use the pre-split list
    allow_credentials=True,
    allow_methods=["*", "OPTIONS"] , # Explicitly added OPTIONS
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
async def stream_answer(query: str, rag_service: RAGService, request_id: str) -> AsyncGenerator[str, None]:
    print(f"[{request_id}] --- ENTERED stream_answer generator --- ")
    print(f"[{request_id}] [Stream] Generating answer for query: '{query}'")
    response_generated = False
    try:
        print(f"[{request_id}] [Stream] Calling rag_service.get_answer...")
        result = await rag_service.get_answer(query)
        response_generated = True
        print(f"[{request_id}] [Stream] rag_service.get_answer returned: {result}") # Log the whole result
        
        answer = result.get("answer", "Sorry, an error occurred while generating the answer.")
        sources = result.get("sources", [])
        print(f"[{request_id}] [Stream] Extracted: Answer='{answer}', Sources Count={len(sources)}")

        # --- Check for "I don't know" and replace --- 
        if answer.strip().lower() == "i don't know.":
            answer = ("I couldn't find specific information about that in the Chaos Chain litepaper. "
                      "Could you try rephrasing or asking about a different topic covered in the document?")
            print(f"[{request_id}] [Stream] Replaced 'I don\'t know.' answer with custom message.")
        # --------------------------------------------

        # --- Stream Text Chunks --- 
        print(f"[{request_id}] [Stream] Starting to stream TEXT chunks...")
        words = answer.split(' ')
        for i, word in enumerate(words):
            chunk = word + (' ' if i < len(words) - 1 else '')
            yield f"0:{json.dumps(chunk)}\n"
            # print(f"[{request_id}] [Stream] Yielded TEXT chunk: 0:{json.dumps(chunk)}") # Reduce noise maybe
            await asyncio.sleep(0.01)
        print(f"[{request_id}] [Stream] Finished streaming TEXT chunks.")
        # --------------------------

        # --- Stream Sources Annotation --- 
        if sources:
            print(f"[{request_id}] [Stream] Starting to stream SOURCES annotation...")
            try:
                serializable_sources = [
                    {
                        "name": src.get("name", "Unknown Source"), 
                        "url": src.get("url", "#"), 
                        "page_content": src.get("page_content", "") 
                    } 
                    for src in sources
                ]
                sources_json_str = json.dumps(serializable_sources)
                yield f"2:{sources_json_str}\n"
                print(f"[{request_id}] [Stream] Yielded SOURCES annotation: 2:{sources_json_str[:100]}...") # Log truncated sources
            except Exception as e_json:
                 print(f"[{request_id}] [Stream] Error serializing sources: {e_json}")
                 error_message = f"Error processing sources: {e_json}"
                 yield f"0:{json.dumps(error_message)}\n"
        else:
             print(f"[{request_id}] [Stream] No sources to stream.")
         # --------------------------

    except Exception as e:
        print(f"[{request_id}] [Stream] Error during answer generation/streaming: {e}")
        traceback.print_exc()
        error_message = f"Sorry, an internal error occurred during streaming: {e}"
        try:
            yield f"0:{json.dumps(error_message)}\n"
            print(f"[{request_id}] [Stream] Yielded ERROR chunk: 0:{json.dumps(error_message)}")
        except Exception as yield_e:
            print(f"[{request_id}] [Stream] CRITICAL: Failed even to yield error message: {yield_e}")
    finally:
        print(f"[{request_id}] --- EXITING stream_answer generator (Response Generated: {response_generated}) --- ")

# --- Add LLM Preprocessing Function --- 
async def preprocess_query(query: str) -> str:
    """Uses an LLM to correct grammar and spelling in the user query."""
    print(f"[Preprocess] Original query: {query}")
    try:
        # Use the same chat model configured for the RAG service
        # Explicitly pass the API key from settings
        llm = ChatOpenAI(
            model=settings.openai_chat_model, 
            temperature=0, 
            openai_api_key=settings.openai_api_key
        )
        
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Your primary task is to correct grammar and spelling mistakes in the user's input, focusing on the topic 'Chaos Chain'. "
                    "Correct potential typos towards 'Chaos' or 'Chain' where appropriate. "
                    "However, if the input is a short phrase that appears to be an instruction or follow-up command related to the previous context (e.g., 'eli5', 'summarize', 'explain more', 'in simple terms'), recognize this intent. "
                    "In such cases, return the instruction itself, perhaps normalized (e.g., 'Explain like I am 5' for 'eli5'), rather than correcting it or generating a response. "
                    "For standard queries needing correction, perform the correction. "
                    "Return ONLY the processed text (corrected query or instruction), with no preamble."
                )
            ),
            HumanMessage(content=query),
        ]
        
        response = await llm.ainvoke(messages)
        corrected_query = response.content
        
        # Basic check to ensure we got a string back
        if not isinstance(corrected_query, str):
            print("[Preprocess] Warning: LLM did not return a string. Using original query.")
            return query

        # Optional: Log if correction changed the query
        if corrected_query.strip().lower() != query.strip().lower():
             print(f"[Preprocess] Corrected query: {corrected_query}")
        else:
             print("[Preprocess] No corrections needed.")

        return corrected_query.strip()
        
    except Exception as e:
        print(f"[Preprocess] Error during preprocessing: {e}. Using original query.")
        return query # Fallback to original query on error
# --- End Preprocessing Function ---

@app.post("/api/chat")
async def handle_chat_request(
    raw_request: Request,
    rag_service_instance: RAGService = Depends(get_rag_service)
):
    request_id = str(uuid.uuid4())
    print(f"--- ENTERED handle_chat_request (ID: {request_id}) --- ")
    
    # Manually parse and validate the request body
    try:
        print(f"[{request_id}] Attempting to read request body...")
        body_bytes = await raw_request.body()
        body_str = body_bytes.decode('utf-8') # Decode for logging
        print(f"[{request_id}] Raw request body received (decoded): {body_str}")
        
        print(f"[{request_id}] Attempting to parse JSON...")
        body = json.loads(body_str) # Parse decoded string
        print(f"[{request_id}] JSON parsed successfully: {body}")
        
        print(f"[{request_id}] Attempting to validate payload with Pydantic...")
        request = VercelChatRequest.model_validate(body)
        print(f"[{request_id}] Pydantic validation successful: {request.dict()}")
        
    except json.JSONDecodeError as e:
        print(f"[{request_id}] Error: Invalid JSON received. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
    except ValidationError as e:
        print(f"[{request_id}] Pydantic Validation Error: {e.errors()}")
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        print(f"[{request_id}] Unexpected error during request parsing/validation: {e}")
        traceback.print_exc() # Print full traceback
        raise HTTPException(status_code=500, detail=f"Server error during request processing: {e}")
    
    # Extract Query
    query = ""
    try:
        print(f"[{request_id}] Attempting to extract query from validated request...")
        if not request.messages or request.messages[-1].role != 'user':
            print(f"[{request_id}] Validation Error: Last message not from user or no messages.")
            raise HTTPException(status_code=400, detail="Invalid request: Last message must be from user.")
        
        last_message = request.messages[-1]
        # Extract query based on content type
        if isinstance(last_message.content, str):
            query = last_message.content
            print(f"[{request_id}] Extracted query from string content: '{query}'")
        elif isinstance(last_message.content, list):
            text_part = next((part for part in last_message.content if part.type == 'text' and part.text is not None), None)
            if text_part:
                query = text_part.text
                print(f"[{request_id}] Extracted query from list content: '{query}'")
            else:
                 print(f"[{request_id}] No text part found in list content.")
        else:
             print(f"[{request_id}] Unexpected content type in last message: {type(last_message.content)}")
        
        if not query:
            print(f"[{request_id}] Validation Error: Query is empty after extraction.")
            raise HTTPException(status_code=400, detail="Query cannot be empty or content structure is invalid")
            
    except HTTPException: # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"[{request_id}] Unexpected error during query extraction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error during query extraction: {e}")

    # --- Preprocess Query --- 
    corrected_query = ""
    try:
        print(f"[{request_id}] Attempting to preprocess query: '{query}'")
        corrected_query = await preprocess_query(query)
        print(f"[{request_id}] Preprocessing completed. Result: '{corrected_query}'")
    except Exception as e:
        print(f"[{request_id}] Error during preprocessing query '{query}': {e}")
        traceback.print_exc()
        # Decide if we should proceed with original query or raise 500
        # For now, let's raise 500 as preprocessing is critical
        raise HTTPException(status_code=500, detail=f"Server error during query preprocessing: {e}")
    # ----------------------

    print(f"[{request_id}] --- BEFORE Creating StreamingResponse --- ")
    print(f"[{request_id}] [API] Returning StreamingResponse formatted for Vercel AI SDK using corrected query: '{corrected_query}'")
    try:
        return StreamingResponse(
            # Use the corrected query
            stream_answer(corrected_query, rag_service_instance, request_id), # Pass request_id
            media_type="text/plain"
        )
    except Exception as e:
         print(f"[{request_id}] Error creating or returning StreamingResponse: {e}")
         traceback.print_exc()
         # Fallback to a standard error response if streaming fails catastrophically
         return JSONResponse(
             status_code=500, 
             content={"detail": f"Server error during response streaming setup: {e}"}
         )

@app.get("/api/sources")
async def get_sources():
    """Endpoint to get repository source information, like last update time."""
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