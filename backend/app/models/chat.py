from pydantic import BaseModel
from typing import List, Optional, Literal

# Model for individual messages matching Vercel AI SDK format
class VercelChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'function', 'data', 'tool']
    content: str
    # Add other potential fields if needed (e.g., name, tool_calls)

# Model for the request body sent by useChat
class VercelChatRequest(BaseModel):
    messages: List[VercelChatMessage]
    # The SDK might send other fields like 'id', 'previewToken', 'data' - add if necessary
    # For now, we only care about messages

class SourceDocument(BaseModel):
    content: str
    metadata: dict = {}

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] = [] 