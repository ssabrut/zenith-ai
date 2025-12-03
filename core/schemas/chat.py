from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user message")
    thread_id: str = Field(..., description="Unique session ID for conversational history")