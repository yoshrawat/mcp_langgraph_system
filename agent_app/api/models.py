from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request body for POST /chat/completion
    """
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID. If omitted, API will create a new session."
    )
    message: str = Field(
        ..., 
        description="User message to send to the LangGraph agent."
    )


class ChatMessage(BaseModel):
    """
    A single message in the chat history.
    Mirrors agent_app.core.state.Message, but used for API output.
    """
    role: str
    content: str


class ChatResponse(BaseModel):
    """
    Response returned by POST /chat/completion
    """
    session_id: str = Field(
        ..., 
        description="Session identifier used for multi-turn conversation."
    )
    response: str = Field(
        ..., 
        description="Final assistant response from LangGraph after reasoning/tool calls."
    )
    messages: List[Dict[str, Any]] = Field(
        ..., 
        description="Full conversation history including human/assistant/tool messages."
    )
