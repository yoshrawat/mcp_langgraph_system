
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class AgentRequest(BaseModel):
    session_id: str = Field(..., description="User session identifier")
    query: str = Field(..., description="User query to the agent")
    metadata: Optional[Dict[str, Any]] = None
