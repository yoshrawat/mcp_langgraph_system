from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid

from agent_app.core.agent_graph import AgentGraph
from agent_app.core.state import AgentState
from agent_app.core.audit_logger import audit_logger


app = FastAPI(title="MCP LangGraph Agent API")

# ------------------------------------------------------------
# CORS (Postman, Streamlit UI, Web Apps)
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Global Session State Store
# ------------------------------------------------------------

SESSION_STORE: Dict[str, AgentState] = {}

AGENT = AgentGraph(
    mcp_endpoint="python mcp_server/run_server.py",
    model="llama3"
)

# ------------------------------------------------------------
# Request / Response Models
# ------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_input: str


class ChatResponse(BaseModel):
    session_id: str
    final_response: Optional[str]
    pending_tool_call: Optional[Dict[str, Any]]
    tool_response: Optional[Any]
    messages: list


# ------------------------------------------------------------
# (1) ORIGINAL ENDPOINT YOU REFERENCED
# ------------------------------------------------------------

@app.post("/chat/completion", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    This matches the earlier version I shared.
    Works identically to /chat — just a different route.
    """

    # Load or create session
    session_id = request.session_id or str(uuid.uuid4())
    state = SESSION_STORE.get(session_id) or AgentState(session_id=session_id)

    # Execute one agent turn
    new_state = await AGENT.arun(
        session_id=session_id,
        user_input=request.user_input,
        prior_state=state
    )

    # Save updated state
    SESSION_STORE[session_id] = new_state

    return ChatResponse(
        session_id=session_id,
        final_response=new_state.final_response,
        pending_tool_call=new_state.pending_tool_call,
        tool_response=new_state.tool_response,
        messages=[m.dict() for m in new_state.messages]
    )


# ------------------------------------------------------------
# (2) NEW SIMPLIFIED ENDPOINT
# ------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Identical to /chat/completion, but simpler URL.
    """

    session_id = request.session_id or str(uuid.uuid4())
    state = SESSION_STORE.get(session_id) or AgentState(session_id=session_id)

    new_state = await AGENT.arun(
        session_id=session_id,
        user_input=request.user_input,
        prior_state=state
    )

    SESSION_STORE[session_id] = new_state

    return ChatResponse(
        session_id=session_id,
        final_response=new_state.final_response,
        pending_tool_call=new_state.pending_tool_call,
        tool_response=new_state.tool_response,
        messages=[m.dict() for m in new_state.messages]
    )


# ------------------------------------------------------------
# Retrieve Full Session State
# ------------------------------------------------------------

@app.get("/state/{session_id}")
def get_state(session_id: str):
    state = SESSION_STORE.get(session_id)
    if not state:
        return {"error": "Invalid session_id"}
    return state.dict()


# ------------------------------------------------------------
# Audit Log Endpoint
# ------------------------------------------------------------

@app.get("/tools/logs")
def tool_logs(session_id: Optional[str] = None, limit: int = 50):
    logs = audit_logger.list_logs(session_id=session_id, limit=limit)
    return {
        "count": len(logs),
        "logs": logs
    }
