from fastapi import APIRouter, HTTPException
import uuid

from agent_app.core.agent_graph import AgentGraph
from agent_app.core.state import AgentState
from agent_app.api.models import ChatRequest, ChatResponse


router = APIRouter(prefix="/chat", tags=["Chat"])

# ---------------------------------------------------------------------
# Internal in-memory session store
# ---------------------------------------------------------------------
SESSION_STORE = {}

# Shared global agent instance
AGENT = AgentGraph(
    mcp_endpoint="python mcp_server/run_server.py",
    model="llama3.2:latest"
)


@router.post("/completion", response_model=ChatResponse)
async def chat_completion(payload: ChatRequest):
    """
    Main multi-turn conversational endpoint.
    Uses LangGraph agent + MCP tools.
    """

    # -------------------------------
    # Session Setup
    # -------------------------------
    if payload.session_id is None:
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = AgentState(session_id=session_id)
    else:
        session_id = payload.session_id
        if session_id not in SESSION_STORE:
            SESSION_STORE[session_id] = AgentState(session_id=session_id)

    state = SESSION_STORE[session_id]

    # -------------------------------
    # Run LangGraph Agent
    # -------------------------------
    result_state = await AGENT.arun(
        session_id=session_id,
        user_input=payload.message,
        prior_state=state
    )

    # Persist updated state
    SESSION_STORE[session_id] = result_state

    return ChatResponse(
        session_id=session_id,
        response=result_state.final_response,
        messages=[m.model_dump() for m in result_state.messages]
    )
