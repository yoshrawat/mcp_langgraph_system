from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid

from agent_app.core.agent_graph import AgentGraph
from agent_app.core.state import AgentState
from agent_app.core.audit_logger import audit_logger

from agent_app.api.models import ChatRequest, ChatResponse


# ----------------------------------------------------------------------
# FastAPI initialization
# ----------------------------------------------------------------------
app = FastAPI(
    title="MCP LangGraph System",
    description="API interface for LangGraph Agent + MCP Tools",
    version="1.0.0"
)


# ----------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Customize for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Global in-memory session store
# ----------------------------------------------------------------------
SESSION_STORE = {}

# Create a single agent instance for all API calls
AGENT = AgentGraph(
    mcp_endpoint="python mcp_server/run_server.py",
    model="llama3.2:latest"
)


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FastAPI server running."}


@app.post("/chat/completion", response_model=ChatResponse)
async def chat_completion(payload: ChatRequest):
    """
    Main entrypoint for programmatic chat.
    Handles multiple-turn conversations per session.
    """

    # Create session if needed
    if payload.session_id is None:
        session_id = str(uuid.uuid4())
        SESSION_STORE[session_id] = AgentState(session_id=session_id)
    else:
        session_id = payload.session_id
        if session_id not in SESSION_STORE:
            SESSION_STORE[session_id] = AgentState(session_id=session_id)

    state = SESSION_STORE[session_id]

    # Run agent
    result_state = await AGENT.arun(
        session_id=session_id,
        user_input=payload.message,
        prior_state=state
    )

    # Save updated session
    SESSION_STORE[session_id] = result_state

    return ChatResponse(
        session_id=session_id,
        response=result_state.final_response,
        messages=[m.model_dump() for m in result_state.messages]
    )


@app.get("/tools/logs")
async def list_tool_logs(session_id: str | None = None, limit: int = 50):
    """
    Retrieve audit logs for debugging, analytics, and observability.
    """
    logs = audit_logger.list_logs(session_id=session_id, limit=limit)
    return {"count": len(logs), "logs": logs}


@app.get("/mcp/tools/list")
async def mcp_list_tools():
    """
    Simple debugging endpoint to ask MCP server for its tool list.
    Useful for Postman or CI-based validation.
    """
    from langchain_mcp_adapters.client import load_mcp_tools, create_session
    from langchain_mcp_adapters.sessions import StdioConnection

    connection: StdioConnection = {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "mcp_server.run_server"],
    }

    async for session in create_session(connection):
        tools = await load_mcp_tools(session, connection=connection)
        return {
            "tools": [{"name": tool.name, "description": tool.description} for tool in tools]
        }
