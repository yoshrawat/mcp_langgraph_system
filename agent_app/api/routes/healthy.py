from fastapi import APIRouter
from agent_app.core.agent_graph import AgentGraph

router = APIRouter(prefix="", tags=["Health"])


@router.get("/health")
async def health_check():
    """
    Liveness check: is FastAPI running?
    """
    return {"status": "ok", "message": "FastAPI application alive."}


@router.get("/ready")
async def readiness_check():
    """
    Readiness check:
      - Confirms AgentGraph can be constructed
      - Confirms MCP server can be probed for tool list
    """
    try:
        # Create lightweight agent instance
        agent = AgentGraph(
            mcp_endpoint="python mcp_server/run_server.py",
            model="llama3.2:latest"
        )

        # Try listing MCP tools (fast)
        from langchain_mcp_adapters.client import MCPClient

        async with MCPClient.from_stdio("python mcp_server/run_server.py") as client:
            tools = await client.list_tools()

        return {
            "status": "ready",
            "tools_available": tools,
        }

    except Exception as e:
        return {
            "status": "not_ready",
            "reason": str(e)
        }
