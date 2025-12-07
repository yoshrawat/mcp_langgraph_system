import asyncio
from mcp.server import FastMCP
from pydantic import BaseModel

# Import tool implementations
from .tools.health_tool import health_check_tool
from .tools.api_fetch_tool import fetch_api_data_tool
from .tools.rag_index_tool import rag_index_tool
from .tools.rag_query_tool import rag_query_tool


# Initialize FastMCP server
mcp = FastMCP(name="mcp-langgraph-server")


# ------------------------------------------------------------------
# Tool 1: Health Check
# ------------------------------------------------------------------
@mcp.tool()
async def health_check():
    """Returns OK if MCP server is alive."""
    return await health_check_tool(input={})


# ------------------------------------------------------------------
# Tool 2: Fetch API Data
# ------------------------------------------------------------------
@mcp.tool()
async def fetch_api_data(url: str) -> str:
    """Fetch JSON from any public API endpoint."""
    from mcp_server.tools.api_fetch_tool import FetchAPIDataInput
    return await fetch_api_data_tool(input=FetchAPIDataInput(url=url))


# ------------------------------------------------------------------
# Tool 3: RAG Index
# ------------------------------------------------------------------
@mcp.tool()
async def rag_index(texts: list, metadatas: list = None, namespace: str = "default") -> dict:
    """Index text into ChromaDB for retrieval."""
    from mcp_server.tools.rag_index_tool import RAGIndexInput
    result = await rag_index_tool(input=RAGIndexInput(
        texts=texts,
        metadatas=metadatas,
        namespace=namespace
    ))
    return result.model_dump()


# ------------------------------------------------------------------
# Tool 4: RAG Query
# ------------------------------------------------------------------
@mcp.tool()
async def rag_query(query: str, namespace: str = "default", k: int = 5) -> dict:
    """Query embeddings from ChromaDB."""
    from mcp_server.tools.rag_query_tool import RAGQueryInput
    result = await rag_query_tool(input=RAGQueryInput(
        query=query,
        namespace=namespace,
        k=k
    ))
    return result.model_dump()


async def start_server_stdio():
    """
    Start MCP server using STDIO transport.
    This is required for LangGraph's MCPClient.from_stdio().
    """
    print("[MCP] Server starting on STDIO...")
    await mcp.run_stdio_async()
    print("[MCP] Server stopped.")


def run():
    """Entry point for synchronous startup."""
    asyncio.run(start_server_stdio())
