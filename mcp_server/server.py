from mcp_server.server import Server

# Import tools
from .tools.health_tool import health_tool
from .tools.api_fetch_tool import api_fetch_tool
from .tools.rag_index_tool import rag_index_tool
from .tools.rag_query_tool import rag_query_tool


def build_mcp_server() -> Server:
    """
    Construct the MCP server with all registered tools.

    This server exposes:
      - health_check
      - fetch_api_data
      - rag_index
      - rag_query

    The server is unified (single namespace) following the design choice (A, A).
    """

    server = Server(name="mcp-rag-server")

    # Register health tool
    server.register_tool(
        "health_check",
        health_tool,
        description="Check server health status."
    )

    # Register API fetch tool
    server.register_tool(
        "fetch_api_data",
        api_fetch_tool,
        description="Fetch remote API data and return content."
    )

    # Register RAG indexer tool
    server.register_tool(
        "rag_index",
        rag_index_tool,
        description="Index documents into ChromaDB using Ollama embeddings."
    )

    # Register RAG search tool
    server.register_tool(
        "rag_query",
        rag_query_tool,
        description="Search indexed embeddings in ChromaDB."
    )

    return server
