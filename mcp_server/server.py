import asyncio
from mcp.server import Server
from mcp.types import Tool, ToolOutput

# Import tool implementations
from mcp_server.tools.health_tool import health_check_tool
from mcp_server.tools.api_fetch_tool import fetch_api_data_tool
from mcp_server.tools.rag_index_tool import rag_index_tool
from mcp_server.tools.rag_query_tool import rag_query_tool


async def start_server_stdio():
    """
    Start MCP server using STDIO transport.
    This is required for LangGraph's MCPClient.from_stdio().
    """

    server = Server(name="mcp-langgraph-server")

    # ------------------------------------------------------------------
    # Register MCP Tools
    # ------------------------------------------------------------------
    server.register_tool(
        Tool(
            name="health_check",
            description="Returns OK if MCP server is alive.",
            input_schema=health_check_tool.input_schema(),
            output_schema=health_check_tool.output_schema(),
            handler=health_check_tool
        )
    )

    server.register_tool(
        Tool(
            name="fetch_api_data",
            description="Fetch JSON from any public API endpoint.",
            input_schema=fetch_api_data_tool.input_schema(),
            output_schema=fetch_api_data_tool.output_schema(),
            handler=fetch_api_data_tool
        )
    )

    server.register_tool(
        Tool(
            name="rag_index",
            description="Index text into ChromaDB for retrieval.",
            input_schema=rag_index_tool.input_schema(),
            output_schema=rag_index_tool.output_schema(),
            handler=rag_index_tool
        )
    )

    server.register_tool(
        Tool(
            name="rag_query",
            description="Query embeddings from ChromaDB.",
            input_schema=rag_query_tool.input_schema(),
            output_schema=rag_query_tool.output_schema(),
            handler=rag_query_tool
        )
    )

    # ------------------------------------------------------------------
    # Start MCP server over STDIO
    # ------------------------------------------------------------------
    print("[MCP] Server starting on STDIO...")

    await server.run_stdio()

    print("[MCP] Server stopped.")


def run():
    """Entry point for synchronous startup."""
    asyncio.run(start_server_stdio())
