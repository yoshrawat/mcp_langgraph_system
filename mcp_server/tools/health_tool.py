"""
Health check tool for MCP server.

This tool is intentionally simple and synchronous-safe.
LangChain MCP adapter supports async tools, so this follows async signature.
"""

async def health_tool() -> dict:
    return {
        "status": "ok",
        "message": "MCP server alive and operational."
    }
