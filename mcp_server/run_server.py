"""
Main entrypoint for launching the MCP server.
This file allows: python -m mcp_server.run_server
"""

from .server import run

if __name__ == "__main__":
    run()
