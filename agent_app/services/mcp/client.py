
from __future__ import annotations
import anyio
import json
from typing import List, Dict, Any
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from agent_app.core.config.settings import get_settings
from agent_app.core.agent.interfaces import MCPClientInterface


class MCPAsyncClient(MCPClientInterface):
    """
    Async MCP client used by the agent to call tools.
    """

    async def list_tools(self) -> List[Dict[str, Any]]:
        settings = get_settings()

        async with streamablehttp_client(settings.MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                tools = await session.list_tools()
                return tools.get("tools", [])

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        settings = get_settings()

        async with streamablehttp_client(settings.MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                result = await session.call_tool(
                    name=name,
                    arguments=args or {}
                )
                return json.loads(json.dumps(result, default=str))
