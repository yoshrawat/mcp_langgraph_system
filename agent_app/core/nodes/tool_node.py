"""
Tool Node for LangGraph Agent
-----------------------------

This node is responsible for:

- Receiving tool requests produced by the LLM node
- Calling the MCP server via langchain_mcp_adapters
- Logging each tool call into SQLite (audit_logger)
- Storing intermediate steps inside AgentState
"""

import asyncio
from typing import Dict, Any

from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MCPClient

from agent_app.core.state import AgentState, IntermediateStep
from agent_app.core.audit_logger import audit_logger


class MCPToolNode(ToolNode):
    """
    Custom ToolNode that:
    - Calls MCP tools via MCPClient.from_stdio()
    - Writes results to audit log
    - Stores intermediate steps into agent state
    """

    def __init__(self, mcp_command: str):
        super().__init__(tools=None)  # MCP dynamic tool loading
        self.mcp_command = mcp_command

    # -----------------------------------------------------------
    # Loading Tool List Dynamically from MCP Server
    # -----------------------------------------------------------
    async def _load_mcp_tools(self):
        """
        Load list of tools from MCP server and cache them.
        """
        async with MCPClient.from_stdio(self.mcp_command) as client:
            tools = await client.list_tools()
            self._tool_names = [t.name for t in tools.tools]
        return self._tool_names

    # -----------------------------------------------------------
    # Main Tool Execution Hook
    # -----------------------------------------------------------
    async def run(self, state: AgentState) -> AgentState:
        """
        Executes the tool requested by the LLM and updates AgentState.
        """
        tool_call = state.pending_tool_call

        if tool_call is None:
            # Nothing to execute
            return state

        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})

        # Ensure tools are loaded
        if not hasattr(self, "_tool_names"):
            await self._load_mcp_tools()

        # Validate tool exists
        if tool_name not in self._tool_names:
            raise ValueError(f"Tool '{tool_name}' not found in MCP server.")

        # -----------------------------------------------------------
        # Execute tool via MCP client
        # -----------------------------------------------------------
        async with MCPClient.from_stdio(self.mcp_command) as client:
            tool_result: Dict[str, Any] = await client.call_tool(
                name=tool_name,
                arguments=tool_args
            )

        # -----------------------------------------------------------
        # Save tool call to audit log
        # -----------------------------------------------------------
        audit_logger.write_record(
            session_id=state.session_id,
            tool_name=tool_name,
            arguments=tool_args,
            result=tool_result
        )

        # -----------------------------------------------------------
        # Store intermediate step in state
        # -----------------------------------------------------------
        new_step = IntermediateStep(
            tool=tool_name,
            args=tool_args,
            result=tool_result
        )

        state.intermediate_steps.append(new_step)

        # Clear pending tool call so LLM can process next step
        state.pending_tool_call = None

        # Provide tool result as next input for LLM node
        state.tool_response = tool_result

        return state
