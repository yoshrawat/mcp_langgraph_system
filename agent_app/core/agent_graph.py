"""
Agent Graph Builder for MCP + LangGraph Agent
----------------------------------------------

This module constructs the full LangGraph workflow:

   user → llm → router → tools → router → llm → ... → done

Nodes:
 - LLMNode: produces assistant replies or tool calls
 - RouterNode: decides next step
 - MCPToolNode: executes tools from MCP server
 - Done node: final output return

The AgentGraph class exposes:
 - arun(): async execution of one agent turn
 - run(): sync wrapper (used by Streamlit + FastAPI)
"""

import asyncio
from langgraph.graph import StateGraph, END

from agent_app.core.state import AgentState
from agent_app.core.nodes.llm_node import LLMNode
from agent_app.core.nodes.router_node import RouterNode
from agent_app.core.nodes.tool_node import MCPToolNode

from langchain_community.chat_models import ChatOllama


class AgentGraph:

    def __init__(self, mcp_endpoint: str, model: str = "llama3"):
        """
        Build the LangGraph agent with:
         - LLM node
         - Router node
         - MCP ToolNode
        """

        # -------------------------------------------------------
        # Initialize LLM
        # -------------------------------------------------------
        llm = ChatOllama(
            model=model,
            temperature=0.2,
            stream=False
        )

        self.llm_node = LLMNode(llm)
        self.router_node = RouterNode()
        self.tools_node = MCPToolNode(mcp_command=mcp_endpoint)

        # -------------------------------------------------------
        # Build LangGraph
        # -------------------------------------------------------
        builder = StateGraph(AgentState)

        builder.add_node("llm", self.llm_node)
        builder.add_node("router", self.router_node)
        builder.add_node("tools", self.tools_node)
        builder.add_node("done", lambda state: state)

        # Edges: LLM → Router
        builder.add_edge("llm", "router")

        # Edges: Tools → Router
        builder.add_edge("tools", "router")

        # Router → LLM
        builder.add_conditional_edges(
            "router",
            lambda o: o["next"],
            {
                "llm": "llm",
                "tools": "tools",
                "done": "done",
            }
        )

        # Graph start node
        builder.set_entry_point("llm")

        self._graph = builder.compile()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def arun(self, session_id: str, user_input: str,
                   prior_state: AgentState | None = None) -> AgentState:
        """
        Run one agent step asynchronously.

        The AgentState carries:
         - messages
         - tool responses
         - pending tool calls
         - final response
        """

        # Restore or initialize session state
        state = prior_state or AgentState(session_id=session_id)

        # Inject user input
        state.user_input = user_input

        # Execute graph
        result_state = await self._graph.ainvoke(state)

        return result_state

    def run(self, session_id: str, user_input: str,
            prior_state: AgentState | None = None) -> AgentState:
        """
        Sync wrapper around arun().
        Used by Streamlit and sometimes FastAPI.
        """

        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(
                self.arun(session_id, user_input, prior_state)
            )
        except RuntimeError:
            # No event loop running — normal case for Streamlit
            return asyncio.run(
                self.arun(session_id, user_input, prior_state)
            )
