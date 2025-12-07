import asyncio
from typing import Optional, Dict, Any

from langgraph.graph import StateGraph, END

from .state import AgentState, append_user_message
from .nodes.router_node import RouterNode
from .nodes.llm_node import LLMNode
from .nodes.tool_node import ToolNode


class AgentGraph:
    """
    This class builds and runs the LangGraph agent.

    Nodes:
      - router   → determines whether to use LLM or tool next
      - llm      → generates assistant response
      - tool     → executes MCP tool and returns results

    Flow:
        user_input → router → (llm | tool) → router → ... → llm → done
    """

    def __init__(
        self,
        mcp_endpoint: str = "python mcp_server/run_server.py",
        model: str = "llama3"
    ):
        # Instantiate nodes
        self.router = RouterNode()
        self.llm_node = LLMNode(model=model)
        self.tool_node = ToolNode(mcp_endpoint=mcp_endpoint)

        # Build graph
        self._graph = self._build_graph()

    # ----------------------------------------------------------------------
    # Build LangGraph structure
    # ----------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("router", self.router.run)
        graph.add_node("llm", self.llm_node.run)
        graph.add_node("tool", self.tool_node.run)

        # Define routing edges
        graph.add_edge("router", "llm")
        graph.add_edge("router", "tool")

        graph.add_edge("tool", "router")
        graph.add_edge("llm", END)

        # Graph entry point
        graph.set_entry_point("router")

        return graph.compile()

    # ----------------------------------------------------------------------
    # High-level agent entrypoints
    # ----------------------------------------------------------------------
    async def arun(
        self,
        session_id: str,
        user_input: str,
        prior_state: Optional[AgentState] = None
    ) -> AgentState:
        """
        Async agent execution (main method).

        Args:
            session_id (str): UI/API session identifier
            user_input (str): user query
            prior_state (AgentState): previous conversation context

        Returns:
            AgentState after full graph execution
        """

        # Initialize state if first turn
        if prior_state is None:
            state = AgentState(session_id=session_id)
        else:
            state = prior_state

        # Add the user message
        append_user_message(state, user_input)

        # Execute LangGraph
        result_state = await self._graph.ainvoke(state)

        return result_state

    # ----------------------------------------------------------------------
    # Sync wrapper (Streamlit safe)
    # ----------------------------------------------------------------------
    def run(
        self,
        session_id: str,
        user_input: str,
        prior_state: Optional[AgentState] = None
    ) -> AgentState:

        try:
            loop = asyncio.get_running_loop()
            # If inside Streamlit event loop → run as task
            coro = self.arun(session_id, user_input, prior_state)
            return loop.run_until_complete(coro)
        except RuntimeError:
            # No running loop → create one
            return asyncio.run(
                self.arun(session_id, user_input, prior_state)
            )
