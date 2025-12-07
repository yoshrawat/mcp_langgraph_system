"""
Router Node for LangGraph Agent
-------------------------------

This node is responsible for deciding the next step of execution:

- "tool" → when the user input or LLM indicates a tool call
- "llm"  → when the agent should respond normally
- "done" → when a final answer already exists

This ensures the agent follows the correct flow of:
  LLM → Router → Tool → Router → LLM → ... → Done
"""

from agent_app.core.state import AgentState


class RouterNode:
    """
    Simple intent-based router for LangGraph agent.
    """

    # Keywords that suggest a tool is needed.
    TRIGGER_KEYWORDS = [
        "search", "fetch", "lookup", "query",
        "rag", "index", "tool", "api"
    ]

    def __call__(self, state: AgentState):
        """
        Determine which node should execute next.
        """

        # 1. If final response already exists → conversation finishes.
        if state.final_response is not None:
            return {"next": "done"}

        # 2. If LLM decided a tool should be called → send to tool node.
        if state.pending_tool_call is not None:
            return {"next": "tools"}

        # 3. If user typed something that indicates needing a tool.
        user_input = (state.user_input or "").lower()

        if any(keyword in user_input for keyword in self.TRIGGER_KEYWORDS):
            return {"next": "tools"}

        # 4. Otherwise → LLM should continue.
        return {"next": "llm"}
