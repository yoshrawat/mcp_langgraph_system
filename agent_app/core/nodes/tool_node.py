from typing import Optional
from ..state import AgentState


class ToolNode:
    """
    Router node decides the next step for the agent:
      - "tool" → if user asks to search, fetch, query, or index
      - "llm"  → if the agent should respond naturally
      - "done" → if final_response already exists
    """

    TRIGGER_KEYWORDS = [
        "search", "fetch", "lookup", "query",
        "rag", "index", "tool", "api"
    ]

    def _detect_tool_intent(self, text: str) -> bool:
        """Return True if query contains any tool-related keywords."""
        lowered = text.lower()
        return any(kw in lowered for kw in self.TRIGGER_KEYWORDS)

    async def run(self, state: AgentState) -> AgentState:
        # ------------------------------------------------------------------
        # If we already have a final response, we are done.
        # ------------------------------------------------------------------
        if state.final_response:
            state.next_step = "done"
            return state

        # ------------------------------------------------------------------
        # Get last user message
        # ------------------------------------------------------------------
        if not state.messages:
            state.next_step = "llm"
            return state

        last_msg = state.messages[-1]

        if last_msg.role != "human":
            state.next_step = "llm"
            return state

        user_text = last_msg.content

        # ------------------------------------------------------------------
        # Check if tool is needed
        # ------------------------------------------------------------------
        if self._detect_tool_intent(user_text):
            # Next node: ToolNode
            state.next_step = "tool"
        else:
            # Default: use LLM
            state.next_step = "llm"

        return state
