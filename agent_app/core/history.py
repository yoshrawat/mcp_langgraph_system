from typing import List
from .state import Message, AgentState


class ConversationHistory:
    """
    Simple conversation history handler.

    This manages:
      - appending human/assistant messages
      - preserving order
      - converting into LLM-friendly format
      - synchronizing with AgentState
    """

    def __init__(self):
        self._messages: List[Message] = []

    # ----------------------------
    # Basic operations
    # ----------------------------

    def add_user(self, text: str):
        self._messages.append(Message(role="human", content=text))

    def add_assistant(self, text: str):
        self._messages.append(Message(role="assistant", content=text))

    def add_tool(self, text: str):
        self._messages.append(Message(role="tool", content=text))

    # ----------------------------
    # Retrieval
    # ----------------------------
    def get(self) -> List[Message]:
        return list(self._messages)

    def last(self) -> Message | None:
        return self._messages[-1] if self._messages else None

    def clear(self):
        self._messages.clear()

    # ----------------------------
    # AgentState integration
    # ----------------------------
    def load_from_state(self, state: AgentState):
        self._messages = list(state.messages)

    def apply_to_state(self, state: AgentState) -> AgentState:
        state.messages = list(self._messages)
        return state

    # ----------------------------
    # LLM-friendly formatting
    # ----------------------------
    def as_llm_messages(self) -> List[dict]:
        """
        Convert into messages compatible with LangChain LLMs:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
        """
        mapped = []
        for m in self._messages:
            role_map = {
                "human": "user",
                "assistant": "assistant",
                "tool": "tool",
            }
            mapped.append({
                "role": role_map.get(m.role, m.role),
                "content": m.content
            })
        return mapped
