from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a message in the conversation history."""
    role: str   # "human", "assistant", "tool"
    content: str


class ToolCallRecord(BaseModel):
    """Represents a single tool invocation for audit logging."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any


class AgentState(BaseModel):
    """
    LangGraph agent state container.

    This flows through each node in the graph:
      - router_node decides what to do next
      - llm_node generates text
      - tool_node executes MCP tools
    """

    messages: List[Message] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)

    # The "result" is set when the agent is ready to reply back to UI/API
    final_response: Optional[str] = None

    # LangGraph internal field for tracking progress
    next_step: Optional[str] = None

    # Session ID (UI sessions or FastAPI context)
    session_id: Optional[str] = None

    model_config = {
        "arbitrary_types_allowed": True
    }


def append_user_message(state: AgentState, text: str) -> AgentState:
    """Helper: append a human message."""
    state.messages.append(Message(role="human", content=text))
    return state


def append_assistant_message(state: AgentState, text: str) -> AgentState:
    """Helper: append an assistant message."""
    state.messages.append(Message(role="assistant", content=text))
    return state


def record_tool_call(
    state: AgentState,
    tool_name: str,
    args: Dict[str, Any],
    result: Any
) -> AgentState:
    """Record tool usage into state (audited separately)."""

    state.tool_calls.append(
        ToolCallRecord(
            tool_name=tool_name,
            arguments=args,
            result=result
        )
    )
    return state
