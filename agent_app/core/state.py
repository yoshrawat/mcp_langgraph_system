"""
Agent State for LangGraph + MCP Agent
-------------------------------------

This module defines the full AgentState object used by the graph.

The state stores:
 - Session ID
 - Message history
 - Pending tool call (from LLM)
 - Tool response (fed back into LLM)
 - Intermediate steps (for debugging + audit)
 - Final response (for Router → Done)

This class is a Pydantic model so it works cleanly with LangGraph.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================
# Message Object (used inside message list)
# ============================================================

class AgentMessage(BaseModel):
    role: str  # "human", "assistant", or "tool"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    tool_name: Optional[str] = None  # Only set for tool messages


# ============================================================
# Intermediate Steps (tool calls + results)
# ============================================================

class IntermediateStep(BaseModel):
    tool: str
    args: Dict[str, Any]
    result: Any
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================
# Agent State Model (central model for LangGraph)
# ============================================================

class AgentState(BaseModel):
    """
    The state that flows through each node in the LangGraph workflow.
    """

    # Session
    session_id: str

    # Chat history
    messages: List[AgentMessage] = Field(default_factory=list)

    # Raw user input for this turn
    user_input: Optional[str] = None

    # LLM → Tool call request
    pending_tool_call: Optional[Dict[str, Any]] = None

    # Tool → LLM result
    tool_response: Optional[Any] = None

    # History of tool invocations
    intermediate_steps: List[IntermediateStep] = Field(default_factory=list)

    # Final head response (Router → Done)
    final_response: Optional[str] = None

    # --------------------------------------------------------
    # Helper to append messages
    # --------------------------------------------------------

    def new_message(
        self,
        role: str,
        content: str,
        tool_name: Optional[str] = None
    ) -> AgentMessage:
        """
        Create and return a new AgentMessage.
        Note: State tracking is managed by the caller.
        """
        return AgentMessage(
            role=role,
            content=content,
            tool_name=tool_name
        )
