from __future__ import annotations

from typing import List
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    LangGraph agent state.
    """
    messages: Annotated[List[BaseMessage], add_messages]
