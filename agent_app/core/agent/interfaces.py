from __future__ import annotations

from typing import Protocol, Any, List, Dict
from langchain_core.messages import BaseMessage


class LLMProviderInterface(Protocol):
    """Interface for any LLM provider."""

    async def acomplete(self, messages: List[BaseMessage]) -> BaseMessage:
        ...


class VectorStoreInterface(Protocol):
    """Vector DB operations."""

    async def add_document(self, text: str, metadata: Dict[str, Any] | None = None) -> str:
        ...

    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        ...


class MCPClientInterface(Protocol):
    """Defines interaction with MCP server."""

    async def list_tools(self) -> List[Dict[str, Any]]:
        ...

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        ...


class AuditRepositoryInterface(Protocol):
    """Audit logging abstraction."""

    def log_tool_call(self, *, session_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        ...

    def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        ...
