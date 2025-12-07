
from __future__ import annotations
import re
from typing import List, Dict, Any, Callable, Awaitable
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import ToolMessage
from agent_app.services.mcp.client import MCPAsyncClient
from agent_app.services.storage.vector_db.vector_factory import get_vector_store
from agent_app.services.storage.audit_repo import AuditRepository
from agent_app.services.storage.vector_db.embeddings import embed_text


def normalize_name(name: str) -> str:
    """Convert a tool name into snake_case."""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')


class RAGSearchInput(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(4, description="Number of results")


async def rag_search(query: str, top_k: int = 4) -> Dict[str, Any]:
    """Perform RAG search using the vector store."""
    vector_store = get_vector_store()
    results = await vector_store.similarity_search(query, k=top_k)
    return {"results": results}


async def wrap_tool_with_audit(
    tool_func: Callable[..., Awaitable[Any]],
    audit_repo: AuditRepository,
    session_id: str,
    tool_name: str
):
    """Return a wrapped coroutine that logs audit entries before executing the tool."""
    async def wrapper(**kwargs):
        await audit_repo.log_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            args=kwargs
        )
        return await tool_func(**kwargs)
    return wrapper


async def load_all_tools(
    mcp_client: MCPAsyncClient,
    audit_repo: AuditRepository,
    session_id: str
) -> List[StructuredTool]:

    tools: List[StructuredTool] = []

    # -------------------------
    # Add RAG Search Tool
    # -------------------------
    async def rag_wrapper(query: str, top_k: int = 4):
        return await rag_search(query=query, top_k=top_k)

    rag_wrapped = await wrap_tool_with_audit(
        rag_wrapper,
        audit_repo,
        session_id,
        "rag_search"
    )

    rag_tool = StructuredTool.from_function(
        name="rag_search",
        description="Search the vector database for relevant documents.",
        coroutine=rag_wrapped
    )
    tools.append(rag_tool)

    # -------------------------
    # Add MCP Tools
    # -------------------------
    mcp_tools = await mcp_client.list_tools()

    for t in mcp_tools:
        raw_name = t.get("name", "mcp_tool")
        normalized = normalize_name(raw_name)
        description = t.get("description", f"MCP tool {raw_name}")

        async def _make_exec(tool_name: str):
            async def exec_tool(**kwargs):
                return await mcp_client.call_tool(tool_name, kwargs)
            return exec_tool

        exec_func = await _make_exec(raw_name)

        wrapped = await wrap_tool_with_audit(
            exec_func,
            audit_repo,
            session_id,
            normalized
        )

        tool = StructuredTool.from_function(
            name=normalized,
            description=description,
            coroutine=wrapped,
        )
        tools.append(tool)

    return tools
