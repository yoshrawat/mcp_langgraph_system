
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_mcp_tool_call(mock_llm):
    from agent_app.core.tools.mcp_client import MCPToolExecutor

    executor = MCPToolExecutor("mock_session")

    with patch.object(executor, "call_mcp_tool", AsyncMock(return_value={"result": "ok"})):
        result = await executor.execute("search_api_results", {"q": "hello"})
        assert result["result"] == "ok"
