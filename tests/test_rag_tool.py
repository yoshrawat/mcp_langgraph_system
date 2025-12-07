
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_rag_tool(mock_llm, mock_embeddings):
    from agent_app.core.tools.rag_tool import RAGTool

    tool = RAGTool()

    with patch("agent_app.core.tools.rag_tool.fetch_api_data", AsyncMock(return_value=[{"content":"hello world"}])):
        result = await tool.run({"query":"hello"})
        assert "answer" in result
