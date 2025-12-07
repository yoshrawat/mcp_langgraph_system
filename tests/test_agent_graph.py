
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_agent_graph(mock_llm):
    from agent_app.core.agent.graph import build_agent_graph
    from agent_app.core.agent.state import AgentState

    graph_factory = build_agent_graph()
    graph = await graph_factory("test-session")

    initial = AgentState(messages=[{"type": "human", "content": "hello"}])

    result = await graph.ainvoke(initial)
    assert "final_answer" in result
