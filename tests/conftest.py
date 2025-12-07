
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

# Mock LLM response
@pytest.fixture
def mock_llm():
    with patch("agent_app.core.llm.ollama_client") as mock:
        mock.return_value = AsyncMock()
        mock.return_value.generate = AsyncMock(return_value={"text":"mocked answer"})
        yield mock

# Mock embeddings
@pytest.fixture
def mock_embeddings():
    with patch("agent_app.core.rag.embeddings") as mock:
        mock.embed_query = lambda x: [0.1,0.2,0.3]
        mock.embed_documents = lambda x: [[0.1,0.2,0.3] for _ in x]
        yield mock

# FastAPI Test Client
@pytest.fixture
def api_client():
    from agent_app.api.fastapi_app import app
    return TestClient(app)
