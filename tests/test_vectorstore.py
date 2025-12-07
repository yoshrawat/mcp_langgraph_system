
import pytest

def test_vector_insert_and_search(mock_embeddings):
    from agent_app.core.rag.vectorstore import SQLiteVectorStore

    store = SQLiteVectorStore(db_path=":memory:")
    store.add_texts(["hello world"], [{"id":1}])

    results = store.search("hello", k=1)
    assert len(results) == 1
