
from __future__ import annotations
import chromadb
from typing import Dict, Any, List
from agent_app.core.agent.interfaces import VectorStoreInterface
from agent_app.services.storage.vector_db.embeddings import embed_text


class ChromaVectorStore(VectorStoreInterface):
    """
    Async-friendly wrapper around Chroma DB.
    """

    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path="data/chroma_store")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def add_document(self, text: str, metadata: Dict[str, Any] | None = None) -> str:
        vec = (await embed_text([text]))[0]
        doc_id = metadata.get("id") if metadata else None
        doc_id = doc_id or f"doc_{self.collection.count() + 1}"

        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            embeddings=[vec],
            ids=[doc_id]
        )
        return doc_id

    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        vec = (await embed_text([query]))[0]
        results = self.collection.query(
            query_embeddings=[vec],
            n_results=k
        )

        docs = []
        for idx in range(len(results["documents"][0])):
            docs.append({
                "id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "score": results["distances"][0][idx]
            })
        return docs
