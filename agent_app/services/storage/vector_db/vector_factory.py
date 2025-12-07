
from __future__ import annotations
from agent_app.core.config.settings import get_settings
from agent_app.core.agent.interfaces import VectorStoreInterface

# Backends
from agent_app.services.storage.vector_db.chroma_backend import ChromaVectorStore
from agent_app.services.storage.vector_db.sqlite_backend import SQLiteVectorStore
from agent_app.services.storage.vector_db.pgvector_backend import PGVectorStore
from agent_app.services.storage.vector_db.faiss_backend import FAISSVectorStore


def get_vector_store() -> VectorStoreInterface:
    """
    Factory for choosing the vector backend based on config.
    """
    settings = get_settings()
    backend = settings.VECTOR_BACKEND.lower()

    if backend == "chroma":
        return ChromaVectorStore()

    if backend == "sqlite":
        return SQLiteVectorStore()

    if backend == "pgvector":
        # DSN must be provided via env
        raise ValueError("PGVector requires DSN configuration.")

    if backend == "faiss":
        return FAISSVectorStore()

    raise ValueError(f"Unknown vector backend: {backend}")
