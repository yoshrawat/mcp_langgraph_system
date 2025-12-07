import os
from typing import Optional

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


# ----------------------------
# Disable Chroma anonymous telemetry
# ----------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "false"


# ----------------------------
# Chroma storage location
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "..", "chroma_db")


# ----------------------------
# Shared embedding model
# ----------------------------
_embeddings = OllamaEmbeddings(model="nomic-embed-text")


# ----------------------------
# Cached vector store instance
# ----------------------------
_chroma_instance: Optional[Chroma] = None


def get_chroma() -> Chroma:
    """
    Get or initialize a persistent ChromaDB instance.

    Returns:
        Chroma: A vector store ready for similarity search and upserts.
    """

    global _chroma_instance

    if _chroma_instance is None:
        os.makedirs(CHROMA_PATH, exist_ok=True)

        # Initialize persistent Chroma
        _chroma_instance = Chroma(
            collection_name="mcp_rag_store",
            embedding_function=_embeddings,
            persist_directory=CHROMA_PATH,
        )

    return _chroma_instance
