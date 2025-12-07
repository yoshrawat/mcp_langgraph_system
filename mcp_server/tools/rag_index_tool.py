from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from ..vectorstore.chroma_store import get_chroma


# Embedding model (shared)
_embeddings = OllamaEmbeddings(model="nomic-embed-text")


async def rag_index_tool(docs: List[Dict[str, str]]) -> dict:
    """
    Index documents into ChromaDB using Ollama embeddings.

    Args:
        docs (list): Items like:
            {
                "id": "...",
                "content": "...",
                "source": "api" | "ui" | ...
            }

    Workflow:
        - Split docs into small chunks
        - Embed using Ollama ("nomic-embed-text")
        - Store into persistent Chroma database

    Returns:
        {"indexed": <num_chunks>}
    """

    if not docs:
        return {"indexed": 0, "message": "No docs provided."}

    chroma = get_chroma()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    texts = []
    metadatas = []

    for doc in docs:
        content = doc.get("content")
        if not content:
            continue

        # Split into chunks
        chunks = splitter.split_text(content)

        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "source": doc.get("source", "unknown"),
                "doc_id": doc.get("id"),
            })

    if not texts:
        return {"indexed": 0, "message": "No chunkable text found."}

    # Persist into Chroma
    chroma.add_texts(texts=texts, metadatas=metadatas)

    return {
        "indexed": len(texts),
        "message": f"Indexed {len(texts)} text chunks into Chroma."
    }
