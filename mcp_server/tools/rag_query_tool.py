from typing import List, Dict, Any

from langchain_ollama import OllamaEmbeddings

from ..vector_store.chroma_store import get_chroma

# Shared embedding model (same space as index)
_embeddings = OllamaEmbeddings(model="nomic-embed-text")


async def rag_query_tool(query: str, k: int = 3) -> Dict[str, Any]:
    """
    Run a semantic similarity search in ChromaDB.

    Args:
        query (str): User question or search phrase.
        k (int): Number of top matches to return. Default = 3.

    Returns:
        dict: {
            "query": "...",
            "k": 3,
            "matches": [
                {
                    "content": "...",
                    "metadata": {...},
                    "score": <float>
                }
            ]
        }
    """

    if not query:
        return {"matches": [], "message": "Empty query provided."}

    chroma = get_chroma()

    # Perform similarity search
    results = chroma.similarity_search_with_score(query, k=k)

    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        })

    return {
        "query": query,
        "k": k,
        "matches": formatted
    }
