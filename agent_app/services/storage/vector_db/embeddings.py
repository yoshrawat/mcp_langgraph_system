
from __future__ import annotations
import httpx
from typing import List
from agent_app.core.config.settings import get_settings


async def embed_text(texts: List[str]) -> List[List[float]]:
    """
    Compute embeddings using Ollama embeddings endpoint.
    """
    settings = get_settings()
    url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"

    payload = {
        "model": settings.EMBEDDING_MODEL,
        "input": texts,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return data.get("embeddings", [])
