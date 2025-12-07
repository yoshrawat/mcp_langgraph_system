
from __future__ import annotations
import asyncpg
from typing import Dict, Any, List
from agent_app.core.agent.interfaces import VectorStoreInterface
from agent_app.services.storage.vector_db.embeddings import embed_text


class PGVectorStore(VectorStoreInterface):

    def __init__(self, dsn: str):
        self.dsn = dsn

    async def _ensure_schema(self):
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(1536)
                );
            """)
        finally:
            await conn.close()

    async def add_document(self, text: str, metadata: Dict[str, Any] | None = None) -> str:
        await self._ensure_schema()
        vec = (await embed_text([text]))[0]
        doc_id = metadata.get("id") if metadata else f"pg_{hash(text)}"

        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute("""
                INSERT INTO documents (id, text, metadata, embedding)
                VALUES ($1, $2, $3, $4)
            """, doc_id, text, metadata, vec)
        finally:
            await conn.close()

        return doc_id

    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        await self._ensure_schema()
        q_emb = (await embed_text([query]))[0]

        conn = await asyncpg.connect(self.dsn)
        try:
            rows = await conn.fetch("""
                SELECT id, text, metadata,
                (embedding <#> $1) AS distance
                FROM documents
                ORDER BY distance
                LIMIT $2
            """, q_emb, k)
        finally:
            await conn.close()

        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "score": float(r["distance"])
            }
            for r in rows
        ]
