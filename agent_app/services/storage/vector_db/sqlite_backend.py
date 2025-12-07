
from __future__ import annotations
import aiosqlite
import json
from typing import Dict, Any, List
from agent_app.core.config.settings import get_settings
from agent_app.core.agent.interfaces import VectorStoreInterface
from agent_app.services.storage.vector_db.embeddings import embed_text


class SQLiteVectorStore(VectorStoreInterface):
    """
    Async SQLite-based vector store.
    """

    def __init__(self):
        settings = get_settings()
        self.db_path = f"data/{settings.SQLITE_VECTOR_PATH}"

    async def _ensure_schema(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB NOT NULL
                );
            """)
            await db.commit()

    async def add_document(self, text: str, metadata: Dict[str, Any] | None = None) -> str:
        await self._ensure_schema()
        emb = (await embed_text([text]))[0]

        doc_id = metadata.get("id") if metadata else None
        doc_id = doc_id or f"doc_{hash(text)}"

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO vectors (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
                (
                    doc_id,
                    text,
                    json.dumps(metadata or {}),
                    json.dumps(emb)
                )
            )
            await db.commit()

        return doc_id

    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        await self._ensure_schema()
        q_emb = (await embed_text([query]))[0]

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT id, text, metadata, embedding FROM vectors")
            rows = await cursor.fetchall()

        scored = []
        for row in rows:
            emb = json.loads(row[3])
            score = sum(a * b for a, b in zip(q_emb, emb))

            scored.append({
                "id": row[0],
                "text": row[1],
                "metadata": json.loads(row[2]),
                "score": score,
            })

        return sorted(scored, key=lambda x: x["score"], reverse=True)[:k]
