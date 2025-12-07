
from __future__ import annotations
import aiosqlite
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from agent_app.core.config.settings import get_settings
from agent_app.core.agent.interfaces import AuditRepositoryInterface


class AuditLogEntry(BaseModel):
    id: int
    session_id: str
    tool_name: str
    arguments: Dict[str, Any]


class AuditRepository(AuditRepositoryInterface):
    """
    Async SQLite audit logger for MCP tool calls.
    """

    def __init__(self):
        settings = get_settings()
        self.db_path = f"data/{settings.SQLITE_AUDIT_PATH}"

    async def _ensure_schema(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments TEXT NOT NULL
                );
            """)
            await db.commit()

    def _serialize_args(self, args: Dict[str, Any]) -> str:
        return json.dumps(args)

    async def log_tool_call(self, *, session_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO audit_logs (session_id, tool_name, arguments) VALUES (?, ?, ?)",
                (session_id, tool_name, self._serialize_args(args))
            )
            await db.commit()

    async def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, session_id, tool_name, arguments FROM audit_logs ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = await cursor.fetchall()

        return [
            AuditLogEntry(
                id=row[0],
                session_id=row[1],
                tool_name=row[2],
                arguments=json.loads(row[3])
            ).dict()
            for row in rows
        ]
