import sqlite3
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import os

DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "audit_logs.sqlite3"
)


class ToolAuditLogger:
    """
    A simple SQLite-based audit logging system that records all MCP tool calls.

    Schema:
        id INTEGER PRIMARY KEY
        timestamp TEXT (ISO8601)
        session_id TEXT
        tool_name TEXT
        arguments TEXT (JSON)
        result TEXT (JSON)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_initialized()

    # -----------------------------------
    # Initialization
    # -----------------------------------
    def _ensure_initialized(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    tool_name TEXT NOT NULL,
                    arguments TEXT,
                    result TEXT
                );
                """
            )
            conn.commit()

    # -----------------------------------
    # Logging
    # -----------------------------------
    def log(
        self,
        session_id: Optional[str],
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
    ):
        """Insert a log record into SQLite."""

        ts = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO tool_logs (timestamp, session_id, tool_name, arguments, result)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    session_id,
                    tool_name,
                    json.dumps(arguments),
                    json.dumps(result),
                )
            )
            conn.commit()

    # -----------------------------------
    # Query logs
    # -----------------------------------
    def list_logs(
        self,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:

        query = "SELECT timestamp, session_id, tool_name, arguments, result FROM tool_logs"
        params = []

        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        output = []
        for ts, sid, tool, args, res in rows:
            output.append({
                "timestamp": ts,
                "session_id": sid,
                "tool_name": tool,
                "arguments": json.loads(args) if args else {},
                "result": json.loads(res) if res else {},
            })

        return output

    # -----------------------------------
    # Clear logs (used in tests only)
    # -----------------------------------
    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tool_logs;")
            conn.commit()


# Singleton logger instance
audit_logger = ToolAuditLogger()
