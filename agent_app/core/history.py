"""
Chat History Manager
--------------------

This module stores and retrieves chat histories
for all MCP agent sessions.

Why separate from AgentState?

- AgentState is in-memory (per session).
- History is persistent (across restarts).
- Follows SOLID: History storage is separate.

Each message stored contains:
 - session_id
 - role ("human" | "assistant" | "tool")
 - content
 - timestamp
 - tool_name (optional)
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional

from agent_app.core.state import AgentMessage

DB_PATH = "agent_app/history.sqlite3"


class HistoryStore:
    """
    SQLite-backed history store for chat messages.
    """

    def __init__(self):
        os.makedirs("agent_app", exist_ok=True)
        self._init_db()

    # ----------------------------------------------------
    # Initialize DB
    # ----------------------------------------------------

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TEXT,
                    tool_name TEXT
                )
                """
            )
            conn.commit()

    # ----------------------------------------------------
    # Store Message
    # ----------------------------------------------------

    def save_message(self, session_id: str, message: AgentMessage):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO chat_history (session_id, role, content, timestamp, tool_name)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    message.role,
                    message.content,
                    message.timestamp,
                    message.tool_name,
                ),
            )
            conn.commit()

    # ----------------------------------------------------
    # Retrieve history for one session
    # ----------------------------------------------------

    def get_history(self, session_id: str) -> List[Dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM chat_history
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [dict(row) for row in rows]

    # ----------------------------------------------------
    # Retrieve all sessions
    # ----------------------------------------------------

    def list_sessions(self) -> List[str]:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT session_id FROM chat_history
                ORDER BY session_id ASC
                """
            ).fetchall()

        return [row[0] for row in rows]

    # ----------------------------------------------------
    # Delete session history
    # ----------------------------------------------------

    def delete_session(self, session_id: str):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?", (session_id,)
            )
            conn.commit()

    # ----------------------------------------------------
    # Wipe all history
    # ----------------------------------------------------

    def clear_all(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM chat_history")
            conn.commit()


# --------------------------------------------------------
# Singleton Export
# --------------------------------------------------------

history_store = HistoryStore()
