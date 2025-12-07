"""
Audit Logger for MCP Tool Calls
-------------------------------

This module writes and reads audit logs of tool executions.
The audit log is stored in SQLite so it persists across sessions.

Each log entry includes:
 - timestamp
 - session_id
 - tool_name
 - arguments (JSON)
 - result (JSON)
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import threading
import os


DB_PATH = "agent_app/audit_logs.sqlite3"


class AuditLogger:
    """
    Simple thread-safe SQLite audit logger for tool calls.
    """

    _lock = threading.Lock()

    def __init__(self):
        self._ensure_db()

    # --------------------------------------------------------
    # Initialize DB
    # --------------------------------------------------------

    def _ensure_db(self):
        """
        Create the SQLite database + table if missing.
        """

        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    session_id TEXT,
                    tool_name TEXT,
                    arguments TEXT,
                    result TEXT
                )
                """
            )
            conn.commit()

    # --------------------------------------------------------
    # Write a tool log record
    # --------------------------------------------------------

    def write_record(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any
    ):
        """
        Store a single tool execution record.
        """

        ts = datetime.utcnow().isoformat()

        with self._lock:  # Ensure thread safety
            with sqlite3.connect(DB_PATH) as conn:
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
                        json.dumps(result)
                    )
                )
                conn.commit()

    # --------------------------------------------------------
    # Retrieve logs
    # --------------------------------------------------------

    def list_logs(
        self,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve up to N logs, optionally filtered by session.
        """

        with self._lock:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row

                if session_id:
                    rows = conn.execute(
                        """
                        SELECT * FROM tool_logs
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (session_id, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM tool_logs
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (limit,)
                    ).fetchall()

        return [dict(r) for r in rows]


# --------------------------------------------------------
# Singleton Instance
# --------------------------------------------------------

audit_logger = AuditLogger()
