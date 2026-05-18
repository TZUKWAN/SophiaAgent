"""Session persistence for SophiaAgent.

SQLite WAL-mode storage with 3 tables:
- sessions: session metadata
- messages: conversation history
- checkpoints: named snapshots for rollback
"""

import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    workspace TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT DEFAULT '',
    tool_calls TEXT,
    tool_call_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS workspaces (
    path TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id);
"""


class SessionManager:
    """Manages conversation sessions with SQLite persistence."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)
        logger.info("Session database initialized: %s", self.db_path)

    def _migrate(self, conn):
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        if "workspace" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN workspace TEXT NOT NULL DEFAULT ''")
            logger.info("Migrated sessions table: added workspace column")

    # ── Sessions ──────────────────────────────────────────────

    def create_session(
        self, title: str = "", model: str = "", session_id: Optional[str] = None,
        workspace: str = "",
    ) -> str:
        sid = session_id or uuid.uuid4().hex[:8]
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, title, model, workspace) VALUES (?, ?, ?, ?)",
                (sid, title, model, workspace),
            )
        return sid

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.id, s.title, s.model, s.workspace, s.created_at, s.updated_at, "
                "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS msg_count "
                "FROM sessions s ORDER BY s.updated_at DESC"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "model": r["model"],
                "workspace": r["workspace"] if "workspace" in r.keys() else "",
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "message_count": r["msg_count"],
            }
            for r in rows
        ]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row:
                return None
            messages = self.get_messages(session_id)
            return {
                "id": row["id"],
                "title": row["title"],
                "model": row["model"],
                "workspace": row["workspace"] if "workspace" in row.keys() else "",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "messages": messages,
            }

    def update_session_title(self, session_id: str, title: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = datetime('now') WHERE id = ?",
                (title, session_id),
            )

    def delete_session(self, session_id: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # ── Messages ──────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str = "",
        tool_calls: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, tool_calls, tool_call_id),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
                (session_id,),
            )

    def add_messages_batch(self, session_id: str, messages: List[Dict[str, Any]]):
        with self._connect() as conn:
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                tc = msg.get("tool_calls")
                tc_json = json.dumps(tc, ensure_ascii=False) if tc else None
                tc_id = msg.get("tool_call_id")
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, tc_json, tc_id),
                )
            conn.execute(
                "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
                (session_id,),
            )

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_calls, tool_call_id FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        result = []
        for r in rows:
            msg: Dict[str, Any] = {"role": r["role"]}
            if r["content"]:
                msg["content"] = r["content"]
            if r["tool_calls"]:
                msg["tool_calls"] = json.loads(r["tool_calls"])
            if r["tool_call_id"]:
                msg["tool_call_id"] = r["tool_call_id"]
            result.append(msg)
        return result

    # ── Checkpoints ───────────────────────────────────────────

    def save_checkpoint(self, session_id: str, label: str) -> int:
        messages = self.get_messages(session_id)
        snapshot = json.dumps(messages, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO checkpoints (session_id, label, snapshot) VALUES (?, ?, ?)",
                (session_id, label, snapshot),
            )
            return cursor.lastrowid

    def list_checkpoints(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, label, created_at FROM checkpoints "
                "WHERE session_id = ? ORDER BY id DESC",
                (session_id,),
            ).fetchall()
        return [
            {"id": r["id"], "label": r["label"], "created_at": r["created_at"]}
            for r in rows
        ]

    def restore_checkpoint(self, session_id: str, checkpoint_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT snapshot FROM checkpoints WHERE id = ? AND session_id = ?",
                (checkpoint_id, session_id),
            ).fetchone()
            if not row:
                return False
            messages = json.loads(row["snapshot"])
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                tc = msg.get("tool_calls")
                tc_json = json.dumps(tc, ensure_ascii=False) if tc else None
                tc_id = msg.get("tool_call_id")
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, tc_json, tc_id),
                )
            conn.execute(
                "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?",
                (session_id,),
            )
        return True

    # ── Workspaces ────────────────────────────────────────────

    def register_workspace(self, path: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO workspaces (path) VALUES (?)", (path,)
            )

    def list_workspaces(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT workspace FROM sessions WHERE workspace != ''"
            ).fetchall()
        return [r["workspace"] for r in rows]

    def list_registered_workspaces(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT path FROM workspaces ORDER BY created_at").fetchall()
        return [r["path"] for r in rows]
