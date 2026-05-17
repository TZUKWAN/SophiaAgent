"""Kanban board for research project management."""
import json
import logging
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KANBAN_SCHEMA = """
CREATE TABLE IF NOT EXISTS kanban_cards (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'backlog',
    assignee TEXT DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 3,
    tags TEXT DEFAULT '[]',
    due_date TEXT,
    goal_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_kanban_session ON kanban_cards(session_id);
"""

VALID_STATUSES = ["backlog", "todo", "in_progress", "done", "archived"]


class KanbanBoard:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(KANBAN_SCHEMA)

    def create_card(self, session_id: str, title: str, description: str = "",
                    status: str = "backlog", assignee: str = "", priority: int = 3,
                    tags: Optional[List[str]] = None, due_date: Optional[str] = None,
                    goal_id: Optional[str] = None) -> Dict:
        card_id = uuid.uuid4().hex[:8]
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO kanban_cards (id, session_id, title, description, status, "
                "assignee, priority, tags, due_date, goal_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (card_id, session_id, title, description, status,
                 assignee, priority, tags_json, due_date, goal_id),
            )
        return self.get_card(card_id)

    def get_card(self, card_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM kanban_cards WHERE id=?", (card_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["tags"] = json.loads(d["tags"])
        return d

    def move_card(self, card_id: str, status: str) -> Optional[Dict]:
        if status not in VALID_STATUSES:
            return None
        with self._connect() as conn:
            conn.execute(
                "UPDATE kanban_cards SET status=?, updated_at=datetime('now') WHERE id=?",
                (status, card_id),
            )
        return self.get_card(card_id)

    def update_card(self, card_id: str, **kwargs) -> Optional[Dict]:
        allowed = {"title", "description", "assignee", "priority", "tags", "due_date"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_card(card_id)
        if "tags" in updates:
            updates["tags"] = json.dumps(updates["tags"], ensure_ascii=False)
        set_parts = []
        values = []
        for k, v in updates.items():
            set_parts.append(f"{k}=?")
            values.append(v)
        set_parts.append("updated_at=datetime('now')")
        values.append(card_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE kanban_cards SET {', '.join(set_parts)} WHERE id=?", values
            )
        return self.get_card(card_id)

    def get_board(self, session_id: str) -> Dict[str, List[Dict]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM kanban_cards WHERE session_id=? ORDER BY priority, created_at",
                (session_id,),
            ).fetchall()
        board = {s: [] for s in VALID_STATUSES}
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            board[d["status"]].append(d)
        return board

    def search_cards(self, session_id: str, query: str) -> List[Dict]:
        q = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM kanban_cards WHERE session_id=? AND "
                "(title LIKE ? OR description LIKE ?) ORDER BY priority",
                (session_id, q, q),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_card(self, card_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM kanban_cards WHERE id=?", (card_id,))
            return cursor.rowcount > 0


def register_kanban_tools(registry, board: KanbanBoard):
    def _create(args):
        card = board.create_card(
            session_id=args.get("session_id", "default"),
            title=args["title"],
            description=args.get("description", ""),
            priority=args.get("priority", 3),
            tags=args.get("tags"),
        )
        return json.dumps(card, ensure_ascii=False, default=str)

    registry.register("kanban_create", "Create a kanban card",
        {"type": "object", "properties": {
            "session_id": {"type": "string"},
            "title": {"type": "string"}, "description": {"type": "string"},
            "priority": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}},
        }, "required": ["title"]}, _create)

    def _move(args):
        card = board.move_card(args["card_id"], args["status"])
        return json.dumps(card or {"error": "Invalid status or card not found"}, ensure_ascii=False, default=str)

    registry.register("kanban_move", "Move a kanban card to a new status",
        {"type": "object", "properties": {
            "card_id": {"type": "string"}, "status": {"type": "string"},
        }, "required": ["card_id", "status"]}, _move)

    def _board(args):
        b = board.get_board(args.get("session_id", "default"))
        summary = {k: len(v) for k, v in b.items()}
        return json.dumps(summary, ensure_ascii=False)

    registry.register("kanban_board", "View the kanban board",
        {"type": "object", "properties": {"session_id": {"type": "string"}}}, _board)
