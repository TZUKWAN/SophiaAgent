"""Memory system for SophiaAgent.

SQLite-backed memory for cross-session research continuity.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'note',
    key TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
"""


@dataclass
class MemoryEntry:
    id: int
    session_id: str
    category: str
    key: str
    content: str
    tags: List[str]
    access_count: int
    created_at: str
    updated_at: str


class MemoryManager:
    def __init__(self, db_path: str, hooks: HookManager = None):
        self.db_path = db_path
        self.hooks = hooks
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(MEMORY_SCHEMA)

    def _row_to_entry(self, row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            session_id=row["session_id"],
            category=row["category"],
            key=row["key"],
            content=row["content"],
            tags=json.loads(row["tags"]),
            access_count=row["access_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def store(self, session_id: str, key: str, content: str,
              category: str = "note", tags: Optional[List[str]] = None) -> int:
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO memories (session_id, category, key, content, tags) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, category, key, content, tags_json),
            )
            entry_id = cursor.lastrowid
        if self.hooks:
            self.hooks.emit(HookEvent.MEMORY_STORE, {
                "entry_id": entry_id, "session_id": session_id,
                "key": key, "category": category,
            })
        return entry_id

    def recall(self, session_id: str, query: str,
               category: Optional[str] = None, limit: int = 10) -> List[MemoryEntry]:
        entries = self._search(session_id, query, category, limit)
        for entry in entries:
            self._increment_access(entry.id)
        if self.hooks and entries:
            self.hooks.emit(HookEvent.MEMORY_RECALL, {
                "session_id": session_id, "query": query,
                "found": len(entries),
            })
        return entries

    def get(self, entry_id: int) -> Optional[MemoryEntry]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (entry_id,)).fetchone()
        return self._row_to_entry(row) if row else None

    def update(self, entry_id: int, **kwargs) -> Optional[MemoryEntry]:
        allowed = {"key", "content", "category", "tags"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get(entry_id)
        if "tags" in updates:
            updates["tags"] = json.dumps(updates["tags"], ensure_ascii=False)
        updates["updated_at"] = "datetime('now')"
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [entry_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE memories SET {set_clause} WHERE id=?", values)
        return self.get(entry_id)

    def delete(self, entry_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id=?", (entry_id,))
            return cursor.rowcount > 0

    def search(self, session_id: str, query: str, limit: int = 10) -> List[MemoryEntry]:
        return self._search(session_id, query, None, limit)

    def get_by_category(self, session_id: str, category: str) -> List[MemoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE session_id=? AND category=? "
                "ORDER BY updated_at DESC",
                (session_id, category),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_all(self, session_id: str, limit: int = 100) -> List[MemoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE session_id=? ORDER BY updated_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def build_context(self, session_id: str, query: str, max_entries: int = 5) -> str:
        entries = self._search(session_id, query, None, max_entries)
        if not entries:
            return ""
        lines = ["[Memory Context]"]
        for e in entries:
            lines.append(f"- ({e.category}) {e.key}: {e.content[:200]}")
        return "\n".join(lines)

    def _search(self, session_id: str, query: str,
                category: Optional[str], limit: int) -> List[MemoryEntry]:
        keywords = query.lower().split()
        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE session_id=? AND category=? "
                    "ORDER BY access_count DESC, updated_at DESC",
                    (session_id, category),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE session_id=? "
                    "ORDER BY access_count DESC, updated_at DESC",
                    (session_id,),
                ).fetchall()
        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            if self._keyword_match(entry, keywords):
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def _keyword_match(self, entry: MemoryEntry, keywords: List[str]) -> bool:
        text = f"{entry.key} {entry.content} {' '.join(entry.tags)}".lower()
        return any(kw in text for kw in keywords)

    def _increment_access(self, entry_id: int):
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET access_count=access_count+1, "
                "updated_at=datetime('now') WHERE id=?",
                (entry_id,),
            )


def register_memory_tools(registry, memory_mgr: MemoryManager):
    def _store(args):
        entry_id = memory_mgr.store(
            session_id=args.get("session_id", "default"),
            key=args["key"],
            content=args["content"],
            category=args.get("category", "note"),
            tags=args.get("tags"),
        )
        return json.dumps({"id": entry_id, "action": "stored"}, ensure_ascii=False)

    registry.register(
        "memory_store",
        "Store a memory entry for future reference",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "key": {"type": "string", "description": "Short identifier for the memory"},
                "content": {"type": "string", "description": "The memory content"},
                "category": {
                    "type": "string",
                    "enum": ["preference", "domain_knowledge", "research_history", "fact", "note"],
                    "description": "Memory category",
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["key", "content"],
        },
        _store,
    )

    def _recall(args):
        entries = memory_mgr.recall(
            session_id=args.get("session_id", "default"),
            query=args["query"],
            category=args.get("category"),
            limit=args.get("limit", 10),
        )
        return json.dumps([{
            "id": e.id, "key": e.key, "content": e.content[:200],
            "category": e.category, "tags": e.tags,
        } for e in entries], ensure_ascii=False)

    registry.register(
        "memory_recall",
        "Recall memories matching a query",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "query": {"type": "string"},
                "category": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
        _recall,
    )

    def _search(args):
        entries = memory_mgr.search(
            session_id=args.get("session_id", "default"),
            query=args["query"],
            limit=args.get("limit", 10),
        )
        return json.dumps([{
            "id": e.id, "key": e.key, "content": e.content[:200],
            "category": e.category,
        } for e in entries], ensure_ascii=False)

    registry.register(
        "memory_search",
        "Search memories by keywords",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
        _search,
    )

    def _delete(args):
        ok = memory_mgr.delete(args["entry_id"])
        return json.dumps({"deleted": ok}, ensure_ascii=False)

    registry.register(
        "memory_delete",
        "Delete a memory entry",
        {
            "type": "object",
            "properties": {
                "entry_id": {"type": "integer"},
            },
            "required": ["entry_id"],
        },
        _delete,
    )
