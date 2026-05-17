"""Git-based checkpoint snapshots for SophiaAgent."""

import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    label TEXT NOT NULL,
    git_commit TEXT,
    file_manifest TEXT,
    message_count INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_snapshots_session ON snapshots(session_id);
"""


class SnapshotManager:
    def __init__(self, db_path: str, workspace: str):
        self.db_path = db_path
        self.workspace = workspace
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SNAPSHOTS_SCHEMA)

    def _compute_file_manifest(self) -> Dict[str, str]:
        """Compute hash of all files in workspace."""
        manifest = {}
        ws = Path(self.workspace)
        if not ws.exists():
            return manifest
        for f in ws.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                try:
                    rel = str(f.relative_to(ws))
                    h = hashlib.md5(f.read_bytes()).hexdigest()
                    manifest[rel] = h
                except (OSError, ValueError):
                    continue
        return manifest

    def _try_git_commit(self, label: str) -> Optional[str]:
        """Try to create a git commit. Returns commit hash or None."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "add", "-A"],
                cwd=self.workspace, capture_output=True, timeout=10,
            )
            result = subprocess.run(
                ["git", "commit", "-m", f"snapshot: {label}", "--allow-empty"],
                cwd=self.workspace, capture_output=True, timeout=10,
            )
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.workspace, capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.decode().strip()[:12]
        except Exception:
            pass
        return None

    def create_snapshot(self, session_id: str, label: str, message_count: int = 0) -> int:
        """Create a new snapshot."""
        git_commit = self._try_git_commit(label)
        manifest = self._compute_file_manifest()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO snapshots (session_id, label, git_commit, file_manifest, message_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, label, git_commit, json.dumps(manifest, ensure_ascii=False), message_count),
            )
            return cursor.lastrowid

    def restore_snapshot(self, snapshot_id: int) -> bool:
        """Try to restore from a git-based snapshot."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if not row:
            return False
        if row["git_commit"]:
            try:
                import subprocess
                subprocess.run(
                    ["git", "checkout", row["git_commit"]],
                    cwd=self.workspace, capture_output=True, timeout=10,
                )
                return True
            except Exception:
                return False
        return False

    def list_snapshots(self, session_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, session_id, label, git_commit, message_count, created_at "
                "FROM snapshots WHERE session_id=? ORDER BY id DESC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_snapshot(self, snapshot_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        return dict(row) if row else None

    def delete_snapshot(self, snapshot_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM snapshots WHERE id=?", (snapshot_id,))
            return cursor.rowcount > 0
