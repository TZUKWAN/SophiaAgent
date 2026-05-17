"""Cron-like scheduling for SophiaAgent."""
import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)

SCHEDULED_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    cron_expr TEXT NOT NULL,
    action_prompt TEXT NOT NULL,
    last_run TEXT,
    next_run TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_scheduled_session ON scheduled_tasks(session_id);
"""


class CronScheduler:
    def __init__(self, run_fn: Callable, hooks: HookManager, db_path: str):
        self._run_fn = run_fn
        self.hooks = hooks
        self.db_path = db_path
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SCHEDULED_TASKS_SCHEMA)

    def schedule(self, session_id: str, name: str, cron_expr: str,
                 action_prompt: str) -> str:
        task_id = uuid.uuid4().hex[:8]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO scheduled_tasks (id, session_id, name, cron_expr, action_prompt) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, session_id, name, cron_expr, action_prompt),
            )
        return task_id

    def unschedule(self, task_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM scheduled_tasks WHERE id=?", (task_id,))
            return cursor.rowcount > 0

    def list_scheduled(self, session_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scheduled_tasks WHERE session_id=? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _should_fire(self, cron_expr: str) -> bool:
        now = datetime.now()
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            return False
        minute_part = parts[0]
        if minute_part.startswith("*/"):
            interval = int(minute_part[2:])
            return now.minute % interval == 0 and now.second < 10
        try:
            return now.minute == int(minute_part) and now.second < 10
        except ValueError:
            return False

    def start_scheduler(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()

    def stop_scheduler(self):
        self._running = False
        self._stop_event.set()

    def _scheduler_loop(self):
        while not self._stop_event.is_set():
            try:
                with self._connect() as conn:
                    rows = conn.execute(
                        "SELECT * FROM scheduled_tasks WHERE status='active'"
                    ).fetchall()
                for row in rows:
                    if self._should_fire(row["cron_expr"]):
                        try:
                            result = self._run_fn(row["action_prompt"])
                            with self._connect() as conn:
                                conn.execute(
                                    "UPDATE scheduled_tasks SET last_run=datetime('now') WHERE id=?",
                                    (row["id"],),
                                )
                            if self.hooks:
                                self.hooks.emit(HookEvent.SCHEDULER_FIRE, {
                                    "task_id": row["id"],
                                    "result": (result or "")[:200],
                                })
                        except Exception as e:
                            logger.warning("Scheduled task %s failed: %s", row["id"], e)
            except Exception as e:
                logger.warning("Scheduler loop error: %s", e)
            self._stop_event.wait(timeout=30)
