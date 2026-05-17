"""Backward-compatible SubAgent facade backed by the swarm orchestrator."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager
from sophia.swarm.orchestrator import SwarmOrchestrator
from sophia.tools.registry import ToolRegistry


@dataclass
class SubAgentTask:
    id: str
    session_id: str
    goal_id: Optional[str]
    prompt: str
    tools: List[str]
    status: str
    result: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]


SUBAGENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS subagent_tasks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    goal_id TEXT,
    prompt TEXT NOT NULL,
    tools TEXT DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_subagent_session ON subagent_tasks(session_id);
"""


class SubAgentManager:
    """Legacy API that delegates execution to SwarmOrchestrator."""

    def __init__(self, run_fn, hooks: HookManager, db_path: str, orchestrator: Optional[SwarmOrchestrator] = None):
        self.hooks = hooks
        self.db_path = db_path
        self.orchestrator = orchestrator or SwarmOrchestrator(run_fn=run_fn, hooks=hooks)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SUBAGENTS_SCHEMA)

    def _row_to_task(self, row) -> SubAgentTask:
        return SubAgentTask(
            id=row["id"],
            session_id=row["session_id"],
            goal_id=row["goal_id"],
            prompt=row["prompt"],
            tools=json.loads(row["tools"]),
            status=row["status"],
            result=row["result"],
            error=row["error"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )

    def delegate(
        self,
        session_id: str,
        prompt: str,
        tools: Optional[List[str]] = None,
        goal_id: Optional[str] = None,
    ) -> SubAgentTask:
        task_id = uuid.uuid4().hex[:8]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO subagent_tasks (id, session_id, goal_id, prompt, tools) VALUES (?, ?, ?, ?, ?)",
                (task_id, session_id, goal_id, prompt, json.dumps(tools or [], ensure_ascii=False)),
            )
        self.hooks.emit(HookEvent.SUBAGENT_SPAWN, {"task_id": task_id, "prompt": prompt})
        return SubAgentTask(task_id, session_id, goal_id, prompt, tools or [], "pending", None, None, "", None)

    def execute(self, task: SubAgentTask) -> SubAgentTask:
        self._update_status(task.id, "running")
        result = self.orchestrator.delegate(task.session_id, task.prompt, task.tools, task.goal_id)
        task.status = result["status"]
        task.result = result.get("result")
        task.error = result.get("error")
        self._update_status(task.id, task.status, result=task.result, error=task.error, completed_at=True)
        if task.status == "completed":
            self.hooks.emit(HookEvent.SUBAGENT_COMPLETE, {"task_id": task.id, "result": task.result})
        else:
            self.hooks.emit(HookEvent.SUBAGENT_ERROR, {"task_id": task.id, "error": task.error})
        return task

    def execute_parallel(self, tasks: List[SubAgentTask], max_workers: int = 3) -> List[SubAgentTask]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.execute, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def delegate_batch(self, session_id: str, tasks: List[Dict]) -> List[SubAgentTask]:
        created = [
            self.delegate(session_id, task["prompt"], tools=task.get("tools"), goal_id=task.get("goal_id"))
            for task in tasks
        ]
        return self.execute_parallel(created)

    def _update_status(
        self,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
        completed_at: bool = False,
    ):
        with self._connect() as conn:
            if completed_at:
                conn.execute(
                    "UPDATE subagent_tasks SET status=?, result=?, error=?, completed_at=datetime('now') WHERE id=?",
                    (status, result, error, task_id),
                )
            else:
                conn.execute("UPDATE subagent_tasks SET status=? WHERE id=?", (status, task_id))

    def get_task(self, task_id: str) -> Optional[SubAgentTask]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM subagent_tasks WHERE id=?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(self, session_id: str) -> List[SubAgentTask]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM subagent_tasks WHERE session_id=? ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]


def register_subagent_tools(registry: ToolRegistry, subagent_mgr: SubAgentManager):
    """Keep legacy tool names working while the swarm tools become primary."""

    def _delegate(args):
        task = subagent_mgr.delegate(
            session_id=args.get("session_id", "default"),
            prompt=args["prompt"],
            tools=args.get("tools"),
            goal_id=args.get("goal_id"),
        )
        result = subagent_mgr.execute(task)
        return json.dumps(
            {
                "id": result.id,
                "status": result.status,
                "result": (result.result or "")[:500],
                "error": result.error,
            },
            ensure_ascii=False,
        )

    registry.register(
        "subagent_delegate",
        "Delegate a task through the swarm-backed sub-agent compatibility layer",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "prompt": {"type": "string"},
                "tools": {"type": "array", "items": {"type": "string"}},
                "goal_id": {"type": "string"},
            },
            "required": ["session_id", "prompt"],
        },
        _delegate,
    )

    def _delegate_batch(args):
        results = subagent_mgr.delegate_batch(args.get("session_id", "default"), args.get("tasks", []))
        return json.dumps(
            [
                {
                    "id": task.id,
                    "status": task.status,
                    "result": (task.result or "")[:200],
                    "error": task.error,
                }
                for task in results
            ],
            ensure_ascii=False,
        )

    registry.register(
        "subagent_delegate_batch",
        "Delegate multiple tasks through the swarm-backed compatibility layer",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "tools": {"type": "array", "items": {"type": "string"}},
                            "goal_id": {"type": "string"},
                        },
                        "required": ["prompt"],
                    },
                },
            },
            "required": ["session_id", "tasks"],
        },
        _delegate_batch,
    )

    def _list(args):
        tasks = subagent_mgr.list_tasks(args.get("session_id", "default"))
        return json.dumps(
            [
                {
                    "id": task.id,
                    "prompt": task.prompt[:100],
                    "status": task.status,
                    "tools": task.tools,
                    "created_at": task.created_at,
                }
                for task in tasks
            ],
            ensure_ascii=False,
        )

    registry.register(
        "subagent_list",
        "List legacy sub-agent task records",
        {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
        _list,
    )
