"""Goal tree system for SophiaAgent.

SQLite-persisted hierarchical goal decomposition with parent-child
relationships, status lifecycle tracking, progress auto-calculation,
and hook integration.
"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)

GOAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES goals(id) ON DELETE SET NULL,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 3,
    progress REAL NOT NULL DEFAULT 0.0,
    deadline TEXT,
    result TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_goals_session ON goals(session_id);
CREATE INDEX IF NOT EXISTS idx_goals_parent ON goals(parent_id);
"""

VALID_STATUSES = {"pending", "active", "completed", "failed", "cancelled"}


@dataclass
class Goal:
    """A single goal node in the goal tree."""

    id: str
    parent_id: Optional[str]
    session_id: str
    title: str
    description: str
    status: str
    priority: int
    progress: float
    deadline: Optional[str]
    result: Optional[str]
    created_at: str
    updated_at: str


def _row_to_goal(row: sqlite3.Row) -> Goal:
    """Convert a sqlite3.Row to a Goal dataclass instance."""
    return Goal(
        id=row["id"],
        parent_id=row["parent_id"],
        session_id=row["session_id"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        priority=row["priority"],
        progress=row["progress"],
        deadline=row["deadline"],
        result=row["result"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class GoalManager:
    """Manages goal trees with SQLite persistence and hook integration."""

    def __init__(self, db_path: str, hooks: HookManager):
        self.db_path = db_path
        self.hooks = hooks
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
            conn.executescript(GOAL_SCHEMA)
        logger.info("Goal database initialized: %s", self.db_path)

    # ── CRUD ──────────────────────────────────────────────────

    def create(
        self,
        session_id: str,
        title: str,
        description: str = "",
        parent_id: Optional[str] = None,
        priority: int = 3,
        deadline: Optional[str] = None,
    ) -> Goal:
        """Create a new goal and return it.

        Args:
            session_id: The session this goal belongs to.
            title: Short title for the goal.
            description: Detailed description.
            parent_id: Optional parent goal ID for tree structure.
            priority: 1 (highest) to 5 (lowest).
            deadline: Optional ISO-format deadline string.

        Returns:
            The newly created Goal.
        """
        goal_id = uuid.uuid4().hex[:12]
        now = "datetime('now')"

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO goals (id, parent_id, session_id, title, description, "
                "status, priority, progress, deadline, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?, 0.0, ?, datetime('now'), datetime('now'))",
                (goal_id, parent_id, session_id, title, description, priority, deadline),
            )
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()

        goal = _row_to_goal(row)
        self.hooks.emit(HookEvent.GOAL_CREATED, {"goal": asdict(goal)})
        logger.info("Goal created: %s (%s)", goal_id, title)
        return goal

    def update(self, goal_id: str, **kwargs) -> Goal:
        """Update mutable fields on a goal.

        Allowed keyword args: status, priority, progress, description,
        title, deadline, result, parent_id.

        Emits HookEvent.GOAL_UPDATED after the update.

        Returns:
            The updated Goal.
        """
        allowed = {
            "status", "priority", "progress", "description",
            "title", "deadline", "result", "parent_id",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get(goal_id)

        if "status" in updates and updates["status"] not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{updates['status']}'. "
                f"Must be one of: {', '.join(sorted(VALID_STATUSES))}"
            )

        set_clauses = []
        values = []
        for col, val in updates.items():
            set_clauses.append(f"{col} = ?")
            values.append(val)

        set_clauses.append("updated_at = datetime('now')")
        values.append(goal_id)

        with self._connect() as conn:
            conn.execute(
                f"UPDATE goals SET {', '.join(set_clauses)} WHERE id = ?",
                values,
            )
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()

        goal = _row_to_goal(row)
        self.hooks.emit(HookEvent.GOAL_UPDATED, {"goal": asdict(goal)})
        logger.info("Goal updated: %s (fields: %s)", goal_id, list(updates.keys()))
        return goal

    def get(self, goal_id: str) -> Optional[Goal]:
        """Retrieve a single goal by ID. Returns None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_goal(row)

    def get_tree(self, session_id: str) -> List[Goal]:
        """Return all goals belonging to a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM goals WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [_row_to_goal(r) for r in rows]

    def get_children(self, parent_id: str) -> List[Goal]:
        """Return direct children of a goal."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM goals WHERE parent_id = ? ORDER BY priority ASC, created_at ASC",
                (parent_id,),
            ).fetchall()
        return [_row_to_goal(r) for r in rows]

    def get_active(self, session_id: str) -> List[Goal]:
        """Return all active goals for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM goals WHERE session_id = ? AND status = 'active' "
                "ORDER BY priority ASC, created_at ASC",
                (session_id,),
            ).fetchall()
        return [_row_to_goal(r) for r in rows]

    # ── Lifecycle ─────────────────────────────────────────────

    def complete(self, goal_id: str, result: str = "") -> Goal:
        """Mark a goal as completed with an optional result.

        Sets progress to 1.0, status to 'completed'.
        Emits HookEvent.GOAL_COMPLETED.

        Returns:
            The updated Goal.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE goals SET status = 'completed', progress = 1.0, "
                "result = ?, updated_at = datetime('now') WHERE id = ?",
                (result, goal_id),
            )
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()

        goal = _row_to_goal(row)
        self.hooks.emit(HookEvent.GOAL_COMPLETED, {"goal": asdict(goal)})
        logger.info("Goal completed: %s", goal_id)

        # Recompute parent progress if this goal has a parent
        if goal.parent_id:
            self._propagate_progress(goal.parent_id)

        return goal

    def fail(self, goal_id: str, reason: str = "") -> Goal:
        """Mark a goal as failed with an optional reason.

        Sets status to 'failed', result to the reason.
        Emits HookEvent.GOAL_FAILED.

        Returns:
            The updated Goal.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE goals SET status = 'failed', result = ?, "
                "updated_at = datetime('now') WHERE id = ?",
                (reason, goal_id),
            )
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()

        goal = _row_to_goal(row)
        self.hooks.emit(HookEvent.GOAL_FAILED, {"goal": asdict(goal)})
        logger.info("Goal failed: %s", goal_id)

        # Recompute parent progress if this goal has a parent
        if goal.parent_id:
            self._propagate_progress(goal.parent_id)

        return goal

    def cancel(self, goal_id: str) -> Goal:
        """Cancel a goal. Sets status to 'cancelled'.

        Returns:
            The updated Goal.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE goals SET status = 'cancelled', "
                "updated_at = datetime('now') WHERE id = ?",
                (goal_id,),
            )
            row = conn.execute(
                "SELECT * FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()

        goal = _row_to_goal(row)
        logger.info("Goal cancelled: %s", goal_id)

        # Recompute parent progress if this goal has a parent
        if goal.parent_id:
            self._propagate_progress(goal.parent_id)

        return goal

    # ── Progress ──────────────────────────────────────────────

    def compute_progress(self, goal_id: str) -> float:
        """Compute and persist progress from children.

        Progress = (number of completed children) / (total children).
        If a goal has no children, returns its current progress.
        Updates the goal's progress column in the database.

        Returns:
            The computed progress value (0.0 to 1.0).
        """
        children = self.get_children(goal_id)
        if not children:
            goal = self.get(goal_id)
            return goal.progress if goal else 0.0

        total = len(children)
        completed = sum(1 for c in children if c.status == "completed")
        progress = completed / total

        with self._connect() as conn:
            conn.execute(
                "UPDATE goals SET progress = ?, updated_at = datetime('now') "
                "WHERE id = ?",
                (progress, goal_id),
            )

        return progress

    def _propagate_progress(self, parent_id: str):
        """Recompute progress for a parent and propagate upward."""
        self.compute_progress(parent_id)
        parent = self.get(parent_id)
        if parent and parent.parent_id:
            self._propagate_progress(parent.parent_id)

    # ── Delete ────────────────────────────────────────────────

    def delete(self, goal_id: str):
        """Delete a goal and all its descendants (cascade).

        Collects the full subtree via BFS and deletes leaf-first
        to respect foreign key constraints.
        """
        # Collect all descendants via BFS
        to_delete: List[str] = [goal_id]
        queue = [goal_id]
        with self._connect() as conn:
            while queue:
                current = queue.pop(0)
                child_rows = conn.execute(
                    "SELECT id FROM goals WHERE parent_id = ?", (current,)
                ).fetchall()
                for row in child_rows:
                    cid = row["id"]
                    to_delete.append(cid)
                    queue.append(cid)

            # Delete in reverse order (deepest children first) to
            # respect foreign key constraints
            for gid in reversed(to_delete):
                conn.execute("DELETE FROM goals WHERE id = ?", (gid,))

        parent_id = None
        # We need to find the parent of the top-level deleted goal
        # to recompute progress. Since we deleted it, we look at siblings.
        # Actually, let's get the parent_id before deletion by checking
        # the first in to_delete (the root of the subtree).
        # We already deleted, so we need to find out who the parent was.
        # Let's refactor: get the parent_id before deletion.

        logger.info("Deleted goal subtree: %s (%d goals)", goal_id, len(to_delete))

    def delete_with_propagation(self, goal_id: str):
        """Delete a goal and all its descendants, then propagate progress upward.

        This is the recommended delete method as it also recomputes
        ancestor progress after deletion.
        """
        # Get the parent before we delete
        goal = self.get(goal_id)
        parent_id = goal.parent_id if goal else None

        self.delete(goal_id)

        # Recompute ancestor progress
        if parent_id:
            parent = self.get(parent_id)
            if parent:
                self._propagate_progress(parent_id)


# ── Tool Registration ─────────────────────────────────────────


def register_goal_tools(registry, goal_mgr: GoalManager):
    """Register goal management tools with the tool registry.

    Tools registered:
        - goal_create: Create a new goal.
        - goal_update: Update goal fields.
        - goal_list: List all goals in a session.
        - goal_complete: Mark a goal as completed.
        - goal_fail: Mark a goal as failed.
    """

    def _goal_create(args: Dict[str, Any]) -> str:
        session_id = args.get("session_id", "")
        title = args.get("title", "")
        description = args.get("description", "")
        parent_id = args.get("parent_id")
        priority = args.get("priority", 3)
        deadline = args.get("deadline")

        if not session_id:
            return json.dumps({"error": "session_id is required"}, ensure_ascii=False)
        if not title:
            return json.dumps({"error": "title is required"}, ensure_ascii=False)

        goal = goal_mgr.create(
            session_id=session_id,
            title=title,
            description=description,
            parent_id=parent_id,
            priority=priority,
            deadline=deadline,
        )
        return json.dumps(asdict(goal), ensure_ascii=False)

    def _goal_update(args: Dict[str, Any]) -> str:
        goal_id = args.get("goal_id", "")
        if not goal_id:
            return json.dumps({"error": "goal_id is required"}, ensure_ascii=False)

        updates = {}
        for key in ("status", "priority", "progress", "description", "title", "deadline", "result"):
            if key in args:
                updates[key] = args[key]

        if not updates:
            return json.dumps({"error": "No fields to update"}, ensure_ascii=False)

        try:
            goal = goal_mgr.update(goal_id, **updates)
            return json.dumps(asdict(goal), ensure_ascii=False)
        except ValueError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _goal_list(args: Dict[str, Any]) -> str:
        session_id = args.get("session_id", "")
        if not session_id:
            return json.dumps({"error": "session_id is required"}, ensure_ascii=False)

        goals = goal_mgr.get_tree(session_id)
        return json.dumps(
            {"goals": [asdict(g) for g in goals], "total": len(goals)},
            ensure_ascii=False,
        )

    def _goal_complete(args: Dict[str, Any]) -> str:
        goal_id = args.get("goal_id", "")
        if not goal_id:
            return json.dumps({"error": "goal_id is required"}, ensure_ascii=False)

        result = args.get("result", "")
        goal = goal_mgr.complete(goal_id, result=result)
        return json.dumps(asdict(goal), ensure_ascii=False)

    def _goal_fail(args: Dict[str, Any]) -> str:
        goal_id = args.get("goal_id", "")
        if not goal_id:
            return json.dumps({"error": "goal_id is required"}, ensure_ascii=False)

        reason = args.get("reason", "")
        goal = goal_mgr.fail(goal_id, reason=reason)
        return json.dumps(asdict(goal), ensure_ascii=False)

    registry.register(
        name="goal_create",
        description="Create a new goal in the goal tree.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID"},
                "title": {"type": "string", "description": "Goal title"},
                "description": {"type": "string", "description": "Goal description"},
                "parent_id": {"type": "string", "description": "Parent goal ID (optional)"},
                "priority": {"type": "integer", "description": "Priority 1-5 (1=highest)"},
                "deadline": {"type": "string", "description": "ISO-format deadline"},
            },
            "required": ["session_id", "title"],
        },
        handler=_goal_create,
    )

    registry.register(
        name="goal_update",
        description="Update fields on an existing goal.",
        parameters={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "Goal ID to update"},
                "status": {"type": "string", "description": "New status"},
                "priority": {"type": "integer", "description": "New priority 1-5"},
                "progress": {"type": "number", "description": "New progress 0.0-1.0"},
                "description": {"type": "string", "description": "New description"},
            },
            "required": ["goal_id"],
        },
        handler=_goal_update,
    )

    registry.register(
        name="goal_list",
        description="List all goals for a session.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID"},
            },
            "required": ["session_id"],
        },
        handler=_goal_list,
    )

    registry.register(
        name="goal_complete",
        description="Mark a goal as completed.",
        parameters={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "Goal ID to complete"},
                "result": {"type": "string", "description": "Completion result or summary"},
            },
            "required": ["goal_id"],
        },
        handler=_goal_complete,
    )

    registry.register(
        name="goal_fail",
        description="Mark a goal as failed.",
        parameters={
            "type": "object",
            "properties": {
                "goal_id": {"type": "string", "description": "Goal ID to fail"},
                "reason": {"type": "string", "description": "Failure reason"},
            },
            "required": ["goal_id"],
        },
        handler=_goal_fail,
    )
