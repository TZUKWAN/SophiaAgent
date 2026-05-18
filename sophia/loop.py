"""Background loop execution system for SophiaAgent.

Manages recurring tasks that run in daemon threads on interval or condition triggers.
Each loop emits hook events on tick, completion, and error.

Usage:
    hooks = HookManager()
    loop_mgr = LoopManager(run_fn=agent.run, hooks=hooks, db_path="loops.db")
    spec = loop_mgr.create(session_id="s1", name="poll", trigger_type="interval",
                           trigger_config={"seconds": 60}, action_prompt="Check status")
    loop_mgr.start(spec.id)
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)

LOOP_SCHEMA = """
CREATE TABLE IF NOT EXISTS loops (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    trigger_config TEXT NOT NULL DEFAULT '{}',
    action_prompt TEXT NOT NULL,
    max_iterations INTEGER NOT NULL DEFAULT 0,
    current_iteration INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    last_result TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_loops_session ON loops(session_id);
"""


@dataclass
class LoopSpec:
    """Specification and state for a single loop."""
    id: str
    session_id: str
    name: str
    trigger_type: str          # interval | condition
    trigger_config: Dict       # interval: {"seconds": N} | condition: {"check_prompt": "..."}
    action_prompt: str
    max_iterations: int        # 0 = unlimited
    current_iteration: int
    status: str                # running | paused | completed | failed
    last_result: Optional[str]
    created_at: str


class LoopManager:
    """Creates, starts, stops, and monitors background loop tasks."""

    def __init__(self, run_fn: Callable, hooks: HookManager, db_path: str):
        """
        Args:
            run_fn: Callable[[str], str] - takes a prompt, returns a response string.
            hooks: HookManager instance for emitting loop events.
            db_path: Path to the SQLite database file.
        """
        self._run_fn = run_fn
        self.hooks = hooks
        self.db_path = db_path
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._pause_events: Dict[str, threading.Event] = {}
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Database helpers ──────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(LOOP_SCHEMA)
        logger.info("Loop database initialized: %s", self.db_path)

    def _row_to_spec(self, row: sqlite3.Row) -> LoopSpec:
        return LoopSpec(
            id=row["id"],
            session_id=row["session_id"],
            name=row["name"],
            trigger_type=row["trigger_type"],
            trigger_config=json.loads(row["trigger_config"]),
            action_prompt=row["action_prompt"],
            max_iterations=row["max_iterations"],
            current_iteration=row["current_iteration"],
            status=row["status"],
            last_result=row["last_result"],
            created_at=row["created_at"],
        )

    def _update_status(self, loop_id: str, status: str,
                       last_result: Optional[str] = None,
                       increment: bool = False):
        with self._connect() as conn:
            if increment:
                conn.execute(
                    "UPDATE loops SET status = ?, last_result = ?, "
                    "current_iteration = current_iteration + 1, "
                    "updated_at = datetime('now') WHERE id = ?",
                    (status, last_result, loop_id),
                )
            else:
                conn.execute(
                    "UPDATE loops SET status = ?, last_result = ?, "
                    "updated_at = datetime('now') WHERE id = ?",
                    (status, last_result, loop_id),
                )

    # ── CRUD ──────────────────────────────────────────────────

    def create(
        self,
        session_id: str,
        name: str,
        trigger_type: str,
        trigger_config: Dict,
        action_prompt: str,
        max_iterations: int = 0,
    ) -> LoopSpec:
        """Create a new loop spec in the database.

        Returns:
            The created LoopSpec.
        """
        loop_id = uuid.uuid4().hex[:12]
        now = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO loops "
                "(id, session_id, name, trigger_type, trigger_config, "
                " action_prompt, max_iterations, current_iteration, status, "
                " last_result, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'pending', NULL, ?, ?)",
                (
                    loop_id,
                    session_id,
                    name,
                    trigger_type,
                    json.dumps(trigger_config, ensure_ascii=False),
                    action_prompt,
                    max_iterations,
                    now,
                    now,
                ),
            )
        return LoopSpec(
            id=loop_id,
            session_id=session_id,
            name=name,
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            action_prompt=action_prompt,
            max_iterations=max_iterations,
            current_iteration=0,
            status="pending",
            last_result=None,
            created_at=now,
        )

    def get(self, loop_id: str) -> Optional[LoopSpec]:
        """Retrieve a loop spec by id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM loops WHERE id = ?", (loop_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_spec(row)

    def list_loops(self, session_id: str) -> List[LoopSpec]:
        """List all loops for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM loops WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [self._row_to_spec(r) for r in rows]

    def list_active(self, session_id: str) -> List[LoopSpec]:
        """List loops with status running or paused for a session."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM loops WHERE session_id = ? AND status IN ('running', 'paused') "
                "ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [self._row_to_spec(r) for r in rows]

    # ── Lifecycle control ─────────────────────────────────────

    def start(self, loop_id: str) -> bool:
        """Start a loop in a background daemon thread.

        Returns:
            True if started, False if loop not found or already running.
        """
        spec = self.get(loop_id)
        if spec is None:
            logger.error("Cannot start loop %s: not found", loop_id)
            return False
        if spec.status == "running":
            logger.warning("Loop %s is already running", loop_id)
            return False

        stop_event = threading.Event()
        pause_event = threading.Event()
        # pause_event is "set" when NOT paused, so threads proceed normally.
        # When we pause, we clear it; resume sets it again.
        pause_event.set()

        self._stop_events[loop_id] = stop_event
        self._pause_events[loop_id] = pause_event

        self._update_status(loop_id, "running")

        thread = threading.Thread(
            target=self._run_loop,
            args=(loop_id,),
            daemon=True,
            name=f"loop-{loop_id}",
        )
        self._threads[loop_id] = thread
        thread.start()
        logger.info("Loop %s started (trigger=%s)", loop_id, spec.trigger_type)
        return True

    def pause(self, loop_id: str) -> bool:
        """Pause a running loop.

        Returns:
            True if paused, False if not found or not running.
        """
        if loop_id not in self._pause_events:
            return False
        spec = self.get(loop_id)
        if spec is None or spec.status != "running":
            return False
        self._pause_events[loop_id].clear()
        self._update_status(loop_id, "paused")
        logger.info("Loop %s paused", loop_id)
        return True

    def resume(self, loop_id: str) -> bool:
        """Resume a paused loop.

        Returns:
            True if resumed, False if not found or not paused.
        """
        if loop_id not in self._pause_events:
            return False
        spec = self.get(loop_id)
        if spec is None or spec.status != "paused":
            return False
        self._pause_events[loop_id].set()
        self._update_status(loop_id, "running")
        logger.info("Loop %s resumed", loop_id)
        return True

    def stop(self, loop_id: str) -> bool:
        """Stop a loop.

        Returns:
            True if stopped, False if not found.
        """
        if loop_id not in self._stop_events:
            return False
        self._stop_events[loop_id].set()
        # Also unpause so the thread doesn't hang on the pause gate
        if loop_id in self._pause_events:
            self._pause_events[loop_id].set()
        self._update_status(loop_id, "completed")
        logger.info("Loop %s stopped", loop_id)
        return True

    # ── Thread worker ─────────────────────────────────────────

    def _run_loop(self, loop_id: str):
        """Internal: the thread function that runs the loop body."""
        stop_event = self._stop_events[loop_id]
        pause_event = self._pause_events[loop_id]

        while not stop_event.is_set():
            spec = self.get(loop_id)
            if spec is None:
                logger.error("Loop %s disappeared from DB", loop_id)
                return

            # Check max_iterations
            if spec.max_iterations > 0 and spec.current_iteration >= spec.max_iterations:
                self._update_status(loop_id, "completed")
                self.hooks.emit(HookEvent.LOOP_COMPLETE, {
                    "loop_id": loop_id,
                    "session_id": spec.session_id,
                    "iterations": spec.current_iteration,
                })
                logger.info("Loop %s completed (max_iterations=%d reached)",
                            loop_id, spec.max_iterations)
                return

            # Determine wait and execution based on trigger type
            if spec.trigger_type == "interval":
                interval = spec.trigger_config.get("seconds", 60)
                # Wait for interval, but check stop_event every 0.1s
                if _interruptible_sleep(stop_event, interval):
                    return  # stopped during sleep

                # Pause gate: block here while paused
                pause_event.wait()
                if stop_event.is_set():
                    return

                # Execute action
                if self._execute_tick(loop_id, spec):
                    return

            elif spec.trigger_type == "condition":
                poll_interval = spec.trigger_config.get("poll_seconds", 10)
                check_prompt = spec.trigger_config.get("check_prompt", "")

                # Wait poll interval
                if _interruptible_sleep(stop_event, poll_interval):
                    return

                # Pause gate
                pause_event.wait()
                if stop_event.is_set():
                    return

                # Check condition via run_fn
                try:
                    check_result = self._run_fn(check_prompt)
                except Exception as e:
                    logger.error("Loop %s condition check failed: %s", loop_id, e)
                    self._update_status(loop_id, "failed", str(e))
                    self.hooks.emit(HookEvent.LOOP_ERROR, {
                        "loop_id": loop_id,
                        "error": str(e),
                    })
                    return

                # If condition result looks truthy, execute the action
                if _is_truthy(check_result):
                    if self._execute_tick(loop_id, spec):
                        return
            else:
                logger.error("Loop %s has unknown trigger_type: %s",
                             loop_id, spec.trigger_type)
                self._update_status(loop_id, "failed",
                                    f"Unknown trigger_type: {spec.trigger_type}")
                return

        # Loop exited because stop_event was set
        logger.debug("Loop %s thread exiting (stop_event set)", loop_id)

    def _execute_tick(self, loop_id: str, spec: LoopSpec) -> bool:
        """Execute one iteration of the loop.

        Returns True when this tick completed the loop's configured work.
        """
        try:
            result = self._run_fn(spec.action_prompt)
            self._update_status(loop_id, "running", last_result=result, increment=True)
            updated = self.get(loop_id)
            self.hooks.emit(HookEvent.LOOP_TICK, {
                "loop_id": loop_id,
                "session_id": spec.session_id,
                "iteration": spec.current_iteration + 1,
                "result": result,
            })
            if (
                updated is not None
                and updated.max_iterations > 0
                and updated.current_iteration >= updated.max_iterations
            ):
                self._update_status(loop_id, "completed", last_result=result)
                self.hooks.emit(HookEvent.LOOP_COMPLETE, {
                    "loop_id": loop_id,
                    "session_id": updated.session_id,
                    "iterations": updated.current_iteration,
                })
                logger.info(
                    "Loop %s completed (max_iterations=%d reached)",
                    loop_id,
                    updated.max_iterations,
                )
                return True
            return False
        except Exception as e:
            logger.error("Loop %s tick failed: %s", loop_id, e)
            self._update_status(loop_id, "failed", str(e))
            self.hooks.emit(HookEvent.LOOP_ERROR, {
                "loop_id": loop_id,
                "error": str(e),
            })
            return True


# ── Helpers ───────────────────────────────────────────────────

def _utcnow_iso() -> str:
    """Return current UTC time as ISO format string matching SQLite datetime."""
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _interruptible_sleep(stop_event: threading.Event, seconds: float) -> bool:
    """Sleep for `seconds`, checking stop_event every 100ms.

    Returns True if stop_event was set during sleep (caller should exit).
    """
    end_time = time.monotonic() + seconds
    while time.monotonic() < end_time:
        if stop_event.is_set():
            return True
        time.sleep(min(0.1, end_time - time.monotonic()))
    return stop_event.is_set()


def _is_truthy(value: str) -> bool:
    """Determine if a string result should be treated as True.

    Truthy: "true", "yes", "1", "ok", non-empty string that doesn't look
    like a denial.
    """
    if not value:
        return False
    stripped = value.strip().lower()
    if stripped in ("false", "no", "0", "none", "null", "n/a", "negatory"):
        return False
    # For condition checks, treat any non-empty, non-false result as truthy
    return True


# ── Tool registration ────────────────────────────────────────

def register_loop_tools(registry, loop_mgr: LoopManager):
    """Register loop management tools on a ToolRegistry.

    Tools:
        loop_start, loop_stop, loop_pause, loop_resume, loop_list
    """

    def _loop_start(args: dict) -> str:
        session_id = args.get("session_id", "")
        name = args.get("name", "")
        trigger_type = args.get("trigger_type", "interval")
        trigger_config = args.get("trigger_config", {})
        action_prompt = args.get("action_prompt", "")
        max_iterations = args.get("max_iterations", 0)

        if not session_id:
            return json.dumps({"error": "session_id is required"}, ensure_ascii=False)
        if not name:
            return json.dumps({"error": "name is required"}, ensure_ascii=False)
        if not action_prompt:
            return json.dumps({"error": "action_prompt is required"}, ensure_ascii=False)

        spec = loop_mgr.create(
            session_id=session_id,
            name=name,
            trigger_type=trigger_type,
            trigger_config=trigger_config,
            action_prompt=action_prompt,
            max_iterations=max_iterations,
        )
        started = loop_mgr.start(spec.id)
        return json.dumps({
            "id": spec.id,
            "status": "running" if started else spec.status,
            "name": spec.name,
            "trigger_type": spec.trigger_type,
        }, ensure_ascii=False)

    def _loop_stop(args: dict) -> str:
        loop_id = args.get("loop_id", "")
        if not loop_id:
            return json.dumps({"error": "loop_id is required"}, ensure_ascii=False)
        success = loop_mgr.stop(loop_id)
        if not success:
            return json.dumps({"error": f"Loop {loop_id} not found or not running"},
                              ensure_ascii=False)
        return json.dumps({"id": loop_id, "status": "completed"}, ensure_ascii=False)

    def _loop_pause(args: dict) -> str:
        loop_id = args.get("loop_id", "")
        if not loop_id:
            return json.dumps({"error": "loop_id is required"}, ensure_ascii=False)
        success = loop_mgr.pause(loop_id)
        if not success:
            return json.dumps({"error": f"Loop {loop_id} not found or not running"},
                              ensure_ascii=False)
        return json.dumps({"id": loop_id, "status": "paused"}, ensure_ascii=False)

    def _loop_resume(args: dict) -> str:
        loop_id = args.get("loop_id", "")
        if not loop_id:
            return json.dumps({"error": "loop_id is required"}, ensure_ascii=False)
        success = loop_mgr.resume(loop_id)
        if not success:
            return json.dumps({"error": f"Loop {loop_id} not found or not paused"},
                              ensure_ascii=False)
        return json.dumps({"id": loop_id, "status": "running"}, ensure_ascii=False)

    def _loop_list(args: dict) -> str:
        session_id = args.get("session_id", "")
        if not session_id:
            return json.dumps({"error": "session_id is required"}, ensure_ascii=False)
        loops = loop_mgr.list_loops(session_id)
        return json.dumps({
            "loops": [
                {
                    "id": l.id,
                    "name": l.name,
                    "trigger_type": l.trigger_type,
                    "status": l.status,
                    "current_iteration": l.current_iteration,
                    "max_iterations": l.max_iterations,
                    "last_result": l.last_result,
                    "created_at": l.created_at,
                }
                for l in loops
            ],
            "total": len(loops),
        }, ensure_ascii=False)

    registry.register(
        name="loop_start",
        description="Start a background loop task with interval or condition trigger.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID"},
                "name": {"type": "string", "description": "Descriptive name for the loop"},
                "trigger_type": {
                    "type": "string",
                    "enum": ["interval", "condition"],
                    "description": "Trigger type",
                },
                "trigger_config": {
                    "type": "object",
                    "description": "Config: interval={'seconds':N} or condition={'check_prompt':'...'}",
                },
                "action_prompt": {
                    "type": "string",
                    "description": "Prompt to execute on each tick",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Max iterations (0 = unlimited)",
                    "default": 0,
                },
            },
            "required": ["session_id", "name", "trigger_type", "trigger_config", "action_prompt"],
        },
        handler=_loop_start,
    )

    registry.register(
        name="loop_stop",
        description="Stop a running or paused loop.",
        parameters={
            "type": "object",
            "properties": {
                "loop_id": {"type": "string", "description": "Loop ID to stop"},
            },
            "required": ["loop_id"],
        },
        handler=_loop_stop,
    )

    registry.register(
        name="loop_pause",
        description="Pause a running loop.",
        parameters={
            "type": "object",
            "properties": {
                "loop_id": {"type": "string", "description": "Loop ID to pause"},
            },
            "required": ["loop_id"],
        },
        handler=_loop_pause,
    )

    registry.register(
        name="loop_resume",
        description="Resume a paused loop.",
        parameters={
            "type": "object",
            "properties": {
                "loop_id": {"type": "string", "description": "Loop ID to resume"},
            },
            "required": ["loop_id"],
        },
        handler=_loop_resume,
    )

    registry.register(
        name="loop_list",
        description="List all loops for a session.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID"},
            },
            "required": ["session_id"],
        },
        handler=_loop_list,
    )
