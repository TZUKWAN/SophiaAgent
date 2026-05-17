"""Event hook system for SophiaAgent.

The nervous system of the agent -- all mechanisms (Goal, SubAgent, Loop,
Guardrails, Recovery, etc.) plug into this via register/emit.

Usage:
    hooks = HookManager()
    hooks.register("tool.pre_dispatch", my_handler, priority=50)
    context = hooks.emit("tool.pre_dispatch", {"tool": "file_read", "args": {...}})
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HookEvent:
    """Predefined event names used throughout SophiaAgent."""

    # Agent lifecycle
    AGENT_PRE_RUN = "agent.pre_run"
    AGENT_POST_RUN = "agent.post_run"
    AGENT_PRE_STREAM = "agent.pre_stream"
    AGENT_POST_STREAM = "agent.post_stream"

    # Tool lifecycle
    TOOL_PRE_DISPATCH = "tool.pre_dispatch"
    TOOL_POST_DISPATCH = "tool.post_dispatch"
    TOOL_ERROR = "tool.error"

    # Goal lifecycle
    GOAL_CREATED = "goal.created"
    GOAL_UPDATED = "goal.updated"
    GOAL_COMPLETED = "goal.completed"
    GOAL_FAILED = "goal.failed"

    # SubAgent lifecycle
    SUBAGENT_SPAWN = "subagent.spawn"
    SUBAGENT_COMPLETE = "subagent.complete"
    SUBAGENT_ERROR = "subagent.error"

    # Swarm lifecycle
    SWARM_ANALYZED = "swarm.analyzed"
    SWARM_PLANNED = "swarm.planned"
    SWARM_SPAWNED = "swarm.spawned"
    SWARM_STAGE_START = "swarm.stage_start"
    SWARM_STAGE_END = "swarm.stage_end"
    SWARM_AGENT_COMPLETE = "swarm.agent_complete"
    SWARM_AGENT_ERROR = "swarm.agent_error"
    SWARM_SYNTHESIZED = "swarm.synthesized"

    # Loop lifecycle
    LOOP_TICK = "loop.tick"
    LOOP_COMPLETE = "loop.complete"
    LOOP_ERROR = "loop.error"

    # Memory
    MEMORY_STORE = "memory.store"
    MEMORY_RECALL = "memory.recall"

    # Context
    CONTEXT_COMPRESS = "context.compress"

    # Credential
    CREDENTIAL_ROTATE = "credential.rotate"
    CREDENTIAL_FAILOVER = "credential.failover"

    # Recovery
    RECOVERY_RETRY = "recovery.retry"

    # Guardrail
    GUARDRAIL_BLOCK = "guardrail.block"

    # Trajectory
    TRAJECTORY_RECORD = "trajectory.record"

    # Scheduler
    SCHEDULER_FIRE = "scheduler.fire"

    # Kanban
    KANBAN_CARD_CREATED = "kanban.card_created"
    KANBAN_CARD_MOVED = "kanban.card_moved"

    # Learning
    LEARNING_ANALYSIS = "learning.analysis"

    # Security
    SECURITY_ALERT = "security.alert"

    # Snapshot
    SNAPSHOT_CREATED = "snapshot.created"
    SNAPSHOT_RESTORED = "snapshot.restored"


class HookManager:
    """Central event hook manager.

    Handlers are called in priority order (lower = earlier).
    Each handler receives and can modify the context dict.
    If a handler sets context["blocked"] = True, subsequent handlers
    are skipped and the emit returns early.
    """

    def __init__(self):
        self._hooks: Dict[str, List[Dict[str, Any]]] = {}

    def register(
        self,
        event: str,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        priority: int = 100,
        name: Optional[str] = None,
    ) -> None:
        """Register a hook handler for an event.

        Args:
            event: Event name (use HookEvent constants).
            handler: Callable taking a context dict, returning an updated context dict.
                     Can also return None (context is used as-is).
            priority: Lower numbers run first. Default 100.
            name: Optional name for easy removal.
        """
        if event not in self._hooks:
            self._hooks[event] = []

        entry = {
            "handler": handler,
            "priority": priority,
            "name": name or handler.__name__,
        }
        self._hooks[event].append(entry)
        self._hooks[event].sort(key=lambda x: x["priority"])
        logger.debug(
            "Hook registered: event=%s, name=%s, priority=%d",
            event, entry["name"], priority,
        )

    def emit(self, event: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Emit an event, running all registered handlers in priority order.

        Args:
            event: Event name.
            context: Initial context dict. Modified in-place by handlers.

        Returns:
            The (possibly modified) context dict.
        """
        if context is None:
            context = {}

        handlers = self._hooks.get(event, [])
        if not handlers:
            return context

        for entry in handlers:
            try:
                result = entry["handler"](context)
                if result is not None:
                    context = result
            except Exception as e:
                logger.warning(
                    "Hook handler '%s' failed for event '%s': %s",
                    entry["name"], event, e,
                )

            if context.get("blocked"):
                logger.debug(
                    "Hook chain blocked at '%s' for event '%s': %s",
                    entry["name"], event, context.get("block_reason", ""),
                )
                break

        return context

    def remove(self, event: str, handler: Optional[Callable] = None, name: Optional[str] = None) -> bool:
        """Remove a handler by reference or name.

        Returns True if a handler was removed.
        """
        if event not in self._hooks:
            return False

        original_len = len(self._hooks[event])
        if handler:
            self._hooks[event] = [
                h for h in self._hooks[event] if h["handler"] is not handler
            ]
        elif name:
            self._hooks[event] = [
                h for h in self._hooks[event] if h["name"] != name
            ]
        else:
            return False

        removed = len(self._hooks[event]) < original_len
        if removed:
            logger.debug("Hook removed: event=%s", event)
        return removed

    def remove_all(self, event: Optional[str] = None) -> None:
        """Remove all handlers for an event, or all events if event is None."""
        if event:
            self._hooks.pop(event, None)
        else:
            self._hooks.clear()

    def list_hooks(self, event: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """List registered hooks, optionally filtered by event.

        Returns a dict mapping event names to lists of handler info dicts.
        Handler info includes 'name' and 'priority' but NOT the callable.
        """
        if event:
            entries = self._hooks.get(event, [])
            return {event: [{"name": h["name"], "priority": h["priority"]} for h in entries]}
        return {
            ev: [{"name": h["name"], "priority": h["priority"]} for h in handlers]
            for ev, handlers in self._hooks.items()
        }

    def has_hooks(self, event: str) -> bool:
        """Check if any handlers are registered for an event."""
        return bool(self._hooks.get(event))
