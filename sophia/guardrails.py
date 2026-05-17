"""Tool guardrails for rate limiting and loop detection.

Prevents runaway tool usage by enforcing:
- Maximum consecutive calls to the same tool
- Maximum total calls per minute across all tools

Integrates with the hook system via tool.pre_dispatch.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)


class ToolGuardrails:
    def __init__(self, hooks: HookManager = None, max_consecutive_calls: int = 5,
                 max_calls_per_minute: int = 60):
        self.hooks = hooks
        self.max_consecutive = max_consecutive_calls
        self.max_per_minute = max_calls_per_minute
        self._call_history: List[Dict[str, Any]] = []
        self._last_tool: Optional[str] = None
        self._consecutive_count: int = 0

    def check_allowed(self, tool_name: str, args: Dict) -> Tuple[bool, str]:
        """Check if a tool call is allowed.

        Returns (allowed: bool, reason: str).

        Blocks if:
        1. Same tool called consecutively more than max_consecutive times
        2. Total calls in last minute exceed max_per_minute
        """
        # Check consecutive limit
        if tool_name == self._last_tool:
            if self._consecutive_count >= self.max_consecutive:
                reason = (
                    f"Tool '{tool_name}' called consecutively "
                    f"{self._consecutive_count} times (limit: {self.max_consecutive})"
                )
                logger.warning("Guardrail blocked: %s", reason)
                return False, reason
        else:
            # Different tool - consecutive counter will reset in record_call
            pass

        # Check rate limit (calls in last 60 seconds)
        self._cleanup_old_history()
        calls_in_window = len(self._call_history)
        if calls_in_window >= self.max_per_minute:
            reason = (
                f"Rate limit exceeded: {calls_in_window} calls in last 60s "
                f"(limit: {self.max_per_minute})"
            )
            logger.warning("Guardrail blocked: %s", reason)
            return False, reason

        return True, ""

    def record_call(self, tool_name: str, args: Dict):
        """Record a tool call for tracking."""
        now = time.time()

        # Update consecutive tracking
        if tool_name == self._last_tool:
            self._consecutive_count += 1
        else:
            self._last_tool = tool_name
            self._consecutive_count = 1

        # Add to history
        self._call_history.append({
            "tool": tool_name,
            "timestamp": now,
            "args": args,
        })

        logger.debug(
            "Recorded tool call: %s (consecutive=%d, total_in_window=%d)",
            tool_name, self._consecutive_count, len(self._call_history),
        )

    def check_hook(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook handler for tool.pre_dispatch.

        Checks if the call is allowed. If not, sets blocked=True and block_reason.
        """
        tool_name = context.get("tool", "")
        args = context.get("args", {})

        allowed, reason = self.check_allowed(tool_name, args)

        if not allowed:
            context["blocked"] = True
            context["block_reason"] = reason

            # Emit guardrail block event
            if self.hooks:
                self.hooks.emit(HookEvent.GUARDRAIL_BLOCK, {
                    "tool": tool_name,
                    "reason": reason,
                    "consecutive_count": self._consecutive_count,
                    "total_calls": len(self._call_history),
                })

            logger.warning("Guardrail blocked tool '%s': %s", tool_name, reason)
        else:
            # Record the call so it counts toward limits
            self.record_call(tool_name, args)

        return context

    def reset(self):
        """Reset all tracking state."""
        self._call_history.clear()
        self._last_tool = None
        self._consecutive_count = 0
        logger.info("Guardrails state reset")

    def _cleanup_old_history(self):
        """Remove entries older than 60 seconds from call history."""
        now = time.time()
        cutoff = now - 60.0
        original_len = len(self._call_history)
        self._call_history = [
            entry for entry in self._call_history
            if entry["timestamp"] > cutoff
        ]
        removed = original_len - len(self._call_history)
        if removed > 0:
            logger.debug("Cleaned up %d old history entries", removed)
