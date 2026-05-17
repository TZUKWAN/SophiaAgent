"""Simplified tool registry for SophiaAgent.

Tools are registered as {name: {schema, handler}} pairs.
No AST discovery, no plugins, no async -- just a dict with dispatch.
"""

import json
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all SophiaAgent tools."""

    def __init__(self, hooks=None):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._hooks = hooks  # Optional[HookManager]

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
    ) -> None:
        """Register a tool.

        Args:
            name: Tool name (e.g. "file_read").
            description: What the tool does.
            parameters: JSON Schema for the tool's parameters.
            handler: Callable that takes a dict of args and returns a JSON string.
        """
        if name in self._tools:
            logger.warning("Tool '%s' already registered, overwriting", name)

        schema = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self._tools[name] = {"schema": schema, "handler": handler}
        logger.debug("Registered tool: %s", name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-format tool schemas for all registered tools."""
        return [
            {"type": "function", "function": t["schema"]}
            for t in self._tools.values()
        ]

    def dispatch(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name.

        Returns:
            JSON string result. Errors are returned as {"error": "..."}.
        """
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)

        # Pre-dispatch hook (guardrails, security checks)
        if self._hooks:
            ctx = self._hooks.emit("tool.pre_dispatch", {"tool": name, "args": args})
            if ctx.get("blocked"):
                reason = ctx.get("block_reason", "Blocked by guardrail")
                return json.dumps({"error": reason}, ensure_ascii=False)

        try:
            result = tool["handler"](args)
            if isinstance(result, str):
                json_result = result
            else:
                json_result = json.dumps(result, ensure_ascii=False)

            # Post-dispatch hook (logging, trajectory recording)
            if self._hooks:
                self._hooks.emit("tool.post_dispatch", {
                    "tool": name, "args": args, "result": json_result
                })

            return json_result
        except Exception as e:
            logger.exception("Tool %s execution failed", name)
            error_result = json.dumps(
                {"error": f"Tool execution failed: {type(e).__name__}: {e}"},
                ensure_ascii=False,
            )
            # Error hook (recovery, learning)
            if self._hooks:
                self._hooks.emit("tool.error", {
                    "tool": name, "args": args, "error": str(e), "error_type": type(e).__name__
                })
            return error_result

    def list_tools(self) -> List[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def set_hooks(self, hooks):
        """Set hook manager after construction."""
        self._hooks = hooks
