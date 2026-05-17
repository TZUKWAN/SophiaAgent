"""Tools for inspecting and manually delegating swarm work.

Automatic startup is handled by SophiaAgent.run/run_stream. These tools are
advanced controls and compatibility surfaces, not the primary user path.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from sophia.swarm.orchestrator import SwarmOrchestrator
from sophia.tools.registry import ToolRegistry


def register_swarm_tools(registry: ToolRegistry, orchestrator: SwarmOrchestrator):
    def _delegate(args: Dict[str, Any]):
        result = orchestrator.delegate(
            session_id=args.get("session_id", "default"),
            prompt=args["prompt"],
            tools=args.get("tools"),
            goal_id=args.get("goal_id"),
        )
        return json.dumps(result, ensure_ascii=False)

    registry.register(
        "swarm_delegate",
        "Delegate one subtask to the automatic swarm system",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "prompt": {"type": "string"},
                "tools": {"type": "array", "items": {"type": "string"}},
                "goal_id": {"type": "string"},
            },
            "required": ["prompt"],
        },
        _delegate,
    )

    def _delegate_batch(args: Dict[str, Any]):
        results = orchestrator.delegate_batch(args.get("session_id", "default"), args.get("tasks", []))
        return json.dumps(results, ensure_ascii=False)

    registry.register(
        "swarm_delegate_batch",
        "Delegate multiple subtasks to the automatic swarm system",
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
            "required": ["tasks"],
        },
        _delegate_batch,
    )

    def _list(args: Dict[str, Any]):
        records = orchestrator.list_executions(args.get("session_id"))
        return json.dumps([record.to_summary() for record in records], ensure_ascii=False)

    registry.register(
        "swarm_list",
        "List automatic swarm execution history",
        {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
        },
        _list,
    )
