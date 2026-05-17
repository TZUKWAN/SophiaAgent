"""Trajectory recording for SophiaAgent.

Records execution events as JSONL for model evaluation and training.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    def __init__(self, hooks: HookManager, output_dir: str):
        self.hooks = hooks
        self.output_dir = output_dir
        self._recording_sessions: Dict[str, List[Dict]] = {}
        self._registered = False
        os.makedirs(output_dir, exist_ok=True)

    def _register_hooks(self):
        if self._registered:
            return
        events = [
            HookEvent.AGENT_PRE_RUN, HookEvent.AGENT_POST_RUN,
            HookEvent.TOOL_PRE_DISPATCH, HookEvent.TOOL_POST_DISPATCH,
            HookEvent.TOOL_ERROR, HookEvent.GOAL_CREATED, HookEvent.GOAL_COMPLETED,
            HookEvent.SUBAGENT_SPAWN, HookEvent.SUBAGENT_COMPLETE,
            HookEvent.LOOP_TICK, HookEvent.CONTEXT_COMPRESS,
        ]
        for event in events:
            self.hooks.register(event, self._on_event, priority=999, name=f"trajectory_{event}")
        self._registered = True

    def start_recording(self, session_id: str):
        self._register_hooks()
        self._recording_sessions[session_id] = []

    def stop_recording(self, session_id: str) -> str:
        entries = self._recording_sessions.pop(session_id, [])
        filepath = os.path.join(self.output_dir, f"trajectory_{session_id}.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return filepath

    def _on_event(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Record to all active sessions
        entry = {
            "timestamp": time.time(),
            "context": {k: str(v)[:500] for k, v in context.items()},
        }
        for session_id, entries in self._recording_sessions.items():
            entries.append(entry)
        return context

    def get_entries(self, session_id: str) -> List[Dict]:
        return self._recording_sessions.get(session_id, [])

    def is_recording(self, session_id: str) -> bool:
        return session_id in self._recording_sessions
