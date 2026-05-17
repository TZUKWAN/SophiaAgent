"""Closed-loop learning for SophiaAgent.

Analyzes execution results to extract patterns and suggest improvements.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)


class LearningManager:
    def __init__(self, hooks: HookManager = None, memory=None):
        self.hooks = hooks
        self.memory = memory  # Optional MemoryManager
        self._execution_log: List[Dict] = []

    def record_execution(self, event: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook handler that records execution events for analysis."""
        entry = {
            "event": event,
            "timestamp": context.get("timestamp") or time.time(),
            "context": {k: str(v)[:200] for k, v in context.items()},
        }

        # Extract structured info for tool events
        if event == "tool.pre_dispatch":
            entry["tool_name"] = context.get("tool", "")
            entry["args"] = context.get("args", {})
            entry["phase"] = "start"
        elif event == "tool.post_dispatch":
            entry["tool_name"] = context.get("tool", "")
            entry["args"] = context.get("args", {})
            entry["result"] = str(context.get("result", ""))[:2000]
            entry["success"] = True
            entry["score"] = self._score_result(context.get("result"), None)
            entry["phase"] = "end"
        elif event == "tool.error":
            entry["tool_name"] = context.get("tool", "")
            entry["args"] = context.get("args", {})
            entry["error"] = str(context.get("error", ""))[:1000]
            entry["success"] = False
            entry["score"] = 0.0
            entry["phase"] = "error"

        self._execution_log.append(entry)
        # Keep only last 100 entries
        if len(self._execution_log) > 100:
            self._execution_log = self._execution_log[-100:]
        return context

    def _score_result(self, result: Any, error: Optional[str] = None) -> float:
        """Score an execution from 0.0 to 1.0 based on result quality."""
        if error:
            return 0.0
        score = 1.0
        if isinstance(result, str):
            try:
                data = json.loads(result)
                if "error" in data and data["error"]:
                    score = 0.3
                if "status" in data and data.get("status") != "success":
                    score = 0.2
                # Boost score for rich outputs
                if "apa" in data:
                    score = min(score + 0.1, 1.0)
                if "result_id" in data:
                    score = min(score + 0.05, 1.0)
            except (json.JSONDecodeError, TypeError):
                pass
        return score

    def get_structured_log(self) -> List[Dict]:
        """Return execution log with structured tool-call entries."""
        return list(self._execution_log)

    def analyze_execution(self, session_id: str = None, goal_id: str = None) -> Dict:
        """Analyze recorded executions for patterns."""
        entries = self._execution_log
        if not entries:
            return {"patterns": [], "summary": "No execution data available"}

        # Count events by type
        event_counts = {}
        tool_calls = []
        errors = []
        for entry in entries:
            evt = entry["event"]
            event_counts[evt] = event_counts.get(evt, 0) + 1
            if evt == "tool.post_dispatch":
                tool_calls.append(entry.get("tool_name", entry["context"].get("tool", "")))
            elif evt == "tool.error":
                errors.append(entry.get("tool_name", entry["context"].get("tool", "")))

        # Find most used tools
        tool_usage = {}
        for t in tool_calls:
            tool_usage[t] = tool_usage.get(t, 0) + 1

        # Find error patterns
        error_tools = {}
        for t in errors:
            error_tools[t] = error_tools.get(t, 0) + 1

        patterns = []
        # Pattern: frequently used tools
        if tool_usage:
            top_tool = max(tool_usage, key=tool_usage.get)
            patterns.append({
                "type": "frequent_tool",
                "tool": top_tool,
                "count": tool_usage[top_tool],
                "suggestion": f"Tool '{top_tool}' is used {tool_usage[top_tool]} times",
            })

        # Pattern: error-prone tools
        if error_tools:
            worst_tool = max(error_tools, key=error_tools.get)
            patterns.append({
                "type": "error_prone_tool",
                "tool": worst_tool,
                "error_count": error_tools[worst_tool],
                "suggestion": f"Tool '{worst_tool}' has {error_tools[worst_tool]} errors - consider alternative approach",
            })

        # Pattern: high-scoring sequences (new)
        sequences = self._extract_sequences()
        if sequences:
            top_seq = max(sequences, key=lambda s: s["score"])
            if top_seq["score"] >= 0.8 and top_seq["count"] >= 3:
                patterns.append({
                    "type": "workflow_sequence",
                    "sequence": top_seq["tools"],
                    "count": top_seq["count"],
                    "avg_score": top_seq["score"],
                    "suggestion": f"High-success workflow detected: {' -> '.join(top_seq['tools'])}",
                })

        return {
            "patterns": patterns,
            "summary": f"Analyzed {len(entries)} events: {len(tool_calls)} tool calls, {len(errors)} errors",
            "event_counts": event_counts,
            "tool_usage": tool_usage,
        }

    def _extract_sequences(self, gap_threshold: float = 300.0) -> List[Dict]:
        """Extract tool call sequences from log.

        A sequence is a consecutive run of tool calls (post_dispatch or error)
        with no gap larger than gap_threshold seconds.
        """
        from collections import Counter
        tool_events = [e for e in self._execution_log
                       if e.get("tool_name") and e.get("phase") in ("end", "error")]
        if not tool_events:
            return []

        # Group into sequences by time gaps
        sequences = []
        current = [tool_events[0]]
        for i in range(1, len(tool_events)):
            prev_ts = current[-1].get("timestamp", 0)
            curr_ts = tool_events[i].get("timestamp", 0)
            if curr_ts - prev_ts <= gap_threshold:
                current.append(tool_events[i])
            else:
                if len(current) >= 2:
                    sequences.append(current)
                current = [tool_events[i]]
        if len(current) >= 2:
            sequences.append(current)

        # Count and score sequences
        seq_counter = Counter()
        seq_scores = {}
        for seq in sequences:
            tools = tuple(e["tool_name"] for e in seq)
            seq_counter[tools] += 1
            scores = [e.get("score", 0.0) for e in seq if e.get("phase") == "end"]
            avg = sum(scores) / len(scores) if scores else 0.0
            if tools not in seq_scores or avg > seq_scores[tools]:
                seq_scores[tools] = avg

        return [
            {"tools": list(tools), "count": cnt, "score": seq_scores.get(tools, 0.0)}
            for tools, cnt in seq_counter.most_common()
        ]

    def extract_patterns(self, session_id: str = None) -> List[Dict]:
        """Extract reusable patterns from execution history."""
        analysis = self.analyze_execution(session_id)
        return analysis.get("patterns", [])

    def suggest_improvements(self, analysis: Dict) -> List[Dict]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        patterns = analysis.get("patterns", [])

        for pattern in patterns:
            if pattern["type"] == "error_prone_tool":
                suggestions.append({
                    "type": "tool_replacement",
                    "description": f"Consider replacing or fixing usage of '{pattern['tool']}'",
                    "priority": "high" if pattern["error_count"] > 3 else "medium",
                })
            elif pattern["type"] == "frequent_tool":
                count = pattern["count"]
                if count > 10:
                    suggestions.append({
                        "type": "automation",
                        "description": f"Tool '{pattern['tool']}' is used very frequently ({count}x) - consider creating a Loop",
                        "priority": "medium",
                    })
            elif pattern["type"] == "workflow_sequence":
                suggestions.append({
                    "type": "skill_creation",
                    "description": pattern["suggestion"],
                    "priority": "high",
                    "sequence": pattern["sequence"],
                })

        return suggestions

    def on_goal_completed(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook handler for goal.completed - auto-analyze."""
        analysis = self.analyze_execution()
        if analysis["patterns"]:
            if self.hooks:
                self.hooks.emit(HookEvent.LEARNING_ANALYSIS, {
                    "goal_id": context.get("goal_id"),
                    "patterns": len(analysis["patterns"]),
                    "summary": analysis["summary"],
                })
            # Store in memory if available
            if self.memory:
                try:
                    suggestions = self.suggest_improvements(analysis)
                    if suggestions:
                        self.memory.store(
                            session_id=context.get("session_id", "learning"),
                            key=f"learning_{context.get('goal_id', 'auto')}",
                            content=json.dumps(suggestions, ensure_ascii=False),
                            category="research_history",
                        )
                except Exception:
                    pass
        return context
