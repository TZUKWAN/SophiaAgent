"""ExecutionPatternMiner: Extract reusable workflow patterns from execution logs.

A pattern is a sequence of tool calls that appears frequently and has a high
success rate.  The miner scans structured execution logs, groups consecutive
tool invocations into sessions, scores each session, and surfaces the
best-performing sequences as candidate skills.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolInvocation:
    """A single tool call with metadata."""
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    score: float = 1.0
    error: Optional[str] = None
    result_id: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class WorkflowPattern:
    """A candidate skill pattern mined from logs."""
    sequence: List[str]                          # ordered tool names
    frequency: int                               # how many times seen
    success_rate: float                          # fraction of sessions w/o error
    avg_score: float                             # average per-step score
    total_sessions: int                          # number of sessions
    params_template: Dict[str, Dict[str, Any]]   # most common params per step
    trigger_keywords: List[str] = field(default_factory=list)
    avg_duration: float = 0.0


class ExecutionPatternMiner:
    """Mine workflow patterns from structured execution logs."""

    def __init__(
        self,
        min_sequence_length: int = 2,
        min_frequency: int = 2,
        min_success_rate: float = 0.7,
        max_gap_seconds: float = 300.0,
    ):
        self.min_sequence_length = min_sequence_length
        self.min_frequency = min_frequency
        self.min_success_rate = min_success_rate
        self.max_gap_seconds = max_gap_seconds

    def mine(self, log_entries: List[Dict]) -> List[WorkflowPattern]:
        """Mine patterns from a list of structured log entries.

        Args:
            log_entries: output of LearningManager.get_structured_log()

        Returns:
            Sorted list of WorkflowPattern (best first).
        """
        sessions = self._split_into_sessions(log_entries)
        if not sessions:
            return []

        # Extract all sequences of length >= min_sequence_length
        raw_sequences = []
        for session in sessions:
            tools = [inv.tool_name for inv in session]
            for length in range(self.min_sequence_length, len(tools) + 1):
                for start in range(len(tools) - length + 1):
                    seq = tuple(tools[start:start + length])
                    raw_sequences.append((seq, session[start:start + length]))

        # Aggregate by sequence
        seq_data: Dict[Tuple[str, ...], List[List[ToolInvocation]]] = defaultdict(list)
        for seq, invocations in raw_sequences:
            seq_data[seq].append(invocations)

        patterns = []
        for seq, groups in seq_data.items():
            freq = len(groups)
            if freq < self.min_frequency:
                continue

            success_count = sum(
                1 for g in groups if all(inv.success for inv in g)
            )
            success_rate = success_count / freq
            if success_rate < self.min_success_rate:
                continue

            all_scores = [inv.score for g in groups for inv in g]
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

            # Compute params template: most common params per step
            params_template = self._build_params_template(groups)

            # Extract trigger keywords from tool descriptions / args
            trigger_keywords = self._extract_trigger_keywords(groups)

            # Duration estimate
            durations = []
            for g in groups:
                if len(g) >= 2 and g[0].timestamp and g[-1].timestamp:
                    durations.append(g[-1].timestamp - g[0].timestamp)
            avg_duration = sum(durations) / len(durations) if durations else 0.0

            patterns.append(WorkflowPattern(
                sequence=list(seq),
                frequency=freq,
                success_rate=success_rate,
                avg_score=avg_score,
                total_sessions=freq,
                params_template=params_template,
                trigger_keywords=trigger_keywords,
                avg_duration=avg_duration,
            ))

        # Sort by composite score: success_rate * log(frequency) * avg_score
        def _score(p: WorkflowPattern) -> float:
            import math
            # Prefer longer sequences (len * 0.05 bonus) among otherwise equal patterns
            return p.success_rate * math.log1p(p.frequency) * p.avg_score + len(p.sequence) * 0.05

        patterns.sort(key=_score, reverse=True)
        return patterns

    def _split_into_sessions(
        self, log_entries: List[Dict]
    ) -> List[List[ToolInvocation]]:
        """Split log entries into sessions (consecutive tool calls)."""
        # Gather tool end/error events with timestamps
        tool_events = []
        for entry in log_entries:
            if entry.get("tool_name") and entry.get("phase") in ("end", "error"):
                tool_events.append(ToolInvocation(
                    tool_name=entry["tool_name"],
                    args=entry.get("args", {}),
                    success=entry.get("success", entry.get("phase") == "end"),
                    score=entry.get("score", 0.0),
                    error=entry.get("error"),
                    result_id=entry.get("result_id"),
                    timestamp=entry.get("timestamp", 0.0),
                ))

        if not tool_events:
            return []

        sessions: List[List[ToolInvocation]] = []
        current = [tool_events[0]]
        for i in range(1, len(tool_events)):
            gap = tool_events[i].timestamp - current[-1].timestamp
            if 0 <= gap <= self.max_gap_seconds:
                current.append(tool_events[i])
            else:
                if len(current) >= self.min_sequence_length:
                    sessions.append(current)
                current = [tool_events[i]]
        if len(current) >= self.min_sequence_length:
            sessions.append(current)

        return sessions

    @staticmethod
    def _build_params_template(
        groups: List[List[ToolInvocation]]
    ) -> Dict[str, Dict[str, Any]]:
        """For each position in the sequence, find the most common params."""
        if not groups:
            return {}

        seq_len = len(groups[0])
        template: Dict[str, Dict[str, Any]] = {}
        for pos in range(seq_len):
            param_counter: Counter = Counter()
            for group in groups:
                if pos < len(group):
                    # Hashable representation of params
                    params = group[pos].args
                    if params:
                        try:
                            key = json.dumps(params, sort_keys=True, ensure_ascii=False)
                            param_counter[key] += 1
                        except (TypeError, ValueError):
                            pass
            if param_counter:
                most_common_key = param_counter.most_common(1)[0][0]
                template[f"step_{pos}"] = json.loads(most_common_key)
        return template

    @staticmethod
    def _extract_trigger_keywords(
        groups: List[List[ToolInvocation]]
    ) -> List[str]:
        """Extract keywords from tool names and args to form trigger profile."""
        keywords = set()
        for group in groups:
            for inv in group:
                # Add tool name fragments
                parts = inv.tool_name.replace("research_", "").split("_")
                keywords.update(parts)
                # Add arg keys
                keywords.update(inv.args.keys())
        # Filter out generic terms
        generic = {"data", "path", "format", "result", "params", "args", "tool"}
        return sorted(k for k in keywords if k not in generic and len(k) > 2)
