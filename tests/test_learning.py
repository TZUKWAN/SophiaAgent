"""Tests for sophia.learning.LearningManager."""

import json
import pytest

from sophia.hooks import HookEvent, HookManager
from sophia.learning import LearningManager


class TestRecordExecution:
    """Tests for record_execution."""

    def test_adds_entry_to_log(self):
        lm = LearningManager()
        assert len(lm._execution_log) == 0

        ctx = {"tool": "file_read", "args": {"path": "/tmp/test.txt"}}
        result = lm.record_execution("tool.post_dispatch", ctx)

        assert len(lm._execution_log) == 1
        entry = lm._execution_log[0]
        assert entry["event"] == "tool.post_dispatch"
        assert entry["context"]["tool"] == "file_read"

    def test_returns_context_unchanged(self):
        lm = LearningManager()
        ctx = {"tool": "file_read", "args": {"path": "/tmp/test.txt"}}
        result = lm.record_execution("tool.post_dispatch", ctx)
        assert result is ctx

    def test_truncates_long_values(self):
        lm = LearningManager()
        long_val = "x" * 500
        ctx = {"big_key": long_val}
        lm.record_execution("tool.post_dispatch", ctx)

        entry = lm._execution_log[0]
        assert len(entry["context"]["big_key"]) == 200

    def test_caps_at_100_entries(self):
        lm = LearningManager()
        for i in range(150):
            lm.record_execution("tool.post_dispatch", {"tool": f"tool_{i}"})

        assert len(lm._execution_log) == 100
        # Should keep the latest entries
        assert lm._execution_log[0]["context"]["tool"] == "tool_50"

    def test_records_multiple_event_types(self):
        lm = LearningManager()
        lm.record_execution("tool.post_dispatch", {"tool": "file_read"})
        lm.record_execution("tool.error", {"tool": "file_write"})
        lm.record_execution("goal.created", {"goal_id": "g1"})

        assert len(lm._execution_log) == 3
        assert lm._execution_log[0]["event"] == "tool.post_dispatch"
        assert lm._execution_log[1]["event"] == "tool.error"
        assert lm._execution_log[2]["event"] == "goal.created"


class TestAnalyzeExecution:
    """Tests for analyze_execution."""

    def test_empty_log(self):
        lm = LearningManager()
        result = lm.analyze_execution()
        assert result["patterns"] == []
        assert result["summary"] == "No execution data available"

    def test_tool_usage_pattern(self):
        lm = LearningManager()
        # Add tool call entries
        for _ in range(5):
            lm.record_execution("tool.post_dispatch", {"tool": "file_read"})
        for _ in range(2):
            lm.record_execution("tool.post_dispatch", {"tool": "web_search"})

        result = lm.analyze_execution()
        assert len(result["patterns"]) >= 1

        freq_pattern = result["patterns"][0]
        assert freq_pattern["type"] == "frequent_tool"
        assert freq_pattern["tool"] == "file_read"
        assert freq_pattern["count"] == 5
        assert result["event_counts"]["tool.post_dispatch"] == 7
        assert result["tool_usage"]["file_read"] == 5

    def test_error_pattern(self):
        lm = LearningManager()
        for _ in range(3):
            lm.record_execution("tool.error", {"tool": "bad_tool"})
        for _ in range(1):
            lm.record_execution("tool.error", {"tool": "other_tool"})

        result = lm.analyze_execution()
        error_patterns = [p for p in result["patterns"] if p["type"] == "error_prone_tool"]
        assert len(error_patterns) == 1
        assert error_patterns[0]["tool"] == "bad_tool"
        assert error_patterns[0]["error_count"] == 3

    def test_mixed_events(self):
        lm = LearningManager()
        lm.record_execution("tool.post_dispatch", {"tool": "file_read"})
        lm.record_execution("tool.error", {"tool": "file_write"})
        lm.record_execution("goal.created", {"goal_id": "g1"})

        result = lm.analyze_execution()
        assert "tool_usage" in result
        assert result["tool_usage"]["file_read"] == 1
        assert "event_counts" in result
        assert "Analyzed 3 events" in result["summary"]

    def test_session_and_goal_params_ignored(self):
        """session_id and goal_id are accepted but not filtered on (future use)."""
        lm = LearningManager()
        lm.record_execution("tool.post_dispatch", {"tool": "file_read"})

        result = lm.analyze_execution(session_id="s1", goal_id="g1")
        assert len(result["patterns"]) == 1


class TestExtractPatterns:
    """Tests for extract_patterns."""

    def test_delegates_to_analyze(self):
        lm = LearningManager()
        for _ in range(3):
            lm.record_execution("tool.post_dispatch", {"tool": "file_read"})

        patterns = lm.extract_patterns()
        assert len(patterns) == 1
        assert patterns[0]["type"] == "frequent_tool"

    def test_empty_returns_empty(self):
        lm = LearningManager()
        assert lm.extract_patterns() == []


class TestSuggestImprovements:
    """Tests for suggest_improvements."""

    def test_error_prone_tool_suggestion(self):
        lm = LearningManager()
        analysis = {
            "patterns": [
                {
                    "type": "error_prone_tool",
                    "tool": "bad_tool",
                    "error_count": 5,
                    "suggestion": "bad_tool has 5 errors",
                }
            ]
        }
        suggestions = lm.suggest_improvements(analysis)
        assert len(suggestions) == 1
        assert suggestions[0]["type"] == "tool_replacement"
        assert suggestions[0]["priority"] == "high"
        assert "bad_tool" in suggestions[0]["description"]

    def test_error_prone_low_count(self):
        lm = LearningManager()
        analysis = {
            "patterns": [
                {
                    "type": "error_prone_tool",
                    "tool": "meh_tool",
                    "error_count": 2,
                    "suggestion": "meh_tool has 2 errors",
                }
            ]
        }
        suggestions = lm.suggest_improvements(analysis)
        assert len(suggestions) == 1
        assert suggestions[0]["priority"] == "medium"

    def test_frequent_tool_above_threshold(self):
        lm = LearningManager()
        analysis = {
            "patterns": [
                {
                    "type": "frequent_tool",
                    "tool": "file_read",
                    "count": 15,
                    "suggestion": "file_read used 15 times",
                }
            ]
        }
        suggestions = lm.suggest_improvements(analysis)
        assert len(suggestions) == 1
        assert suggestions[0]["type"] == "automation"
        assert suggestions[0]["priority"] == "medium"

    def test_frequent_tool_below_threshold(self):
        lm = LearningManager()
        analysis = {
            "patterns": [
                {
                    "type": "frequent_tool",
                    "tool": "file_read",
                    "count": 5,
                    "suggestion": "file_read used 5 times",
                }
            ]
        }
        suggestions = lm.suggest_improvements(analysis)
        # count=5 is below the >10 threshold, so no suggestion
        assert len(suggestions) == 0

    def test_no_patterns(self):
        lm = LearningManager()
        suggestions = lm.suggest_improvements({"patterns": []})
        assert suggestions == []


class TestOnGoalCompleted:
    """Tests for on_goal_completed hook handler."""

    def test_returns_context(self):
        lm = LearningManager()
        ctx = {"goal_id": "g1", "session_id": "s1"}
        result = lm.on_goal_completed(ctx)
        assert result is ctx

    def test_emits_learning_analysis_when_patterns(self):
        hooks = HookManager()
        lm = LearningManager(hooks=hooks)

        # Add some data so analysis finds patterns
        for _ in range(3):
            lm.record_execution("tool.post_dispatch", {"tool": "file_read"})

        received = []
        hooks.register(HookEvent.LEARNING_ANALYSIS, lambda ctx: (received.append(ctx), ctx)[1])

        lm.on_goal_completed({"goal_id": "g1", "session_id": "s1"})

        assert len(received) == 1
        assert received[0]["goal_id"] == "g1"
        assert received[0]["patterns"] == 1

    def test_no_emit_when_no_patterns(self):
        hooks = HookManager()
        lm = LearningManager(hooks=hooks)

        received = []
        hooks.register(HookEvent.LEARNING_ANALYSIS, lambda ctx: (received.append(ctx), ctx)[1])

        lm.on_goal_completed({"goal_id": "g1"})
        assert len(received) == 0

    def test_stores_in_memory(self):
        class FakeMemory:
            def __init__(self):
                self.stored = []

            def store(self, session_id, key, content, category):
                self.stored.append({"session_id": session_id, "key": key, "content": content, "category": category})

        mem = FakeMemory()
        hooks = HookManager()
        lm = LearningManager(hooks=hooks, memory=mem)

        # Add error data to produce a suggestion
        for _ in range(5):
            lm.record_execution("tool.error", {"tool": "bad_tool"})

        lm.on_goal_completed({"goal_id": "g1", "session_id": "s1"})

        assert len(mem.stored) == 1
        assert mem.stored[0]["session_id"] == "s1"
        stored_content = json.loads(mem.stored[0]["content"])
        assert len(stored_content) >= 1
        assert stored_content[0]["type"] == "tool_replacement"

    def test_memory_error_is_swallowed(self):
        class BrokenMemory:
            def store(self, **kwargs):
                raise RuntimeError("disk full")

        hooks = HookManager()
        lm = LearningManager(hooks=hooks, memory=BrokenMemory())

        for _ in range(3):
            lm.record_execution("tool.error", {"tool": "bad_tool"})

        # Should not raise
        result = lm.on_goal_completed({"goal_id": "g1"})
        assert result["goal_id"] == "g1"
