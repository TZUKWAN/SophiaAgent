"""Tests for ToolGuardrails: rate limiting and loop detection."""
import time
from unittest.mock import patch

import pytest

from sophia.guardrails import ToolGuardrails
from sophia.hooks import HookEvent, HookManager


@pytest.fixture
def guardrails():
    """Create guardrails with low limits for testing."""
    return ToolGuardrails(max_consecutive_calls=3, max_calls_per_minute=10)


@pytest.fixture
def guardrails_with_hooks():
    """Create guardrails with hooks attached."""
    hooks = HookManager()
    g = ToolGuardrails(hooks=hooks, max_consecutive_calls=3, max_calls_per_minute=10)
    return g, hooks


class TestToolGuardrails:
    def test_allow_normal_call(self, guardrails):
        """Normal tool call is allowed."""
        allowed, reason = guardrails.check_allowed("file_read", {"path": "test.txt"})
        assert allowed is True
        assert reason == ""

    def test_allow_up_to_consecutive_limit(self, guardrails):
        """Calls up to the consecutive limit are allowed."""
        for i in range(3):
            guardrails.record_call("file_read", {"path": f"file{i}.txt"})

        # The 3rd call was recorded, so consecutive_count = 3.
        # The next check_allowed should block because _consecutive_count (3) >= max (3).
        allowed, reason = guardrails.check_allowed("file_read", {"path": "file.txt"})
        assert allowed is False
        assert "consecutively" in reason.lower() or "consecutive" in reason.lower()

    def test_block_consecutive_exceeding_limit(self, guardrails):
        """Consecutive calls exceeding the limit are blocked."""
        # Record max allowed consecutive calls
        for i in range(3):
            guardrails.record_call("file_read", {"path": f"f{i}.txt"})

        # Next call to the same tool should be blocked
        allowed, reason = guardrails.check_allowed("file_read", {"path": "f.txt"})
        assert allowed is False
        assert "3" in reason  # mentions the count

    def test_allow_different_tools_resets_consecutive(self, guardrails):
        """Switching to a different tool resets the consecutive counter."""
        # Make 3 consecutive calls to file_read
        for i in range(3):
            guardrails.record_call("file_read", {"path": f"f{i}.txt"})

        # Calling a different tool should be allowed
        allowed, reason = guardrails.check_allowed("file_write", {"path": "out.txt"})
        assert allowed is True

    def test_block_rate_exceeding_limit(self, guardrails):
        """Total calls exceeding per-minute limit are blocked."""
        # Fill up to the rate limit
        for i in range(10):
            guardrails.record_call(f"tool_{i % 3}", {"arg": i})

        # Next call should be blocked due to rate limit
        allowed, reason = guardrails.check_allowed("new_tool", {})
        assert allowed is False
        assert "rate" in reason.lower() or "limit" in reason.lower()

    def test_record_call_updates_tracking(self, guardrails):
        """record_call updates consecutive and history tracking."""
        guardrails.record_call("file_read", {"path": "a.txt"})
        assert guardrails._last_tool == "file_read"
        assert guardrails._consecutive_count == 1
        assert len(guardrails._call_history) == 1

        guardrails.record_call("file_read", {"path": "b.txt"})
        assert guardrails._consecutive_count == 2
        assert len(guardrails._call_history) == 2

        guardrails.record_call("file_write", {"path": "c.txt"})
        assert guardrails._last_tool == "file_write"
        assert guardrails._consecutive_count == 1
        assert len(guardrails._call_history) == 3

    def test_check_hook_allows(self, guardrails):
        """check_hook allows a normal call and records it."""
        ctx = {"tool": "file_read", "args": {"path": "test.txt"}}
        result = guardrails.check_hook(ctx)
        assert "blocked" not in result or result.get("blocked") is not True
        # Call should be recorded
        assert len(guardrails._call_history) == 1

    def test_check_hook_blocks_consecutive(self, guardrails):
        """check_hook blocks when consecutive limit exceeded."""
        for i in range(3):
            guardrails.record_call("file_read", {"path": f"f{i}.txt"})

        ctx = {"tool": "file_read", "args": {"path": "f.txt"}}
        result = guardrails.check_hook(ctx)
        assert result["blocked"] is True
        assert "block_reason" in result

    def test_check_hook_blocks_rate(self, guardrails):
        """check_hook blocks when rate limit exceeded."""
        for i in range(10):
            guardrails.record_call(f"tool_{i}", {"i": i})

        ctx = {"tool": "another_tool", "args": {}}
        result = guardrails.check_hook(ctx)
        assert result["blocked"] is True

    def test_check_hook_emits_guardrail_block(self, guardrails_with_hooks):
        """check_hook emits GUARDRAIL_BLOCK event when blocking."""
        guardrails, hooks = guardrails_with_hooks

        events = []
        hooks.register(
            HookEvent.GUARDRAIL_BLOCK,
            lambda ctx: (events.append(ctx), ctx)[1],
        )

        # Exhaust consecutive limit
        for i in range(3):
            guardrails.record_call("file_read", {"path": f"f{i}.txt"})

        ctx = {"tool": "file_read", "args": {"path": "blocked.txt"}}
        guardrails.check_hook(ctx)

        assert len(events) == 1
        assert events[0]["tool"] == "file_read"
        assert "reason" in events[0]

    def test_reset_clears_state(self, guardrails):
        """Reset clears all tracking state."""
        guardrails.record_call("file_read", {"path": "a.txt"})
        guardrails.record_call("file_read", {"path": "b.txt"})
        guardrails.record_call("file_write", {"path": "c.txt"})

        assert len(guardrails._call_history) == 3
        assert guardrails._consecutive_count > 0

        guardrails.reset()

        assert len(guardrails._call_history) == 0
        assert guardrails._last_tool is None
        assert guardrails._consecutive_count == 0

    def test_after_reset_allows_again(self, guardrails):
        """After reset, previously blocked calls are allowed."""
        # Exhaust consecutive limit
        for i in range(3):
            guardrails.record_call("file_read", {"path": f"f{i}.txt"})

        allowed, _ = guardrails.check_allowed("file_read", {})
        assert allowed is False

        guardrails.reset()

        allowed, reason = guardrails.check_allowed("file_read", {})
        assert allowed is True
        assert reason == ""

    def test_cleanup_old_history(self, guardrails):
        """Old history entries (>60s) are cleaned up."""
        # Add an old entry manually
        guardrails._call_history.append({
            "tool": "old_tool",
            "timestamp": time.time() - 120,  # 2 minutes ago
            "args": {},
        })
        guardrails._call_history.append({
            "tool": "recent_tool",
            "timestamp": time.time() - 10,  # 10 seconds ago
            "args": {},
        })

        assert len(guardrails._call_history) == 2
        guardrails._cleanup_old_history()
        assert len(guardrails._call_history) == 1
        assert guardrails._call_history[0]["tool"] == "recent_tool"

    def test_different_tools_dont_affect_consecutive(self, guardrails):
        """Interleaving different tools resets consecutive counter."""
        guardrails.record_call("tool_a", {})
        guardrails.record_call("tool_a", {})
        guardrails.record_call("tool_b", {})  # resets consecutive
        guardrails.record_call("tool_a", {})

        # tool_a was called, then tool_b reset, then tool_a once more
        # So tool_a consecutive is 1, which is fine
        allowed, reason = guardrails.check_allowed("tool_a", {})
        assert allowed is True

    @patch("sophia.guardrails.time.time")
    def test_rate_window_is_60_seconds(self, mock_time, guardrails):
        """Rate limit window is exactly 60 seconds."""
        base_time = 1000.0
        mock_time.return_value = base_time

        # Add 10 calls at base_time
        for i in range(10):
            guardrails._call_history.append({
                "tool": f"tool_{i}",
                "timestamp": base_time,
                "args": {},
            })

        # At base_time, all 10 calls are in window
        guardrails._cleanup_old_history()
        assert len(guardrails._call_history) == 10

        # At base_time + 61, all calls are old
        mock_time.return_value = base_time + 61
        guardrails._cleanup_old_history()
        assert len(guardrails._call_history) == 0
