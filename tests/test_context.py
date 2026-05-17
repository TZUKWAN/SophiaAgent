"""Tests for Context Compression."""
from sophia.context import ContextCompressor
from sophia.hooks import HookManager


def _make_messages(n, start_role="user"):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}: " + "x" * 50})
    return msgs


class TestContextCompressor:
    def test_should_compress_false_small(self):
        comp = ContextCompressor(max_tokens=8000, trigger_ratio=0.8)
        msgs = _make_messages(10)
        assert comp.should_compress(msgs) is False

    def test_should_compress_true_large(self):
        comp = ContextCompressor(max_tokens=300, trigger_ratio=0.5)
        msgs = _make_messages(20)
        assert comp.should_compress(msgs) is True

    def test_compress_keeps_system(self):
        comp = ContextCompressor(max_tokens=500, trigger_ratio=0.5, keep_recent=3)
        msgs = _make_messages(20)
        result = comp.compress(msgs)
        assert result[0]["role"] == "system"
        assert "You are a helpful assistant" in result[0]["content"]

    def test_compress_keeps_recent(self):
        comp = ContextCompressor(max_tokens=500, trigger_ratio=0.5, keep_recent=5)
        msgs = _make_messages(30)
        result = comp.compress(msgs)
        # Last 5 non-system messages should be preserved
        recent = [m for m in result if m["role"] != "system"]
        assert len(recent) >= 5

    def test_compress_adds_summary(self):
        comp = ContextCompressor(max_tokens=500, trigger_ratio=0.5, keep_recent=3)
        msgs = _make_messages(30)
        result = comp.compress(msgs)
        summaries = [m for m in result if "Summary" in m.get("content", "")]
        assert len(summaries) == 1

    def test_compress_reduces_count(self):
        comp = ContextCompressor(max_tokens=500, trigger_ratio=0.5, keep_recent=3)
        msgs = _make_messages(30)
        result = comp.compress(msgs)
        assert len(result) < len(msgs)

    def test_no_compress_when_below_threshold(self):
        comp = ContextCompressor(max_tokens=8000, trigger_ratio=0.8)
        msgs = _make_messages(5)
        result = comp.compress(msgs)
        assert len(result) == len(msgs)

    def test_on_pre_run_hook(self):
        comp = ContextCompressor(max_tokens=500, trigger_ratio=0.5, keep_recent=3)
        msgs = _make_messages(30)
        ctx = {"messages": msgs}
        result = comp.on_pre_run(ctx)
        assert len(result["messages"]) < len(msgs)

    def test_on_pre_run_no_compress_needed(self):
        comp = ContextCompressor(max_tokens=8000)
        msgs = _make_messages(5)
        ctx = {"messages": msgs}
        result = comp.on_pre_run(ctx)
        assert len(result["messages"]) == len(msgs)

    def test_count_tokens_approx(self):
        comp = ContextCompressor()
        msgs = [{"role": "user", "content": "Hello world"}]
        count = comp.estimate_tokens(msgs)
        assert count > 0

    def test_empty_messages(self):
        comp = ContextCompressor()
        assert comp.should_compress([]) is False
        assert comp.compress([]) == []
