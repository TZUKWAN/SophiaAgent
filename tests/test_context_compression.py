"""Tests for token-aware context compression."""

import pytest

from sophia.context import ContextCompressor


class TestTokenEstimation:
    def test_estimate_empty(self):
        comp = ContextCompressor()
        assert comp.estimate_tokens([]) == 0

    def test_estimate_simple_message(self):
        comp = ContextCompressor()
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = comp.estimate_tokens(messages)
        assert tokens > 0
        # "Hello world" ≈ 2-3 tokens with tiktoken, ~3 with heuristic
        assert tokens < 50

    def test_estimate_cjk(self):
        comp = ContextCompressor()
        messages = [{"role": "user", "content": "这是一个中文测试"}]
        tokens = comp.estimate_tokens(messages)
        assert tokens > 0
        # 8 CJK chars ≈ 4 tokens (heuristic: 0.5/char)

    def test_estimate_tool_call_overhead(self):
        comp = ContextCompressor()
        messages = [
            {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1"}]},
        ]
        tokens = comp.estimate_tokens(messages)
        assert tokens >= 200  # overhead for tool_calls block


class TestCompressionTrigger:
    def test_no_compression_under_threshold(self):
        comp = ContextCompressor(max_tokens=1000, trigger_ratio=0.8)
        messages = [{"role": "user", "content": "short"}]
        result = comp.maybe_compress(messages)
        assert result == messages  # unchanged

    def test_compression_triggered_over_threshold(self):
        comp = ContextCompressor(max_tokens=1000, trigger_ratio=0.5, keep_recent=1)
        # Generate enough text to exceed 500 tokens
        long_text = "word " * 300  # ~300 tokens
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        result = comp.maybe_compress(messages)
        # Should compress older messages into a summary
        assert len(result) < len(messages)
        assert any(m.get("role") == "system" and "Summary" in m.get("content", "") for m in result)

    def test_preserves_recent_messages(self):
        comp = ContextCompressor(max_tokens=1000, trigger_ratio=0.5, keep_recent=2)
        long_text = "word " * 300
        messages = [
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "msg4"},
            {"role": "user", "content": "msg5"},
            {"role": "assistant", "content": "msg6"},
            {"role": "user", "content": "msg7"},
            {"role": "assistant", "content": "msg8"},
        ]
        result = comp.maybe_compress(messages)
        # Last 4 messages (keep_recent=2 pairs) should be preserved
        contents = [m["content"] for m in result]
        assert "msg7" in contents
        assert "msg8" in contents

    def test_summarize_extracts_research_question(self):
        comp = ContextCompressor()
        messages = [
            {"role": "user", "content": "我想研究政策对就业的影响"},
            {"role": "assistant", "content": "好的，我们来分析"},
        ]
        summary = comp._summarize(messages)
        assert "research question" in summary or "问题" in summary

    def test_summarize_extracts_tool_results(self):
        comp = ContextCompressor()
        messages = [
            {"role": "tool", "content": '{"result_id": "res_abc", "apa": "t(98)=3.24, p=.002"}'},
        ]
        summary = comp._summarize(messages)
        assert "res_abc" in summary
        assert "t(98)=3.24" in summary

    def test_64k_default_settings(self):
        comp = ContextCompressor()
        assert comp.max_tokens == 64000
        assert comp.trigger_tokens == 41600  # 65% of 64K
