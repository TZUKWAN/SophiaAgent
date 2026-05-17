"""Tests for Autopilot orchestration layer."""

import pytest

from sophia.autopilot import AutopilotRouter, ExecutionMonitor, AutopilotOrchestrator


class TestAutopilotRouter:
    def test_detects_research_intent_chinese(self):
        assert AutopilotRouter.is_research_intent("我想研究最低工资对就业的影响")
        assert AutopilotRouter.is_research_intent("做一个回归分析")
        assert AutopilotRouter.is_research_intent("检验两组差异是否显著")

    def test_detects_research_intent_english(self):
        assert AutopilotRouter.is_research_intent("Run a regression analysis")
        assert AutopilotRouter.is_research_intent("What is the causal effect of")
        assert AutopilotRouter.is_research_intent("Compare treatment and control")

    def test_rejects_non_research_intent(self):
        assert not AutopilotRouter.is_research_intent("Hello")
        assert not AutopilotRouter.is_research_intent("What's the weather today")
        assert not AutopilotRouter.is_research_intent("打开文件")

    def test_augment_messages_adds_hint(self):
        messages = [{"role": "user", "content": "研究政策效应"}]
        result = AutopilotRouter.augment_messages(messages, "研究政策效应")
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "methodology_advise" in result[0]["content"]

    def test_augment_messages_no_hint_for_casual(self):
        messages = [{"role": "user", "content": "Hello there"}]
        result = AutopilotRouter.augment_messages(messages, "Hello there")
        assert len(result) == 1  # no extra system message

    def test_detects_repetitive_intent(self):
        assert AutopilotRouter.is_repetitive_intent("每天帮我跑一遍")
        assert AutopilotRouter.is_repetitive_intent("能不能记住这个流程")
        assert not AutopilotRouter.is_repetitive_intent("分析一下数据")


class TestExecutionMonitor:
    def test_detects_repeated_sequence(self):
        monitor = ExecutionMonitor()
        # Simulate 3 repetitions of load -> did -> plot
        for _ in range(3):
            monitor.on_tool_post_dispatch({"tool": "research_load_data", "timestamp": 1})
            monitor.on_tool_post_dispatch({"tool": "research_did", "timestamp": 2})
            monitor.on_tool_post_dispatch({"tool": "research_plot", "timestamp": 3})

        assert len(monitor._tool_sequence) == 9
        # The last 3 tools form a repeated pattern
        assert monitor._tool_sequence[-3]["tool"] == "research_load_data"

    def test_no_discovery_with_short_sequence(self):
        monitor = ExecutionMonitor()
        monitor.on_tool_post_dispatch({"tool": "a", "timestamp": 1})
        monitor.on_tool_post_dispatch({"tool": "b", "timestamp": 2})
        # Only 2 tools, not enough for pattern detection
        assert len(monitor._tool_sequence) == 2


class TestAutopilotOrchestrator:
    def test_before_run_augment(self):
        class FakeAgent:
            pass

        orch = AutopilotOrchestrator(FakeAgent())
        messages = [{"role": "user", "content": "研究最低工资影响"}]
        result = orch.before_run("研究最低工资影响", messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
