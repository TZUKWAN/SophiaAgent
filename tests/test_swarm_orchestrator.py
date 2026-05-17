from sophia.swarm.models import SwarmDecision
from sophia.swarm.orchestrator import SwarmOrchestrator


def test_no_swarm_decision_calls_parent_once():
    calls = []
    orch = SwarmOrchestrator(lambda prompt, tools=None: (calls.append((prompt, tools)), "single")[1])
    assert orch.execute(SwarmDecision(False), "你好") == "single"
    assert calls == [("你好", None)]


def test_swarm_execution_creates_record_and_results():
    orch = SwarmOrchestrator(lambda prompt, tools=None: f"done:{prompt[:5]}")
    text = orch.execute(SwarmDecision(True, recommended_roles=["writer", "critic"]), "复杂写作任务")
    records = orch.list_executions()
    assert "writer" in text or "critic" in text
    assert records[0].status == "completed"
    assert len(records[0].results) >= 2


def test_bus_context_reaches_later_pipeline_stage():
    prompts = []
    orch = SwarmOrchestrator(lambda prompt, tools=None: (prompts.append(prompt), "out")[1])
    orch.execute(SwarmDecision(True), "写一篇数字经济文献综述")
    assert any("蜂群通信总线" in prompt for prompt in prompts[1:])
