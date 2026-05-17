from sophia.swarm.models import SwarmDecision
from sophia.swarm.orchestrator import FilteredToolRegistry, SwarmOrchestrator


class DummyRegistry:
    def get_schemas(self):
        return [
            {"type": "function", "function": {"name": "literature_search"}},
            {"type": "function", "function": {"name": "data_load"}},
        ]

    def dispatch(self, name, args):
        return f"{name}:{args}"

    def list_tools(self):
        return ["literature_search", "data_load"]


def test_no_swarm_decision_calls_parent_once():
    calls = []
    orch = SwarmOrchestrator(lambda prompt, tools=None: (calls.append((prompt, tools)), "single")[1])
    assert orch.execute(SwarmDecision(False), "hello") == "single"
    assert calls == [("hello", None)]


def test_swarm_execution_creates_record_and_results():
    orch = SwarmOrchestrator(lambda prompt, tools=None: f"done:{prompt[:5]}")
    text = orch.execute(SwarmDecision(True, recommended_roles=["writer", "critic"]), "complex writing task")
    records = orch.list_executions()
    assert "writer" in text or "critic" in text
    assert records[0].status == "completed"
    assert len(records[0].results) >= 2


def test_bus_context_reaches_later_pipeline_stage():
    prompts = []
    orch = SwarmOrchestrator(lambda prompt, tools=None: (prompts.append(prompt), "out")[1])
    orch.execute(SwarmDecision(True, recommended_roles=["writer", "reviewer"], workflow="pipeline"), "write a paper")
    assert any("蜂群通信总线" in prompt for prompt in prompts[1:])


def test_filtered_registry_empty_list_means_no_tools():
    registry = FilteredToolRegistry(DummyRegistry(), [])
    assert registry.get_schemas() == []
    assert registry.list_tools() == []
    assert "not allowed" in registry.dispatch("literature_search", {})


def test_filtered_registry_none_means_all_tools():
    registry = FilteredToolRegistry(DummyRegistry(), None)
    assert len(registry.get_schemas()) == 2
    assert registry.list_tools() == ["literature_search", "data_load"]
