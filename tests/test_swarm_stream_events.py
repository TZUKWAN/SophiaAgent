from sophia.swarm.models import SwarmDecision
from sophia.swarm.orchestrator import SwarmOrchestrator


def test_stream_events_for_swarm_path():
    orch = SwarmOrchestrator(lambda prompt, tools=None: "agent-result")
    events = list(orch.execute_stream(SwarmDecision(True, recommended_roles=["writer"]), "复杂写作任务"))
    types = [event["type"] for event in events]
    assert types[0] == "swarm_analyze"
    assert "swarm_plan" in types
    assert "swarm_done" in types
    assert any(event["type"] == "token" and "agent-result" in event["content"] for event in events)


def test_stream_events_for_skip_path():
    orch = SwarmOrchestrator(lambda prompt, tools=None: "single")
    events = list(orch.execute_stream(SwarmDecision(False, reason="simple"), "你好"))
    assert events[0]["type"] == "swarm_skip"
    assert events[1] == {"type": "token", "content": "single"}
