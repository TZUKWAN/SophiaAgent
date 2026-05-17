from datetime import datetime, timedelta

from sophia.swarm.models import AgentResult, AgentSpec, Stage, SwarmDecision, SwarmExecutionRecord, SwarmPlan


def test_swarm_decision_to_dict():
    decision = SwarmDecision(True, "complex", 3, 0.8, ["writer"], "mixed")
    assert decision.to_dict()["recommended_roles"] == ["writer"]


def test_plan_agent_count_and_nested_dicts():
    spec = AgentSpec("a1", "writer", "write")
    plan = SwarmPlan(stages=[Stage("s1", agents=[spec])])
    assert plan.agent_count == 1
    assert plan.to_dict()["stages"][0]["agents"][0]["role_id"] == "writer"


def test_agent_result_duration_and_record_summary():
    start = datetime.now()
    result = AgentResult("a1", "writer", "completed", start_time=start, end_time=start + timedelta(seconds=1))
    record = SwarmExecutionRecord(
        "e1",
        "s1",
        SwarmDecision(True),
        SwarmPlan(stages=[Stage("s1", agents=[AgentSpec("a1", "writer", "x")])]),
        results=[result],
        status="completed",
    )
    assert result.duration_ms == 1000
    assert record.to_summary()["completed_agents"] == 1
