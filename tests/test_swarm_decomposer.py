from sophia.swarm.decomposer import TaskDecomposer
from sophia.swarm.models import SwarmDecision


def test_literature_review_rule_plan_has_pipeline_roles():
    plan = TaskDecomposer(use_llm=False).decompose("写一篇数字经济文献综述", SwarmDecision(True))
    roles = [agent.role_id for stage in plan.stages for agent in stage.agents]
    assert plan.workflow == "pipeline"
    assert "literature_searcher" in roles
    assert "reviewer" in roles


def test_pure_roles_do_not_receive_tools():
    plan = TaskDecomposer(use_llm=False).decompose("全面对比两种政策并批判逻辑漏洞", SwarmDecision(True))
    for stage in plan.stages:
        for agent in stage.agents:
            if agent.role_id in {"critic", "synthesizer"}:
                assert agent.tools == []


def test_llm_plan_skips_unknown_roles_and_adds_dependencies():
    llm = lambda prompt: '{"workflow":"pipeline","stages":[{"stage_id":"one","parallel":true,"agents":[{"role_id":"writer","task_prompt":"写"}]},{"stage_id":"two","parallel":false,"agents":[{"role_id":"unknown","task_prompt":"x"},{"role_id":"reviewer","task_prompt":"审"}]}],"coordinator_prompt":"合并"}'
    plan = TaskDecomposer(llm_call=llm).decompose("制定材料", SwarmDecision(True))
    assert len(plan.stages) == 2
    assert plan.stages[1].depends_on == ["one"]


def test_fallback_uses_recommended_roles_and_reviewer():
    decision = SwarmDecision(True, recommended_roles=["writer"])
    plan = TaskDecomposer(use_llm=False).decompose("复杂任务但无规则命中", decision)
    roles = [agent.role_id for stage in plan.stages for agent in stage.agents]
    assert roles == ["writer", "reviewer"]
