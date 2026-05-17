from sophia.swarm.analyzer import TaskAnalyzer


def test_simple_greeting_does_not_start_swarm():
    decision = TaskAnalyzer(use_llm=False).analyze("你好")
    assert decision.need_swarm is False


def test_complex_research_writing_starts_swarm():
    decision = TaskAnalyzer(use_llm=False).analyze("帮我写一篇关于数字经济的文献综述，要全面分析方法、引用和质量问题")
    assert decision.need_swarm is True
    assert "literature_searcher" in decision.recommended_roles
    assert decision.estimated_roles <= 5


def test_llm_json_decision_parsed_when_rules_inconclusive():
    analyzer = TaskAnalyzer(
        llm_call=lambda prompt: '{"need_swarm": true, "reason": "需要协作", "estimated_roles": 2, "recommended_roles": ["writer", "reviewer"], "workflow": "pipeline"}'
    )
    decision = analyzer.analyze("撰写研究备忘录")
    assert decision.need_swarm is True
    assert decision.workflow == "pipeline"


def test_llm_bad_json_falls_back_without_fake_success():
    analyzer = TaskAnalyzer(llm_call=lambda prompt: "not json")
    decision = analyzer.analyze("撰写研究备忘录")
    assert decision.need_swarm is False
    assert "失败" in decision.reason
