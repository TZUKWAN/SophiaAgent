from sophia.swarm.models import AgentResult, SwarmPlan
from sophia.swarm.synthesizer import ResultSynthesizer


def test_single_completed_result_returns_directly():
    result = AgentResult("a1", "writer", "completed", "final")
    assert ResultSynthesizer().synthesize([result], SwarmPlan()) == "final"


def test_failed_only_returns_failure_summary():
    result = AgentResult("a1", "writer", "failed", error="boom")
    text = ResultSynthesizer().synthesize([result], SwarmPlan())
    assert "boom" in text


def test_multiple_results_concat_without_llm():
    results = [AgentResult("a1", "writer", "completed", "draft"), AgentResult("a2", "reviewer", "completed", "review")]
    text = ResultSynthesizer().synthesize(results, SwarmPlan(coordinator_prompt="合并"))
    assert "writer" in text and "reviewer" in text


def test_llm_synthesis_used_and_falls_back_on_error():
    called = []
    synth = ResultSynthesizer(llm_call=lambda prompt: (called.append(prompt), "llm-final")[1])
    results = [AgentResult("a1", "writer", "completed", "draft"), AgentResult("a2", "reviewer", "completed", "review")]
    assert synth.synthesize(results, SwarmPlan(original_query="q")) == "llm-final"
    assert called
    fallback = ResultSynthesizer(llm_call=lambda prompt: (_ for _ in ()).throw(RuntimeError("bad")))
    assert "writer" in fallback.synthesize(results, SwarmPlan())
