from sophia.agent import SophiaAgent
from sophia.providers.base import ProviderResponse
from sophia.swarm.models import SwarmDecision


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.responses = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        if self.responses:
            return ProviderResponse(content=self.responses.pop(0))
        return ProviderResponse(content="single-response")

    def chat_stream(self, messages, tools=None):
        self.calls.append((messages, tools))
        yield "single-response"
        return ProviderResponse(content="single-response")


def _captured_user_content(fake: FakeProvider) -> str:
    return "\n".join(
        message.get("content") or ""
        for messages, _ in fake.calls
        for message in messages
    )


def test_agent_initializes_swarm_and_tools(config):
    agent = SophiaAgent(config)
    assert agent.swarm_orchestrator is not None
    assert "swarm_delegate" in agent.tools.list_tools()
    assert "subagent_delegate" in agent.tools.list_tools()


def test_simple_agent_run_does_not_start_swarm(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake
    assert agent.run("hello") == "single-response"
    assert len(fake.calls) == 1


def test_complex_agent_run_starts_swarm_without_user_trigger(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake
    text = agent.run("帮我写一篇关于数字经济的文献综述，正文不少于8000字，要全面分析方法、引用和质量问题")
    assert "single-response" in text
    assert len(agent.swarm_orchestrator.list_executions()) == 1


def test_stream_complex_agent_emits_swarm_events(config):
    agent = SophiaAgent(config)
    agent.provider = FakeProvider()
    events = list(
        agent.run_stream("帮我写一篇关于数字经济的文献综述，正文不少于8000字，要全面分析方法、引用和质量问题")
    )
    types = [event["type"] for event in events]
    assert "swarm_analyze" in types
    assert types[-1] == "done"


def test_stream_simple_agent_does_not_emit_workspace_scan(config):
    agent = SophiaAgent(config)
    agent.provider = FakeProvider()

    events = list(agent.run_stream("hello", allow_swarm=False))

    assert "workspace_scan_done" not in [event["type"] for event in events]
    assert events[-1]["type"] == "done"


def test_workspace_request_injects_local_evidence(config, tmp_path):
    config.session.workspace = str(tmp_path)
    (tmp_path / "paper.md").write_text(
        "Authentic workspace literature about Chinese culture communication.",
        encoding="utf-8",
    )
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake

    agent.run("基于工作空间中的论文，仔细阅读后写理论论文，正文不少于8000字", allow_swarm=False)

    user_content = _captured_user_content(fake)
    assert "Authentic workspace literature about Chinese culture communication." in user_content
    assert "Workspace literature has been read" in user_content


def test_stream_workspace_request_emits_context_tool_card(config, tmp_path):
    config.session.workspace = str(tmp_path)
    (tmp_path / "paper.md").write_text(
        "Authentic workspace literature.",
        encoding="utf-8",
    )
    agent = SophiaAgent(config)
    agent.provider = FakeProvider()

    events = list(agent.run_stream("基于工作空间中的论文写理论论文，正文不少于8000字", allow_swarm=False))

    assert events[0]["type"] == "workspace_scan_start"
    assert any(event["type"] == "workspace_file_done" for event in events)
    assert any(event["type"] == "workspace_scan_done" for event in events)
    assert events[-1]["type"] == "done"
    assert "single-response" in events[-1]["response"]
    assert ".sophia" in events[-1]["response"]
    assert "generated_documents" in events[-1]["response"]


def test_paper_request_without_references_prompts_before_search(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake

    agent.run(
        "write a theoretical paper about generative AI and cultural communication, "
        "at least 8000 words",
        allow_swarm=False,
    )

    assert "Before independently searching for references" in _captured_user_content(fake)


def test_underspecified_paper_request_asks_for_requirements(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake

    response = agent.run("请写一篇生成式人工智能论文", allow_swarm=False)

    assert "确认" in response
    assert "论文类型" in response
    assert "目标正文字数" in response
    assert fake.calls == []


def test_stream_underspecified_paper_request_asks_for_requirements(config):
    agent = SophiaAgent(config)
    agent.provider = FakeProvider()

    events = list(agent.run_stream("请写一篇生成式人工智能论文", allow_swarm=False))

    assert events[-1]["type"] == "done"
    assert "论文类型" in events[-1]["response"]


def test_paper_request_with_references_prioritizes_user_sources(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake

    agent.run(
        "write a theoretical paper, at least 8000 words. "
        "References: Smith (2024). Generative AI and culture.",
        allow_swarm=False,
    )

    assert "The user has supplied references" in _captured_user_content(fake)


def test_short_paper_triggers_internal_repair_pass(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    fake.responses = [
        "# Paper\n\nshort draft\n\n## References\n1. Smith, J. (2024). Real paper. Journal.",
        "# Paper\n\nrepaired draft still short",
    ]
    agent.provider = fake

    response = agent.run(
        "write a theoretical paper about generative AI, at least 8000 words",
        allow_swarm=False,
    )

    assert "repaired draft still short" in response
    assert len(fake.calls) == 2
    assert "Sophia internal quality repair" in _captured_user_content(fake)


def test_empirical_request_injects_task_harness_and_preflight(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake

    agent.run(
        "做一个关于数字经济影响创新的实证回归，并完成稳健性和可信性检验",
        allow_swarm=False,
    )

    content = _captured_user_content(fake)
    assert "[Sophia task harness]" in content
    assert "Mandatory execution loop" in content
    assert "Empirical execution contract" in content
    assert "empirical_workflow_plan" in content
    assert "Credibility checks are mandatory" in content
    assert "[Empirical preflight plan generated by Sophia]" in content
    assert "data_path_or_result_id" in content


def test_swarm_stream_failure_falls_back_to_main_agent(config):
    agent = SophiaAgent(config)
    agent.provider = FakeProvider()

    def fail_stream(*args, **kwargs):
        raise RuntimeError("swarm broke")
        yield  # pragma: no cover

    agent.swarm_orchestrator.analyze = lambda message: SwarmDecision(True, recommended_roles=["writer"])
    agent.swarm_orchestrator.execute_stream = fail_stream
    events = list(
        agent.run_stream(
            "甯垜鍐欎竴绡囧叧浜庢暟瀛楃粡娴庣殑鏂囩尞缁艰堪锛岃鍏ㄩ潰鍒嗘瀽鏂规硶銆佸紩鐢ㄥ拰璐ㄩ噺闂"
        )
    )

    assert any(event["type"] == "swarm_error" for event in events)
    assert events[-1]["type"] == "done"
    assert "single-response" in events[-1]["response"]


def test_swarm_run_failure_falls_back_to_main_agent(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake

    def fail_execute(*args, **kwargs):
        raise RuntimeError("swarm broke")

    agent.swarm_orchestrator.analyze = lambda message: SwarmDecision(True, recommended_roles=["writer"])
    agent.swarm_orchestrator.execute = fail_execute
    text = agent.run(
        "甯垜鍐欎竴绡囧叧浜庢暟瀛楃粡娴庣殑鏂囩尞缁艰堪锛岃鍏ㄩ潰鍒嗘瀽鏂规硶銆佸紩鐢ㄥ拰璐ㄩ噺闂"
    )

    assert text == "single-response"
