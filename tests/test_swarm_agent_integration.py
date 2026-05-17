from sophia.agent import SophiaAgent
from sophia.providers.base import ProviderResponse


class FakeProvider:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
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
    text = agent.run("帮我写一篇关于数字经济的文献综述，要全面分析方法、引用和质量问题")
    assert "single-response" in text
    assert len(agent.swarm_orchestrator.list_executions()) == 1


def test_stream_complex_agent_emits_swarm_events(config):
    agent = SophiaAgent(config)
    agent.provider = FakeProvider()
    events = list(
        agent.run_stream("帮我写一篇关于数字经济的文献综述，要全面分析方法、引用和质量问题")
    )
    types = [event["type"] for event in events]
    assert "swarm_analyze" in types
    assert types[-1] == "done"


def test_workspace_request_injects_local_evidence(config, tmp_path):
    config.session.workspace = str(tmp_path)
    (tmp_path / "paper.md").write_text(
        "Authentic workspace literature about Chinese culture communication.",
        encoding="utf-8",
    )
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake

    agent.run("基于工作空间中的论文，仔细阅读后写论文", allow_swarm=False)

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

    events = list(agent.run_stream("基于工作空间中的论文写论文", allow_swarm=False))

    assert events[0]["type"] == "tool_call"
    assert events[0]["name"] == "workspace_context_read"
    assert events[-1]["type"] == "done"
    assert "single-response" in events[-1]["response"]
    assert ".sophia" in events[-1]["response"]
    assert "generated_documents" in events[-1]["response"]


def test_paper_request_without_references_prompts_before_search(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake

    agent.run("write a paper about generative AI and cultural communication", allow_swarm=False)

    assert "Before independently searching for references" in _captured_user_content(fake)


def test_paper_request_with_references_prioritizes_user_sources(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake

    agent.run(
        "write a paper. References: Smith (2024). Generative AI and culture.",
        allow_swarm=False,
    )

    assert "The user has supplied references" in _captured_user_content(fake)
