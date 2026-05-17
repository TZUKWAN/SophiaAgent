from sophia.agent import SophiaAgent
from sophia.providers.base import ProviderResponse


class FakeProvider:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        return ProviderResponse(content="single-response")


def test_agent_initializes_swarm_and_tools(config):
    agent = SophiaAgent(config)
    assert agent.swarm_orchestrator is not None
    assert "swarm_delegate" in agent.tools.list_tools()
    assert "subagent_delegate" in agent.tools.list_tools()


def test_simple_agent_run_does_not_start_swarm(config):
    agent = SophiaAgent(config)
    fake = FakeProvider()
    agent.provider = fake
    assert agent.run("你好") == "single-response"
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
    events = list(agent.run_stream("帮我写一篇关于数字经济的文献综述，要全面分析方法、引用和质量问题"))
    types = [event["type"] for event in events]
    assert "swarm_analyze" in types
    assert types[-1] == "done"
