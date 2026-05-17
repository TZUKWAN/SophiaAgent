import json
from pathlib import Path

from sophia import cli


class DummyTools:
    def get_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "dummy_tool",
                    "description": "A dummy tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def dispatch(self, name, args):
        return json.dumps({"name": name, "args": args}, ensure_ascii=False)


class DummyAgent:
    def __init__(self):
        self.tools = DummyTools()

    def run(self, prompt):
        return f"answered: {prompt}"


def test_mcp_tools_include_sophia_ask_first():
    tools = cli._mcp_tools_for_agent(DummyAgent())

    assert tools[0]["name"] == "sophia_ask"
    assert tools[0]["inputSchema"]["required"] == ["prompt"]
    assert tools[1]["name"] == "dummy_tool"


def test_mcp_sophia_ask_calls_agent_run():
    result = cli._call_mcp_tool(DummyAgent(), "sophia_ask", {"prompt": "写综述"})

    assert result == {"response": "answered: 写综述", "tool": "sophia_ask"}


def test_mcp_sophia_ask_rejects_empty_prompt():
    result = cli._call_mcp_tool(DummyAgent(), "sophia_ask", {"prompt": " "})

    assert "error" in result


def test_mcp_prompt_sophia_mentions_sophia_ask():
    messages = cli._mcp_prompt_messages("sophia", {"request": "分析论文"})

    assert messages[0]["role"] == "user"
    assert "sophia_ask" in messages[0]["content"]
    assert "分析论文" in messages[0]["content"]


def test_default_workspace_override_uses_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert Path(cli._default_workspace_override()) == tmp_path
