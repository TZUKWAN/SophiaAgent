import json

from sophia.hooks import HookManager
from sophia.subagent import SubAgentManager, register_subagent_tools
from sophia.tools.registry import ToolRegistry


def test_legacy_manager_uses_swarm_backend_and_keeps_shape(tmp_path):
    mgr = SubAgentManager(lambda prompt, tools=None: "ok", HookManager(), str(tmp_path / "db.sqlite"))
    task = mgr.delegate("s1", "p")
    result = mgr.execute(task)
    assert result.status == "completed"
    assert result.result == "ok"


def test_legacy_tools_keep_names(tmp_path):
    mgr = SubAgentManager(lambda prompt, tools=None: "ok", HookManager(), str(tmp_path / "db.sqlite"))
    registry = ToolRegistry()
    register_subagent_tools(registry, mgr)
    assert "subagent_delegate" in registry.list_tools()
    result = json.loads(registry.dispatch("subagent_delegate", {"session_id": "s1", "prompt": "p"}))
    assert result["status"] == "completed"
