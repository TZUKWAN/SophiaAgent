"""Tests for SubAgent delegation system."""
import json
from unittest.mock import MagicMock

from sophia.hooks import HookManager
from sophia.subagent import SubAgentManager, register_subagent_tools
from sophia.tools.registry import ToolRegistry


def _make_manager(tmp_path, run_fn=None, hooks=None):
    db_path = str(tmp_path / "test.db")
    if run_fn is None:
        run_fn = lambda prompt, tools=None: f"result for: {prompt[:30]}"
    if hooks is None:
        hooks = HookManager()
    return SubAgentManager(run_fn, hooks, db_path)


class TestSubAgentManager:
    def test_delegate_creates_task(self, tmp_path):
        mgr = _make_manager(tmp_path)
        task = mgr.delegate("sess1", "search literature about AI")
        assert task.id
        assert task.session_id == "sess1"
        assert task.prompt == "search literature about AI"
        assert task.status == "pending"

    def test_execute_task(self, tmp_path):
        called_prompts = []
        run_fn = lambda p, t=None: (called_prompts.append(p), f"done: {p[:20]}")[1]
        mgr = _make_manager(tmp_path, run_fn=run_fn)
        task = mgr.delegate("sess1", "test prompt")
        result = mgr.execute(task)
        assert result.status == "completed"
        assert "done:" in result.result
        assert len(called_prompts) == 1

    def test_execute_with_tool_restriction(self, tmp_path):
        received_tools = []
        run_fn = lambda p, t=None: (received_tools.append(t), "ok")[1]
        mgr = _make_manager(tmp_path, run_fn=run_fn)
        task = mgr.delegate("sess1", "prompt", tools=["file_read", "web_search"])
        mgr.execute(task)
        assert received_tools[-1] == ["file_read", "web_search"]

    def test_execute_handles_error(self, tmp_path):
        def failing_fn(prompt, tools=None):
            raise ValueError("something went wrong")
        mgr = _make_manager(tmp_path, run_fn=failing_fn)
        task = mgr.delegate("sess1", "will fail")
        result = mgr.execute(task)
        assert result.status == "failed"
        assert "something went wrong" in result.error

    def test_execute_parallel(self, tmp_path):
        call_count = []
        run_fn = lambda p, t=None: (call_count.append(1), f"result-{len(call_count)}")[1]
        mgr = _make_manager(tmp_path, run_fn=run_fn)
        tasks = [mgr.delegate("sess1", f"task-{i}") for i in range(3)]
        results = mgr.execute_parallel(tasks)
        assert len(results) == 3
        assert all(r.status == "completed" for r in results)
        assert len(call_count) == 3

    def test_delegate_batch(self, tmp_path):
        mgr = _make_manager(tmp_path)
        results = mgr.delegate_batch("sess1", [
            {"prompt": "task 1"},
            {"prompt": "task 2"},
        ])
        assert len(results) == 2
        assert all(r.status == "completed" for r in results)

    def test_get_task(self, tmp_path):
        mgr = _make_manager(tmp_path)
        task = mgr.delegate("sess1", "test")
        retrieved = mgr.get_task(task.id)
        assert retrieved is not None
        assert retrieved.id == task.id

    def test_list_tasks(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.delegate("sess1", "task 1")
        mgr.delegate("sess1", "task 2")
        mgr.delegate("sess2", "other session")
        tasks = mgr.list_tasks("sess1")
        assert len(tasks) == 2

    def test_hook_emission(self, tmp_path):
        hooks = HookManager()
        events = []
        hooks.register("subagent.spawn", lambda ctx: (events.append("spawn"), ctx)[1])
        hooks.register("subagent.complete", lambda ctx: (events.append("complete"), ctx)[1])
        mgr = _make_manager(tmp_path, hooks=hooks)
        task = mgr.delegate("sess1", "test")
        assert "spawn" in events
        mgr.execute(task)
        assert "complete" in events


class TestSubAgentTools:
    def test_delegate_tool(self, tmp_path):
        mgr = _make_manager(tmp_path)
        reg = ToolRegistry()
        register_subagent_tools(reg, mgr)
        result = json.loads(reg.dispatch("subagent_delegate", {
            "session_id": "s1", "prompt": "test prompt",
        }))
        assert result["status"] == "completed"

    def test_list_tool(self, tmp_path):
        mgr = _make_manager(tmp_path)
        reg = ToolRegistry()
        register_subagent_tools(reg, mgr)
        reg.dispatch("subagent_delegate", {"session_id": "s1", "prompt": "p1"})
        result = json.loads(reg.dispatch("subagent_list", {"session_id": "s1"}))
        assert len(result) == 1

    def test_batch_tool(self, tmp_path):
        mgr = _make_manager(tmp_path)
        reg = ToolRegistry()
        register_subagent_tools(reg, mgr)
        result = json.loads(reg.dispatch("subagent_delegate_batch", {
            "session_id": "s1",
            "tasks": [{"prompt": "a"}, {"prompt": "b"}],
        }))
        assert len(result) == 2
