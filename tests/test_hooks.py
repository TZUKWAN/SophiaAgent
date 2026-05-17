"""Tests for Hook system."""
from sophia.hooks import HookEvent, HookManager


class TestHookManager:
    def test_register_and_emit(self):
        hooks = HookManager()
        called = []
        hooks.register("test.event", lambda ctx: called.append(ctx) or ctx)
        hooks.emit("test.event", {"value": 42})
        assert len(called) == 1
        assert called[0]["value"] == 42

    def test_emit_no_handlers(self):
        hooks = HookManager()
        result = hooks.emit("nonexistent.event", {"x": 1})
        assert result == {"x": 1}

    def test_emit_default_context(self):
        hooks = HookManager()
        result = hooks.emit("nonexistent.event")
        assert result == {}

    def test_handler_modifies_context(self):
        hooks = HookManager()
        hooks.register("test", lambda ctx: {**ctx, "modified": True})
        result = hooks.emit("test", {"original": 1})
        assert result["original"] == 1
        assert result["modified"] is True

    def test_handler_returns_none(self):
        hooks = HookManager()
        ctx = {"value": 1}
        hooks.register("test", lambda ctx: None)
        result = hooks.emit("test", ctx)
        assert result["value"] == 1

    def test_priority_ordering(self):
        hooks = HookManager()
        order = []
        hooks.register("test", lambda ctx: (order.append(2), ctx)[1], priority=200)
        hooks.register("test", lambda ctx: (order.append(1), ctx)[1], priority=50)
        hooks.register("test", lambda ctx: (order.append(3), ctx)[1], priority=100)
        hooks.emit("test")
        assert order == [1, 3, 2]

    def test_blocked_chain(self):
        hooks = HookManager()
        called = []
        hooks.register("test", lambda ctx: {**ctx, "blocked": True, "block_reason": "stop"})
        hooks.register("test", lambda ctx: (called.append("should_not_run"), ctx)[1], priority=200)
        result = hooks.emit("test")
        assert result["blocked"] is True
        assert called == []

    def test_remove_by_handler(self):
        hooks = HookManager()
        handler = lambda ctx: ctx
        hooks.register("test", handler)
        assert hooks.has_hooks("test")
        removed = hooks.remove("test", handler=handler)
        assert removed is True
        assert not hooks.has_hooks("test")

    def test_remove_by_name(self):
        hooks = HookManager()
        hooks.register("test", lambda ctx: ctx, name="my_handler")
        removed = hooks.remove("test", name="my_handler")
        assert removed is True
        assert not hooks.has_hooks("test")

    def test_remove_nonexistent(self):
        hooks = HookManager()
        assert hooks.remove("nonexistent", name="x") is False

    def test_remove_all_event(self):
        hooks = HookManager()
        hooks.register("test1", lambda ctx: ctx)
        hooks.register("test2", lambda ctx: ctx)
        hooks.remove_all("test1")
        assert not hooks.has_hooks("test1")
        assert hooks.has_hooks("test2")

    def test_remove_all_everything(self):
        hooks = HookManager()
        hooks.register("test1", lambda ctx: ctx)
        hooks.register("test2", lambda ctx: ctx)
        hooks.remove_all()
        assert not hooks.has_hooks("test1")
        assert not hooks.has_hooks("test2")

    def test_list_hooks(self):
        hooks = HookManager()
        hooks.register("test1", lambda ctx: ctx, name="h1")
        hooks.register("test1", lambda ctx: ctx, name="h2")
        hooks.register("test2", lambda ctx: ctx, name="h3")
        result = hooks.list_hooks()
        assert len(result["test1"]) == 2
        assert len(result["test2"]) == 1
        assert result["test1"][0]["name"] == "h1"

    def test_list_hooks_filtered(self):
        hooks = HookManager()
        hooks.register("test1", lambda ctx: ctx, name="h1")
        hooks.register("test2", lambda ctx: ctx, name="h2")
        result = hooks.list_hooks("test1")
        assert "test1" in result
        assert "test2" not in result

    def test_handler_exception_doesnt_break_chain(self):
        hooks = HookManager()
        called = []
        hooks.register("test", lambda ctx: (_ for _ in ()).throw(ValueError("boom")), priority=1)
        hooks.register("test", lambda ctx: (called.append("ok"), ctx)[1], priority=2)
        result = hooks.emit("test", {"x": 1})
        assert called == ["ok"]
        assert result["x"] == 1


class TestHookEvent:
    def test_event_constants_exist(self):
        assert HookEvent.AGENT_PRE_RUN == "agent.pre_run"
        assert HookEvent.TOOL_PRE_DISPATCH == "tool.pre_dispatch"
        assert HookEvent.GOAL_CREATED == "goal.created"
        assert HookEvent.SUBAGENT_SPAWN == "subagent.spawn"
        assert HookEvent.LOOP_TICK == "loop.tick"
        assert HookEvent.MEMORY_STORE == "memory.store"
        assert HookEvent.CONTEXT_COMPRESS == "context.compress"
        assert HookEvent.CREDENTIAL_ROTATE == "credential.rotate"
        assert HookEvent.RECOVERY_RETRY == "recovery.retry"
        assert HookEvent.GUARDRAIL_BLOCK == "guardrail.block"
        assert HookEvent.TRAJECTORY_RECORD == "trajectory.record"
