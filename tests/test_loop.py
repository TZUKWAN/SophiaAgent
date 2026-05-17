"""Tests for the loop execution system (sophia.loop)."""

import json
import time
import threading

import pytest

from sophia.hooks import HookEvent, HookManager
from sophia.loop import LoopManager, LoopSpec, register_loop_tools
from sophia.tools.registry import ToolRegistry


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def hooks():
    return HookManager()


@pytest.fixture
def mock_run_fn():
    """A mock run_fn that records calls and returns a fixed response."""
    calls = []
    lock = threading.Lock()

    def run(prompt: str) -> str:
        with lock:
            calls.append(prompt)
        return f"result-for: {prompt}"

    run.calls = calls
    run.lock = lock
    return run


@pytest.fixture
def loop_mgr(tmp_path, hooks, mock_run_fn):
    db_path = str(tmp_path / "test_loops.db")
    return LoopManager(run_fn=mock_run_fn, hooks=hooks, db_path=db_path)


# ── 1. Create a loop ────────────────────────────────────────

class TestCreateLoop:
    def test_create_sets_fields(self, loop_mgr):
        spec = loop_mgr.create(
            session_id="sess1",
            name="test-loop",
            trigger_type="interval",
            trigger_config={"seconds": 60},
            action_prompt="Do something",
            max_iterations=5,
        )
        assert isinstance(spec, LoopSpec)
        assert spec.session_id == "sess1"
        assert spec.name == "test-loop"
        assert spec.trigger_type == "interval"
        assert spec.trigger_config == {"seconds": 60}
        assert spec.action_prompt == "Do something"
        assert spec.max_iterations == 5
        assert spec.current_iteration == 0
        assert spec.status == "pending"
        assert spec.last_result is None
        assert len(spec.id) > 0

    def test_create_with_condition_trigger(self, loop_mgr):
        spec = loop_mgr.create(
            session_id="sess1",
            name="cond-loop",
            trigger_type="condition",
            trigger_config={"check_prompt": "Is it done?", "poll_seconds": 5},
            action_prompt="Handle completion",
        )
        assert spec.trigger_type == "condition"
        assert spec.trigger_config["check_prompt"] == "Is it done?"

    def test_create_default_max_iterations(self, loop_mgr):
        spec = loop_mgr.create(
            session_id="sess1",
            name="unlimited",
            trigger_type="interval",
            trigger_config={"seconds": 10},
            action_prompt="Run forever",
        )
        assert spec.max_iterations == 0


# ── 2. Start a loop with interval trigger ────────────────────

class TestStartLoop:
    def test_interval_executes(self, loop_mgr, mock_run_fn):
        spec = loop_mgr.create(
            session_id="sess1",
            name="fast-loop",
            trigger_type="interval",
            trigger_config={"seconds": 0.3},
            action_prompt="Tick",
            max_iterations=3,
        )
        started = loop_mgr.start(spec.id)
        assert started is True

        # Wait enough for 3 ticks at 0.3s intervals
        time.sleep(1.5)

        # Check that run_fn was called at least 3 times
        with mock_run_fn.lock:
            call_count = len(mock_run_fn.calls)
        assert call_count >= 3

        # Verify DB state reflects completion
        updated = loop_mgr.get(spec.id)
        assert updated is not None
        assert updated.current_iteration >= 3
        assert updated.status in ("completed", "running")

    def test_start_nonexistent_returns_false(self, loop_mgr):
        assert loop_mgr.start("nonexistent-id") is False

    def test_status_becomes_running(self, loop_mgr):
        spec = loop_mgr.create(
            session_id="sess1",
            name="status-check",
            trigger_type="interval",
            trigger_config={"seconds": 10},
            action_prompt="Wait",
            max_iterations=1,
        )
        assert spec.status == "pending"
        loop_mgr.start(spec.id)
        updated = loop_mgr.get(spec.id)
        assert updated.status == "running"


# ── 3. Stop a loop ──────────────────────────────────────────

class TestStopLoop:
    def test_stop_changes_status(self, loop_mgr):
        spec = loop_mgr.create(
            session_id="sess1",
            name="stoppable",
            trigger_type="interval",
            trigger_config={"seconds": 60},
            action_prompt="Should not execute",
            max_iterations=100,
        )
        loop_mgr.start(spec.id)
        time.sleep(0.2)

        stopped = loop_mgr.stop(spec.id)
        assert stopped is True

        updated = loop_mgr.get(spec.id)
        assert updated.status == "completed"

    def test_stop_nonexistent_returns_false(self, loop_mgr):
        assert loop_mgr.stop("nonexistent") is False


# ── 4. Max iterations ───────────────────────────────────────

class TestMaxIterations:
    def test_stops_after_max(self, loop_mgr, mock_run_fn):
        spec = loop_mgr.create(
            session_id="sess1",
            name="limited",
            trigger_type="interval",
            trigger_config={"seconds": 0.3},
            action_prompt="Limited tick",
            max_iterations=2,
        )
        loop_mgr.start(spec.id)

        # Wait enough for 2 ticks + some margin
        time.sleep(1.5)

        updated = loop_mgr.get(spec.id)
        assert updated is not None
        assert updated.current_iteration == 2
        assert updated.status == "completed"


# ── 5. List loops ────────────────────────────────────────────

class TestListLoops:
    def test_list_loops_for_session(self, loop_mgr):
        loop_mgr.create(
            session_id="sess1",
            name="loop-a",
            trigger_type="interval",
            trigger_config={"seconds": 10},
            action_prompt="A",
        )
        loop_mgr.create(
            session_id="sess1",
            name="loop-b",
            trigger_type="condition",
            trigger_config={"check_prompt": "Check?"},
            action_prompt="B",
        )
        loop_mgr.create(
            session_id="sess2",
            name="loop-c",
            trigger_type="interval",
            trigger_config={"seconds": 10},
            action_prompt="C",
        )

        loops_s1 = loop_mgr.list_loops("sess1")
        loops_s2 = loop_mgr.list_loops("sess2")

        assert len(loops_s1) == 2
        assert len(loops_s2) == 1
        assert all(l.session_id == "sess1" for l in loops_s1)
        assert loops_s2[0].name == "loop-c"

    def test_list_active(self, loop_mgr):
        spec1 = loop_mgr.create(
            session_id="sess1",
            name="running-loop",
            trigger_type="interval",
            trigger_config={"seconds": 60},
            action_prompt="Run",
            max_iterations=100,
        )
        spec2 = loop_mgr.create(
            session_id="sess1",
            name="pending-loop",
            trigger_type="interval",
            trigger_config={"seconds": 60},
            action_prompt="Wait",
        )

        loop_mgr.start(spec1.id)

        active = loop_mgr.list_active("sess1")
        assert len(active) == 1
        assert active[0].id == spec1.id


# ── 6. Get loop by id ───────────────────────────────────────

class TestGetLoop:
    def test_get_existing(self, loop_mgr):
        spec = loop_mgr.create(
            session_id="sess1",
            name="findable",
            trigger_type="interval",
            trigger_config={"seconds": 10},
            action_prompt="Test",
        )
        found = loop_mgr.get(spec.id)
        assert found is not None
        assert found.id == spec.id
        assert found.name == "findable"

    def test_get_nonexistent(self, loop_mgr):
        found = loop_mgr.get("does-not-exist")
        assert found is None


# ── 7. Pause and resume ─────────────────────────────────────

class TestPauseResume:
    def test_pause_sets_status(self, loop_mgr):
        spec = loop_mgr.create(
            session_id="sess1",
            name="pausable",
            trigger_type="interval",
            trigger_config={"seconds": 60},
            action_prompt="Pause me",
            max_iterations=100,
        )
        loop_mgr.start(spec.id)
        time.sleep(0.2)

        paused = loop_mgr.pause(spec.id)
        assert paused is True

        updated = loop_mgr.get(spec.id)
        assert updated.status == "paused"

    def test_resume_sets_status_back(self, loop_mgr):
        spec = loop_mgr.create(
            session_id="sess1",
            name="resumable",
            trigger_type="interval",
            trigger_config={"seconds": 60},
            action_prompt="Resume me",
            max_iterations=100,
        )
        loop_mgr.start(spec.id)
        time.sleep(0.2)

        loop_mgr.pause(spec.id)
        resumed = loop_mgr.resume(spec.id)
        assert resumed is True

        updated = loop_mgr.get(spec.id)
        assert updated.status == "running"

    def test_pause_nonexistent(self, loop_mgr):
        assert loop_mgr.pause("nope") is False

    def test_resume_nonexistent(self, loop_mgr):
        assert loop_mgr.resume("nope") is False

    def test_resume_not_paused(self, loop_mgr):
        spec = loop_mgr.create(
            session_id="sess1",
            name="not-paused",
            trigger_type="interval",
            trigger_config={"seconds": 60},
            action_prompt="Test",
        )
        # Never started, so not paused
        assert loop_mgr.resume(spec.id) is False


# ── 8. Tool registration and dispatch ────────────────────────

class TestLoopTools:
    def test_register_all_tools(self, loop_mgr):
        registry = ToolRegistry()
        register_loop_tools(registry, loop_mgr)
        tools = registry.list_tools()
        assert "loop_start" in tools
        assert "loop_stop" in tools
        assert "loop_pause" in tools
        assert "loop_resume" in tools
        assert "loop_list" in tools

    def test_loop_start_tool(self, loop_mgr):
        registry = ToolRegistry()
        register_loop_tools(registry, loop_mgr)

        result = json.loads(registry.dispatch("loop_start", {
            "session_id": "sess1",
            "name": "tool-loop",
            "trigger_type": "interval",
            "trigger_config": {"seconds": 60},
            "action_prompt": "Tool test",
            "max_iterations": 1,
        }))
        assert "id" in result
        assert result["status"] == "running"
        assert result["name"] == "tool-loop"

        # Clean up
        loop_mgr.stop(result["id"])

    def test_loop_stop_tool(self, loop_mgr):
        registry = ToolRegistry()
        register_loop_tools(registry, loop_mgr)

        start_result = json.loads(registry.dispatch("loop_start", {
            "session_id": "sess1",
            "name": "stoppable",
            "trigger_type": "interval",
            "trigger_config": {"seconds": 60},
            "action_prompt": "Stop me",
        }))
        loop_id = start_result["id"]

        stop_result = json.loads(registry.dispatch("loop_stop", {"loop_id": loop_id}))
        assert stop_result["status"] == "completed"

    def test_loop_list_tool(self, loop_mgr):
        registry = ToolRegistry()
        register_loop_tools(registry, loop_mgr)

        # Create a loop directly (not via tool, so we don't start it)
        loop_mgr.create(
            session_id="sess1",
            name="listed",
            trigger_type="interval",
            trigger_config={"seconds": 10},
            action_prompt="List me",
        )

        result = json.loads(registry.dispatch("loop_list", {"session_id": "sess1"}))
        assert result["total"] == 1
        assert result["loops"][0]["name"] == "listed"

    def test_loop_start_missing_fields(self, loop_mgr):
        registry = ToolRegistry()
        register_loop_tools(registry, loop_mgr)

        result = json.loads(registry.dispatch("loop_start", {}))
        assert "error" in result

    def test_loop_pause_resume_tools(self, loop_mgr):
        registry = ToolRegistry()
        register_loop_tools(registry, loop_mgr)

        start_result = json.loads(registry.dispatch("loop_start", {
            "session_id": "sess1",
            "name": "pr-loop",
            "trigger_type": "interval",
            "trigger_config": {"seconds": 60},
            "action_prompt": "Pause/resume test",
        }))
        loop_id = start_result["id"]
        time.sleep(0.2)

        pause_result = json.loads(registry.dispatch("loop_pause", {"loop_id": loop_id}))
        assert pause_result["status"] == "paused"

        resume_result = json.loads(registry.dispatch("loop_resume", {"loop_id": loop_id}))
        assert resume_result["status"] == "running"

        # Clean up
        loop_mgr.stop(loop_id)


# ── 9. Hook emission on tick ─────────────────────────────────

class TestHookEmission:
    def test_loop_tick_hook_fired(self, tmp_path, hooks):
        tick_events = []

        def capture_tick(ctx):
            tick_events.append(dict(ctx))
            return ctx

        hooks.register(HookEvent.LOOP_TICK, capture_tick)

        db_path = str(tmp_path / "hook_test.db")
        call_count = 0
        lock = threading.Lock()

        def run_fn(prompt: str) -> str:
            nonlocal call_count
            with lock:
                call_count += 1
            return f"tick-{call_count}"

        mgr = LoopManager(run_fn=run_fn, hooks=hooks, db_path=db_path)

        spec = mgr.create(
            session_id="sess1",
            name="hook-loop",
            trigger_type="interval",
            trigger_config={"seconds": 0.3},
            action_prompt="Fire hooks",
            max_iterations=2,
        )
        mgr.start(spec.id)

        # Wait for 2 ticks
        time.sleep(1.5)

        assert len(tick_events) >= 2
        first = tick_events[0]
        assert first["loop_id"] == spec.id
        assert first["session_id"] == "sess1"
        assert "iteration" in first
        assert "result" in first

    def test_loop_complete_hook_fired(self, tmp_path, hooks):
        complete_events = []

        def capture_complete(ctx):
            complete_events.append(dict(ctx))
            return ctx

        hooks.register(HookEvent.LOOP_COMPLETE, capture_complete)

        db_path = str(tmp_path / "hook_complete.db")

        def run_fn(prompt: str) -> str:
            return "done"

        mgr = LoopManager(run_fn=run_fn, hooks=hooks, db_path=db_path)

        spec = mgr.create(
            session_id="sess1",
            name="complete-hook",
            trigger_type="interval",
            trigger_config={"seconds": 0.3},
            action_prompt="Finish fast",
            max_iterations=2,
        )
        mgr.start(spec.id)

        time.sleep(1.5)

        assert len(complete_events) >= 1
        assert complete_events[0]["loop_id"] == spec.id
        assert complete_events[0]["iterations"] == 2

    def test_loop_error_hook_fired(self, tmp_path, hooks):
        error_events = []

        def capture_error(ctx):
            error_events.append(dict(ctx))
            return ctx

        hooks.register(HookEvent.LOOP_ERROR, capture_error)

        db_path = str(tmp_path / "hook_error.db")
        call_count = 0

        def failing_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise RuntimeError("Intentional failure")
            return "ok"

        mgr = LoopManager(run_fn=failing_fn, hooks=hooks, db_path=db_path)

        spec = mgr.create(
            session_id="sess1",
            name="error-loop",
            trigger_type="interval",
            trigger_config={"seconds": 0.3},
            action_prompt="Fail me",
            max_iterations=5,
        )
        mgr.start(spec.id)

        time.sleep(1.0)

        assert len(error_events) >= 1
        assert "Intentional failure" in error_events[0]["error"]

        updated = mgr.get(spec.id)
        assert updated.status == "failed"
