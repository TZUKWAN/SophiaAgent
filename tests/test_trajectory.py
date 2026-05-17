"""Tests for sophia.trajectory.TrajectoryRecorder."""

import json
import os
import time

import pytest

from sophia.hooks import HookEvent, HookManager
from sophia.trajectory import TrajectoryRecorder


@pytest.fixture
def hooks():
    return HookManager()


@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path / "trajectories")


@pytest.fixture
def recorder(hooks, output_dir):
    return TrajectoryRecorder(hooks, output_dir)


class TestStartStopRecording:
    """Tests for start_recording and stop_recording."""

    def test_start_recording_creates_session(self, recorder):
        recorder.start_recording("sess1")
        assert recorder.is_recording("sess1")

    def test_stop_recording_ends_session(self, recorder):
        recorder.start_recording("sess1")
        path = recorder.stop_recording("sess1")
        assert not recorder.is_recording("sess1")

    def test_stop_recording_returns_filepath(self, recorder, output_dir):
        recorder.start_recording("sess1")
        path = recorder.stop_recording("sess1")
        assert path == os.path.join(output_dir, "trajectory_sess1.jsonl")

    def test_stop_creates_output_dir(self, hooks, tmp_path):
        out = str(tmp_path / "new_dir" / "traj")
        rec = TrajectoryRecorder(hooks, out)
        rec.start_recording("s1")
        path = rec.stop_recording("s1")
        assert os.path.isdir(os.path.dirname(path))

    def test_stop_empty_session_writes_empty_file(self, recorder, output_dir):
        recorder.start_recording("sess1")
        path = recorder.stop_recording("sess1")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        assert lines == []

    def test_stop_nonexistent_session_returns_empty(self, recorder, output_dir):
        path = recorder.stop_recording("nonexistent")
        assert os.path.exists(path)


class TestOnEventRecordsEntries:
    """Tests for _on_event hook handler."""

    def test_records_to_active_session(self, recorder):
        recorder.start_recording("sess1")
        recorder._on_event({"tool": "file_read", "args": {"path": "/tmp"}})

        entries = recorder.get_entries("sess1")
        assert len(entries) == 1
        assert entries[0]["context"]["tool"] == "file_read"

    def test_records_to_multiple_sessions(self, recorder):
        recorder.start_recording("s1")
        recorder.start_recording("s2")

        recorder._on_event({"action": "test"})

        assert len(recorder.get_entries("s1")) == 1
        assert len(recorder.get_entries("s2")) == 1

    def test_does_not_record_to_stopped_session(self, recorder):
        recorder.start_recording("s1")
        recorder._on_event({"first": True})
        recorder.stop_recording("s1")

        recorder._on_event({"second": True})
        # s1 is stopped, so no new entries
        assert recorder.get_entries("s1") == []

    def test_entry_has_timestamp(self, recorder):
        recorder.start_recording("s1")
        before = time.time()
        recorder._on_event({"data": "test"})
        after = time.time()

        entries = recorder.get_entries("s1")
        assert before <= entries[0]["timestamp"] <= after

    def test_truncates_long_values(self, recorder):
        recorder.start_recording("s1")
        long_val = "x" * 1000
        recorder._on_event({"big": long_val})

        entries = recorder.get_entries("s1")
        assert len(entries[0]["context"]["big"]) == 500

    def test_returns_context_unchanged(self, recorder):
        recorder.start_recording("s1")
        ctx = {"key": "value"}
        result = recorder._on_event(ctx)
        assert result is ctx


class TestGetEntries:
    """Tests for get_entries."""

    def test_empty_for_unstarted_session(self, recorder):
        assert recorder.get_entries("no_such_session") == []

    def test_returns_recorded_entries(self, recorder):
        recorder.start_recording("s1")
        recorder._on_event({"a": 1})
        recorder._on_event({"b": 2})

        entries = recorder.get_entries("s1")
        assert len(entries) == 2


class TestIsRecording:
    """Tests for is_recording."""

    def test_false_before_start(self, recorder):
        assert recorder.is_recording("s1") is False

    def test_true_after_start(self, recorder):
        recorder.start_recording("s1")
        assert recorder.is_recording("s1") is True

    def test_false_after_stop(self, recorder):
        recorder.start_recording("s1")
        recorder.stop_recording("s1")
        assert recorder.is_recording("s1") is False


class TestOutputFileValidJSONL:
    """Tests that stop_recording writes valid JSONL."""

    def test_valid_jsonl_output(self, recorder):
        recorder.start_recording("s1")
        recorder._on_event({"tool": "file_read"})
        recorder._on_event({"tool": "web_search"})

        path = recorder.stop_recording("s1")

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "context" in parsed

    def test_jsonl_preserves_context(self, recorder):
        recorder.start_recording("s1")
        recorder._on_event({"goal_id": "g1", "status": "completed"})

        path = recorder.stop_recording("s1")

        with open(path, encoding="utf-8") as f:
            entry = json.loads(f.readline())

        assert entry["context"]["goal_id"] == "g1"
        assert entry["context"]["status"] == "completed"

    def test_multiple_sessions_separate_files(self, recorder):
        recorder.start_recording("s1")
        recorder._on_event({"session": 1})
        recorder.stop_recording("s1")

        recorder.start_recording("s2")
        recorder._on_event({"session": 2})
        recorder.stop_recording("s2")

        with open(os.path.join(recorder.output_dir, "trajectory_s1.jsonl"), encoding="utf-8") as f:
            lines1 = f.readlines()
        with open(os.path.join(recorder.output_dir, "trajectory_s2.jsonl"), encoding="utf-8") as f:
            lines2 = f.readlines()

        assert len(lines1) == 1
        assert len(lines2) == 1
        assert json.loads(lines1[0])["context"]["session"] == "1"
        assert json.loads(lines2[0])["context"]["session"] == "2"


class TestHookRegistration:
    """Tests that hooks are properly registered."""

    def test_register_hooks_on_first_start(self, hooks, output_dir):
        rec = TrajectoryRecorder(hooks, output_dir)
        assert not rec._registered

        rec.start_recording("s1")
        assert rec._registered

        # Check hooks were registered
        hook_list = hooks.list_hooks(HookEvent.AGENT_PRE_RUN)
        assert any("trajectory_" in h["name"] for h in hook_list.get(HookEvent.AGENT_PRE_RUN, []))

    def test_hooks_fire_to_recorder(self, hooks, output_dir):
        rec = TrajectoryRecorder(hooks, output_dir)
        rec.start_recording("s1")

        # Emit a hook event
        hooks.emit(HookEvent.TOOL_POST_DISPATCH, {"tool": "file_read"})
        hooks.emit(HookEvent.GOAL_CREATED, {"goal_id": "g1"})

        entries = rec.get_entries("s1")
        assert len(entries) == 2
        assert entries[0]["context"]["tool"] == "file_read"
        assert entries[1]["context"]["goal_id"] == "g1"

    def test_no_double_registration(self, hooks, output_dir):
        rec = TrajectoryRecorder(hooks, output_dir)
        rec.start_recording("s1")
        rec.start_recording("s2")

        # Should only register once
        hook_list = hooks.list_hooks(HookEvent.AGENT_PRE_RUN)
        trajectory_hooks = [h for h in hook_list.get(HookEvent.AGENT_PRE_RUN, []) if "trajectory_" in h["name"]]
        assert len(trajectory_hooks) == 1
