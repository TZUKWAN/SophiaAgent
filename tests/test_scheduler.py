"""Tests for Cron Scheduler."""
import json
from sophia.scheduler import CronScheduler
from sophia.hooks import HookManager


def _make_scheduler(tmp_path, run_fn=None, hooks=None):
    db = str(tmp_path / "test.db")
    if run_fn is None:
        run_fn = lambda prompt: f"executed: {prompt[:20]}"
    return CronScheduler(run_fn, hooks or HookManager(), db)


class TestCronScheduler:
    def test_schedule(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        task_id = sched.schedule("s1", "test task", "*/5 * * * *", "run this")
        assert task_id

    def test_unschedule(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        tid = sched.schedule("s1", "test", "*/5 * * * *", "prompt")
        assert sched.unschedule(tid) is True
        assert sched.unschedule("nonexistent") is False

    def test_list_scheduled(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        sched.schedule("s1", "task1", "*/5 * * * *", "p1")
        sched.schedule("s1", "task2", "*/10 * * * *", "p2")
        tasks = sched.list_scheduled("s1")
        assert len(tasks) == 2

    def test_should_fire_every_5(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        # */5 fires on minutes 0, 5, 10, 15, ...
        import datetime
        now = datetime.datetime.now()
        if now.minute % 5 == 0 and now.second < 10:
            assert sched._should_fire("*/5 * * * *") is True
        else:
            assert sched._should_fire("*/5 * * * *") is False

    def test_should_fire_invalid(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        assert sched._should_fire("invalid") is False

    def test_start_stop_scheduler(self, tmp_path):
        import time
        sched = _make_scheduler(tmp_path)
        sched.start_scheduler()
        assert sched._running is True
        sched.stop_scheduler()
        time.sleep(0.5)
        assert sched._running is False

    def test_list_scheduled_filters_by_session(self, tmp_path):
        sched = _make_scheduler(tmp_path)
        sched.schedule("s1", "t1", "*/5 * * * *", "p1")
        sched.schedule("s2", "t2", "*/5 * * * *", "p2")
        assert len(sched.list_scheduled("s1")) == 1
        assert len(sched.list_scheduled("s2")) == 1
