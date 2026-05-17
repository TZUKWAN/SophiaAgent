import sophia.lifecycle as lifecycle


def test_register_and_unregister_child_process():
    lifecycle.register_child_process(12345)
    assert 12345 in lifecycle._CHILD_PIDS
    lifecycle.unregister_child_process(12345)
    assert 12345 not in lifecycle._CHILD_PIDS


def test_cleanup_registered_children_calls_terminator(monkeypatch):
    killed = []
    monkeypatch.setattr(lifecycle, "_terminate_process_tree", lambda pid: killed.append(pid))
    lifecycle.register_child_process(111)
    lifecycle.register_child_process(222)
    lifecycle.cleanup_registered_children()
    assert sorted(killed) == [111, 222]
    assert not lifecycle._CHILD_PIDS


def test_install_lifecycle_hooks_is_idempotent(monkeypatch):
    monkeypatch.setattr(lifecycle, "_INSTALLED", False)
    calls = []
    monkeypatch.setattr(lifecycle, "_install_windows_job_object", lambda: calls.append("job"))
    monkeypatch.setattr(lifecycle, "_install_signal_handlers", lambda: calls.append("signals"))
    lifecycle.install_process_lifecycle_hooks(monitor_parent=False, use_windows_job=False)
    lifecycle.install_process_lifecycle_hooks(monitor_parent=False, use_windows_job=False)
    assert calls == ["signals"]
