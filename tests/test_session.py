"""Tests for SessionManager."""


def test_create_session(session_mgr):
    sid = session_mgr.create_session(title="test", model="m1")
    assert len(sid) == 8
    s = session_mgr.get_session(sid)
    assert s is not None
    assert s["title"] == "test"


def test_list_sessions(session_mgr):
    session_mgr.create_session(title="first")
    session_mgr.create_session(title="second")
    sessions = session_mgr.list_sessions()
    assert len(sessions) == 2


def test_delete_session(session_mgr):
    sid = session_mgr.create_session(title="to-delete")
    session_mgr.delete_session(sid)
    assert session_mgr.get_session(sid) is None


def test_add_and_get_messages(session_mgr):
    sid = session_mgr.create_session(title="msg-test")
    session_mgr.add_message(sid, "user", "hello")
    session_mgr.add_message(sid, "assistant", "hi there")
    msgs = session_mgr.get_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["content"] == "hi there"


def test_batch_messages(session_mgr):
    sid = session_mgr.create_session(title="batch")
    session_mgr.add_messages_batch(sid, [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ])
    msgs = session_mgr.get_messages(sid)
    assert len(msgs) == 3


def test_checkpoint_save_restore(session_mgr):
    sid = session_mgr.create_session(title="cp-test")
    session_mgr.add_message(sid, "user", "msg1")
    session_mgr.add_message(sid, "assistant", "msg2")

    cp_id = session_mgr.save_checkpoint(sid, "before-delete")
    session_mgr.add_message(sid, "user", "msg3")

    msgs_before = session_mgr.get_messages(sid)
    assert len(msgs_before) == 3

    ok = session_mgr.restore_checkpoint(sid, cp_id)
    assert ok
    msgs_after = session_mgr.get_messages(sid)
    assert len(msgs_after) == 2


def test_list_checkpoints(session_mgr):
    sid = session_mgr.create_session(title="cp-list")
    session_mgr.add_message(sid, "user", "x")
    session_mgr.save_checkpoint(sid, "cp1")
    session_mgr.save_checkpoint(sid, "cp2")
    cps = session_mgr.list_checkpoints(sid)
    assert len(cps) == 2
