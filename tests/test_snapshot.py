"""Tests for sophia.snapshot.SnapshotManager."""

import json
import os
import pytest

from sophia.snapshot import SnapshotManager


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "snapshots.db")


@pytest.fixture
def mgr(db_path, workspace):
    return SnapshotManager(db_path, workspace)


class TestCreateSnapshot:
    """Tests for create_snapshot."""

    def test_creates_and_returns_id(self, mgr):
        sid = mgr.create_snapshot("session1", "initial", message_count=5)
        assert isinstance(sid, int)
        assert sid >= 1

    def test_multiple_snapshots_increment_id(self, mgr):
        id1 = mgr.create_snapshot("session1", "first")
        id2 = mgr.create_snapshot("session1", "second")
        assert id2 > id1

    def test_stores_label_and_session(self, mgr):
        sid = mgr.create_snapshot("session1", "checkpoint_A", message_count=10)
        snap = mgr.get_snapshot(sid)
        assert snap["session_id"] == "session1"
        assert snap["label"] == "checkpoint_A"
        assert snap["message_count"] == 10

    def test_no_git_commit_without_git(self, mgr, workspace):
        sid = mgr.create_snapshot("session1", "no_git")
        snap = mgr.get_snapshot(sid)
        # Without a git repo, git_commit should be None
        assert snap["git_commit"] is None

    def test_file_manifest_stored(self, mgr, workspace):
        # Create a file in workspace
        with open(os.path.join(workspace, "test.txt"), "w") as f:
            f.write("hello")

        sid = mgr.create_snapshot("session1", "with_file")
        snap = mgr.get_snapshot(sid)
        manifest = json.loads(snap["file_manifest"])
        assert "test.txt" in manifest
        assert len(manifest["test.txt"]) == 32  # MD5 hex digest


class TestListSnapshots:
    """Tests for list_snapshots."""

    def test_empty_list(self, mgr):
        result = mgr.list_snapshots("session1")
        assert result == []

    def test_lists_by_session(self, mgr):
        mgr.create_snapshot("session1", "a")
        mgr.create_snapshot("session1", "b")
        mgr.create_snapshot("session2", "c")

        result = mgr.list_snapshots("session1")
        assert len(result) == 2
        # Ordered by id DESC
        assert result[0]["label"] == "b"
        assert result[1]["label"] == "a"

    def test_list_excludes_other_session(self, mgr):
        mgr.create_snapshot("session1", "a")
        mgr.create_snapshot("session2", "b")

        result = mgr.list_snapshots("session1")
        assert len(result) == 1
        assert result[0]["label"] == "a"


class TestGetSnapshot:
    """Tests for get_snapshot."""

    def test_returns_snapshot_dict(self, mgr):
        sid = mgr.create_snapshot("session1", "test_label", message_count=7)
        snap = mgr.get_snapshot(sid)
        assert snap["id"] == sid
        assert snap["session_id"] == "session1"
        assert snap["label"] == "test_label"
        assert snap["message_count"] == 7
        assert snap["created_at"] is not None

    def test_returns_none_for_missing(self, mgr):
        result = mgr.get_snapshot(9999)
        assert result is None

    def test_returns_none_for_wrong_id(self, mgr):
        sid = mgr.create_snapshot("session1", "exists")
        result = mgr.get_snapshot(sid + 100)
        assert result is None


class TestDeleteSnapshot:
    """Tests for delete_snapshot."""

    def test_delete_existing(self, mgr):
        sid = mgr.create_snapshot("session1", "to_delete")
        assert mgr.delete_snapshot(sid) is True
        assert mgr.get_snapshot(sid) is None

    def test_delete_nonexistent(self, mgr):
        assert mgr.delete_snapshot(9999) is False

    def test_delete_twice(self, mgr):
        sid = mgr.create_snapshot("session1", "delete_twice")
        assert mgr.delete_snapshot(sid) is True
        assert mgr.delete_snapshot(sid) is False


class TestRestoreSnapshot:
    """Tests for restore_snapshot."""

    def test_restore_without_git_returns_false(self, mgr):
        sid = mgr.create_snapshot("session1", "no_git")
        # No git_commit stored, so restore should return False
        assert mgr.restore_snapshot(sid) is False

    def test_restore_nonexistent_returns_false(self, mgr):
        assert mgr.restore_snapshot(9999) is False


class TestComputeFileManifest:
    """Tests for _compute_file_manifest."""

    def test_empty_workspace(self, mgr, workspace):
        manifest = mgr._compute_file_manifest()
        assert manifest == {}

    def test_includes_files(self, mgr, workspace):
        with open(os.path.join(workspace, "a.txt"), "w") as f:
            f.write("aaa")
        with open(os.path.join(workspace, "b.txt"), "w") as f:
            f.write("bbb")

        manifest = mgr._compute_file_manifest()
        assert "a.txt" in manifest
        assert "b.txt" in manifest
        assert manifest["a.txt"] != manifest["b.txt"]

    def test_excludes_dotfiles(self, mgr, workspace):
        with open(os.path.join(workspace, ".hidden"), "w") as f:
            f.write("hidden")

        manifest = mgr._compute_file_manifest()
        assert ".hidden" not in manifest

    def test_nested_files(self, mgr, workspace):
        subdir = os.path.join(workspace, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "nested.txt"), "w") as f:
            f.write("nested content")

        manifest = mgr._compute_file_manifest()
        assert os.path.join("sub", "nested.txt") in manifest

    def test_nonexistent_workspace(self, db_path):
        mgr = SnapshotManager(db_path, "/nonexistent/path/workspace")
        manifest = mgr._compute_file_manifest()
        assert manifest == {}

    def test_md5_consistent(self, mgr, workspace):
        with open(os.path.join(workspace, "same.txt"), "w") as f:
            f.write("same content")

        m1 = mgr._compute_file_manifest()
        m2 = mgr._compute_file_manifest()
        assert m1 == m2
