"""Shared test fixtures."""

import pytest

from sophia.config import Config
from sophia.session import SessionManager


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def config(tmp_workspace, tmp_path):
    """Create a test config."""
    c = Config()
    c.session.workspace = tmp_workspace
    c.session.db_path = str(tmp_path / "test_sessions.db")
    c.model.api_key = "test-key"
    c.model.base_url = "http://localhost:9999/v1"
    c.model.name = "test-model"
    return c


@pytest.fixture
def session_mgr(tmp_path):
    """Create a test SessionManager."""
    db_path = str(tmp_path / "test.db")
    return SessionManager(db_path)
