"""Tests for Config module."""
import os

from sophia.config import Config


def test_default_config():
    c = Config()
    assert c.model.provider == "openai-compat"
    assert c.model.max_turns == 50


def test_env_var_override(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SOPHIA_API_KEY=env-test-key\n")

    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  api_key: ${SOPHIA_API_KEY}\n  name: test\n")

    os.environ["SOPHIA_API_KEY"] = "env-test-key"
    try:
        c = Config.load(str(config_file))
        assert c.model.api_key == "env-test-key"
    finally:
        os.environ.pop("SOPHIA_API_KEY", None)


def test_resolve_env_vars():
    result = Config._resolve_env_vars({"key": "${HOME}", "nested": {"val": 42}})
    assert result["key"] == os.environ.get("HOME", "")
    assert result["nested"]["val"] == 42


def test_path_expansion():
    c = Config()
    assert "~" not in c.session.workspace
    assert "~" not in c.session.db_path
