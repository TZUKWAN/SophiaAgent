import json
import subprocess

from sophia import integrations


def test_sophia_mcp_server_config_uses_stdio():
    config = integrations.sophia_mcp_server_config("python")

    assert config == {
        "type": "stdio",
        "command": "python",
        "args": ["-m", "sophia", "serve", "--stdio"],
    }


def test_codex_install_skips_when_command_not_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(integrations, "detect_client", lambda name: None)

    result = integrations.install_codex(home=tmp_path)

    assert result.client == "codex"
    assert result.detected is False
    assert result.installed is False
    assert not (tmp_path / ".agents").exists()


def test_codex_install_force_writes_plugin_and_marketplace(tmp_path, monkeypatch):
    monkeypatch.setattr(integrations, "detect_client", lambda name: None)

    result = integrations.install_codex(
        home=tmp_path,
        force=True,
        python_executable="python",
    )

    plugin_json = tmp_path / ".agents" / "plugins" / "plugins" / "sophia" / ".codex-plugin" / "plugin.json"
    mcp_json = tmp_path / ".agents" / "plugins" / "plugins" / "sophia" / ".mcp.json"
    marketplace = tmp_path / ".agents" / "plugins" / "marketplace.json"

    assert result.installed is True
    assert json.loads(plugin_json.read_text(encoding="utf-8"))["name"] == "sophia"
    assert json.loads(mcp_json.read_text(encoding="utf-8"))["mcpServers"]["sophia"]["command"] == "python"
    assert json.loads(marketplace.read_text(encoding="utf-8"))["plugins"][0]["name"] == "sophia"


def test_claude_install_uses_cli_registration_when_detected(tmp_path, monkeypatch):
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(integrations, "detect_client", lambda name: "claude")

    result = integrations.install_claude_code(
        home=tmp_path,
        python_executable="python",
        runner=fake_runner,
    )

    command_file = tmp_path / ".claude" / "commands" / "sophia.md"
    skill_file = tmp_path / ".claude" / "skills" / "sophia" / "SKILL.md"

    assert result.installed is True
    assert command_file.exists()
    assert skill_file.exists()
    assert commands
    assert commands[0][:4] == ["claude", "mcp", "add-json", "sophia"]
    assert "sophia_ask" in command_file.read_text(encoding="utf-8")


def test_repo_integration_files_are_created(tmp_path):
    paths = integrations.repo_integration_files(tmp_path, python_executable="python")

    assert tmp_path / ".mcp.json" in paths
    assert (tmp_path / ".claude" / "commands" / "sophia.md").exists()
    assert (tmp_path / "plugins" / "sophia" / ".codex-plugin" / "plugin.json").exists()
    assert (tmp_path / ".agents" / "plugins" / "marketplace.json").exists()

