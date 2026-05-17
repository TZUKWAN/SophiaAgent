"""Install SophiaAgent integrations for external coding agents."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


Runner = Callable[..., subprocess.CompletedProcess]


@dataclass
class IntegrationInstallResult:
    """Result for one client integration attempt."""

    client: str
    detected: bool
    installed: bool
    paths: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def sophia_mcp_server_config(python_executable: Optional[str] = None) -> Dict[str, Any]:
    """Return a stdio MCP server config that launches the installed SophiaAgent."""
    return {
        "type": "stdio",
        "command": python_executable or sys.executable,
        "args": ["-m", "sophia", "--workspace", ".", "serve", "--stdio"],
    }


def detect_client(command_name: str) -> Optional[str]:
    """Return the executable path for an installed client command, if available."""
    return shutil.which(command_name)


def install_all(
    *,
    home: Optional[Path] = None,
    force: bool = False,
    python_executable: Optional[str] = None,
    runner: Runner = subprocess.run,
) -> List[IntegrationInstallResult]:
    """Detect installed coding agents and install matching Sophia integrations."""
    return [
        install_claude_code(
            home=home, force=force, python_executable=python_executable, runner=runner
        ),
        install_codex(
            home=home, force=force, python_executable=python_executable
        ),
    ]


def install_claude_code(
    *,
    home: Optional[Path] = None,
    force: bool = False,
    python_executable: Optional[str] = None,
    runner: Runner = subprocess.run,
) -> IntegrationInstallResult:
    """Install Claude Code MCP, slash command, and skill integration."""
    home = Path(home or Path.home())
    claude_path = detect_client("claude")
    result = IntegrationInstallResult(
        client="claude-code",
        detected=bool(claude_path),
        installed=False,
    )
    if not claude_path and not force:
        result.messages.append("Claude Code command `claude` was not detected on PATH.")
        return result

    command_path = home / ".claude" / "commands" / "sophia.md"
    skill_path = home / ".claude" / "skills" / "sophia" / "SKILL.md"
    _write_text(command_path, _claude_slash_command_text(), result)
    _write_text(skill_path, _claude_skill_text(), result)

    if claude_path:
        config = sophia_mcp_server_config(python_executable)
        try:
            command = _claude_mcp_add_command(claude_path, config)
            completed = runner(command, capture_output=True, text=True, check=False)
            output = (completed.stderr or completed.stdout or "").strip()
            if completed.returncode != 0 and "already exists" in output.lower():
                remove_command = [claude_path, "mcp", "remove", "sophia", "--scope", "user"]
                removed = runner(remove_command, capture_output=True, text=True, check=False)
                if removed.returncode == 0:
                    completed = runner(command, capture_output=True, text=True, check=False)
                    output = (completed.stderr or completed.stdout or "").strip()
                    if completed.returncode == 0:
                        result.messages.append("Updated existing SophiaAgent MCP server in Claude Code.")
            if completed.returncode == 0:
                if not any("MCP server" in message for message in result.messages):
                    result.messages.append("Registered SophiaAgent MCP server with Claude Code.")
            else:
                result.errors.append(
                    f"Claude Code MCP registration command failed: {output}"
                )
        except OSError as exc:
            result.errors.append(f"Claude Code MCP registration failed: {exc}")
    else:
        result.messages.append("Wrote Claude command/skill files; MCP CLI registration skipped.")

    result.installed = not result.errors
    return result


def _claude_mcp_add_command(claude_path: str, config: Dict[str, Any]) -> List[str]:
    return [
        claude_path,
        "mcp",
        "add-json",
        "sophia",
        json.dumps(config, ensure_ascii=False),
        "--scope",
        "user",
    ]


def install_codex(
    *,
    home: Optional[Path] = None,
    force: bool = False,
    python_executable: Optional[str] = None,
) -> IntegrationInstallResult:
    """Install a local Codex plugin carrying Sophia MCP and skill metadata."""
    home = Path(home or Path.home())
    codex_path = detect_client("codex")
    result = IntegrationInstallResult(
        client="codex",
        detected=bool(codex_path),
        installed=False,
    )
    if not codex_path and not force:
        result.messages.append("Codex command `codex` was not detected on PATH.")
        return result

    marketplace_path = home / ".agents" / "plugins" / "marketplace.json"
    plugin_root = home / ".agents" / "plugins" / "plugins" / "sophia"
    _write_text(
        plugin_root / ".codex-plugin" / "plugin.json",
        json.dumps(_codex_plugin_manifest(), ensure_ascii=False, indent=2) + "\n",
        result,
    )
    _write_text(
        plugin_root / ".mcp.json",
        json.dumps({"mcpServers": {"sophia": sophia_mcp_server_config(python_executable)}},
                   ensure_ascii=False, indent=2) + "\n",
        result,
    )
    _write_text(plugin_root / "skills" / "sophia" / "SKILL.md", _codex_skill_text(), result)
    _upsert_marketplace(marketplace_path, result)
    result.installed = not result.errors
    return result


def repo_integration_files(root: Path, python_executable: Optional[str] = None) -> List[Path]:
    """Create project-level integration files inside the repository."""
    created: List[Path] = []
    root = Path(root)
    files = {
        root / ".mcp.json": json.dumps(
            {"mcpServers": {"sophia": sophia_mcp_server_config(python_executable)}},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        root / ".claude" / "commands" / "sophia.md": _claude_slash_command_text(),
        root / ".claude" / "skills" / "sophia" / "SKILL.md": _claude_skill_text(),
        root / "plugins" / "sophia" / ".codex-plugin" / "plugin.json": json.dumps(
            _codex_plugin_manifest(), ensure_ascii=False, indent=2
        ) + "\n",
        root / "plugins" / "sophia" / ".mcp.json": json.dumps(
            {"mcpServers": {"sophia": sophia_mcp_server_config(python_executable)}},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        root / "plugins" / "sophia" / "skills" / "sophia" / "SKILL.md": _codex_skill_text(),
        root / ".agents" / "plugins" / "marketplace.json": json.dumps(
            _repo_marketplace(), ensure_ascii=False, indent=2
        ) + "\n",
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        created.append(path)
    return created


def _write_text(path: Path, text: str, result: IntegrationInstallResult) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        result.paths.append(str(path))
    except OSError as exc:
        result.errors.append(f"Failed to write {path}: {exc}")


def _upsert_marketplace(path: Path, result: IntegrationInstallResult) -> None:
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"name": "local", "interface": {"displayName": "Local Plugins"}, "plugins": []}
        data.setdefault("name", "local")
        data.setdefault("interface", {}).setdefault("displayName", "Local Plugins")
        plugins = data.setdefault("plugins", [])
        entry = {
            "name": "sophia",
            "source": {"source": "local", "path": "./plugins/sophia"},
            "policy": {"installation": "AVAILABLE", "authentication": "ON_USE"},
            "category": "Productivity",
        }
        for idx, plugin in enumerate(plugins):
            if plugin.get("name") == "sophia":
                plugins[idx] = entry
                break
        else:
            plugins.append(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result.paths.append(str(path))
    except (OSError, json.JSONDecodeError) as exc:
        result.errors.append(f"Failed to update Codex marketplace {path}: {exc}")


def _claude_slash_command_text() -> str:
    return """---
description: Delegate the current request to SophiaAgent
argument-hint: [research, writing, review, data, or workflow request]
---

Use the SophiaAgent MCP tool `sophia_ask` for the following request and return
SophiaAgent's final answer. If the MCP server is unavailable, tell the user to
run `sophia integrate --auto` and do not fabricate a Sophia result.

$ARGUMENTS
"""


def _claude_skill_text() -> str:
    return """---
name: sophia
description: Use SophiaAgent for humanities and social-science research, literature review, academic writing, document review, citation management, data analysis, and multi-agent swarm workflows.
---

When a user asks for research synthesis, literature review, academic writing,
methodology advice, citation work, data analysis, or a complex multi-step
humanities/social-science task, prefer the SophiaAgent MCP tool `sophia_ask`.

Pass the complete user request as the `prompt` argument. SophiaAgent decides
internally whether to launch its automatic swarm. Return the final SophiaAgent
answer and clearly report any MCP/tool error instead of inventing output.
"""


def _codex_skill_text() -> str:
    return """---
name: sophia
description: Delegate humanities/social-science research, writing, review, citation, and data-analysis tasks to SophiaAgent through MCP.
---

Use SophiaAgent when the user asks for literature research, academic writing,
document review, citation management, research methodology, data analysis, or
multi-step scholarly workflows. Call the MCP tool `sophia_ask` with the full
user request as `prompt`. SophiaAgent will automatically decide whether to use
its internal swarm and return one final answer.

For explicit `/sophia ...` style requests, treat the text after `/sophia` as the
prompt for `sophia_ask`. Never fabricate SophiaAgent results if the MCP server
is unavailable; report the integration error and suggest `sophia integrate --auto`.
"""


def _codex_plugin_manifest() -> Dict[str, Any]:
    return {
        "name": "sophia",
        "version": "0.1.0",
        "description": "SophiaAgent MCP integration for research, writing, review, and swarm workflows.",
        "author": {"name": "SophiaAgent", "url": "https://github.com/TZUKWAN/SophiaAgent"},
        "homepage": "https://github.com/TZUKWAN/SophiaAgent",
        "repository": "https://github.com/TZUKWAN/SophiaAgent",
        "license": "MIT",
        "keywords": ["mcp", "research", "writing", "swarm", "academic"],
        "skills": "./skills/",
        "mcpServers": "./.mcp.json",
        "interface": {
            "displayName": "SophiaAgent",
            "shortDescription": "Academic research and writing swarm assistant.",
            "longDescription": (
                "SophiaAgent exposes a local MCP server so Codex can delegate "
                "research, academic writing, review, citation, data-analysis, "
                "and automatic swarm workflows."
            ),
            "developerName": "SophiaAgent",
            "category": "Productivity",
            "capabilities": ["MCP", "Research", "Writing", "Automation"],
            "websiteURL": "https://github.com/TZUKWAN/SophiaAgent",
            "defaultPrompt": [
                "Use SophiaAgent to review this paper.",
                "Use SophiaAgent to research this topic.",
                "Use SophiaAgent to draft a literature review.",
            ],
            "brandColor": "#B45309",
        },
    }


def _repo_marketplace() -> Dict[str, Any]:
    return {
        "name": "sophia-local",
        "interface": {"displayName": "SophiaAgent Local Plugins"},
        "plugins": [
            {
                "name": "sophia",
                "source": {"source": "local", "path": "./plugins/sophia"},
                "policy": {"installation": "AVAILABLE", "authentication": "ON_USE"},
                "category": "Productivity",
            }
        ],
    }
