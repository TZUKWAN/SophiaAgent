"""Runtime health checks for SophiaAgent installations."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from sophia.config import Config
from sophia.tools.files import register_file_tools
from sophia.tools.registry import ToolRegistry
from sophia.tools.writing import register_writing_tools


@dataclass
class DoctorCheck:
    name: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DoctorReport:
    ok: bool
    checks: List[DoctorCheck]

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}


REQUIRED_IMPORTS = {
    "yaml": "pyyaml",
    "httpx": "httpx",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "docx": "python-docx",
    "fitz": "pymupdf",
    "scipy": "scipy",
    "rich": "rich",
    "prompt_toolkit": "prompt_toolkit",
}


def run_doctor(
    config: Config,
    *,
    network: bool = False,
    fix: bool = False,
    config_path: Optional[str] = None,
) -> DoctorReport:
    checks: List[DoctorCheck] = []
    checks.append(_check_python())
    checks.extend(_check_imports())
    checks.append(_check_workspace(config))
    checks.append(_check_model_config(config, fix=fix, config_path=config_path))
    checks.append(_check_tool_smoke(config))
    checks.append(_check_integrations())
    if network:
        checks.extend(_check_network())
    ok = not any(check.status == "fail" for check in checks)
    return DoctorReport(ok=ok, checks=checks)


def render_report(report: DoctorReport) -> str:
    icons = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    lines = ["SophiaAgent Doctor", ""]
    for check in report.checks:
        lines.append(f"[{icons.get(check.status, check.status.upper())}] {check.name}: {check.message}")
        for key, value in check.details.items():
            lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Overall: " + ("OK" if report.ok else "Needs attention"))
    return "\n".join(lines)


def _check_python() -> DoctorCheck:
    version = sys.version_info
    supported = (3, 10) <= (version.major, version.minor) <= (3, 12)
    minimum_ok = (version.major, version.minor) >= (3, 10)
    status = "pass" if supported else ("warn" if minimum_ok else "fail")
    message = f"Python {platform.python_version()}"
    if status == "warn":
        message += " is newer than the CI-tested range 3.10-3.12."
    return DoctorCheck(
        "python",
        status,
        message,
        {"executable": sys.executable, "platform": platform.platform()},
    )


def _check_imports() -> List[DoctorCheck]:
    checks = []
    missing = []
    for module, package in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    if missing:
        checks.append(DoctorCheck(
            "dependencies",
            "fail",
            "Missing required Python packages.",
            {"install": f"pip install {' '.join(sorted(set(missing)))}"},
        ))
    else:
        checks.append(DoctorCheck("dependencies", "pass", "Required imports are available."))
    return checks


def _check_workspace(config: Config) -> DoctorCheck:
    workspace = Path(config.session.workspace).expanduser().resolve()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        probe = workspace / ".sophia" / "doctor_write_test.txt"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        if probe.read_text(encoding="utf-8") != "ok":
            raise OSError("write/read mismatch")
        probe.unlink(missing_ok=True)
        return DoctorCheck(
            "workspace",
            "pass",
            "Workspace is readable and writable.",
            {"workspace": str(workspace)},
        )
    except Exception as exc:
        return DoctorCheck(
            "workspace",
            "fail",
            f"Workspace is not usable: {type(exc).__name__}: {exc}",
            {"workspace": str(workspace)},
        )


def _check_model_config(
    config: Config,
    *,
    fix: bool,
    config_path: Optional[str],
) -> DoctorCheck:
    provider = config.model.provider
    if provider == "anthropic":
        if config.model.api_key:
            return DoctorCheck("model", "pass", "Anthropic provider is configured.")
        return DoctorCheck(
            "model",
            "warn",
            "Anthropic provider has no API key. Chat will not work until ANTHROPIC_API_KEY or SOPHIA_API_KEY is set.",
        )

    if provider == "openai-compat":
        if config.model.base_url and config.model.api_key:
            return DoctorCheck("model", "pass", "OpenAI-compatible provider is configured.")
        ollama = _detect_ollama()
        if ollama:
            if fix:
                _write_ollama_hint_config(config, ollama)
                _save_config(config, config_path)
                return DoctorCheck(
                    "model",
                    "pass",
                    "Detected local Ollama-compatible endpoint and wrote it to config.",
                    ollama,
                )
            return DoctorCheck(
                "model",
                "warn",
                "Detected local Ollama-compatible endpoint. Run `sophia doctor --fix` to use it by default.",
                ollama,
            )
        return DoctorCheck(
            "model",
            "warn",
            "No model endpoint/API key detected. Local tools work, but chat needs SOPHIA_BASE_URL+SOPHIA_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or a local OpenAI-compatible server.",
        )

    return DoctorCheck("model", "fail", f"Unknown provider: {provider}")


def _check_tool_smoke(config: Config) -> DoctorCheck:
    try:
        registry = ToolRegistry()
        register_file_tools(registry, config.session.workspace)
        register_writing_tools(registry, config.session.workspace)
        write_result = json.loads(registry.dispatch("file_write", {
            "path": ".sophia/doctor_tool_test.txt",
            "content": "tool-ok",
        }))
        read_result = json.loads(registry.dispatch("file_read", {
            "path": ".sophia/doctor_tool_test.txt",
        }))
        doc_result = json.loads(registry.dispatch("doc_create", {
            "id": f"doctor_{int(time.time())}",
            "title": "Doctor Test",
            "doc_type": "paper",
        }))
        if "error" in write_result or "tool-ok" not in read_result.get("content", ""):
            raise RuntimeError("file tools failed")
        if doc_result.get("action") != "created":
            raise RuntimeError("document tools failed")
        return DoctorCheck("tools", "pass", "Core file and document tools work.")
    except Exception as exc:
        return DoctorCheck(
            "tools",
            "fail",
            f"Core tool smoke test failed: {type(exc).__name__}: {exc}",
        )


def _check_integrations() -> DoctorCheck:
    detected = {
        "sophia": shutil.which("sophia") or "",
        "codex": shutil.which("codex") or "",
        "claude": shutil.which("claude") or "",
    }
    status = "pass" if detected["sophia"] else "warn"
    message = "CLI command is available." if detected["sophia"] else "sophia command was not found on PATH."
    return DoctorCheck("integrations", status, message, detected)


def _check_network() -> List[DoctorCheck]:
    checks = []
    endpoints = {
        "duckduckgo": "https://html.duckduckgo.com/html/?q=sophia",
        "crossref": "https://api.crossref.org/works?query=sophia&rows=1",
        "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/search?query=sophia&limit=1&fields=title",
        "arxiv": "https://export.arxiv.org/api/query?search_query=all:sophia&max_results=1",
    }
    for name, url in endpoints.items():
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": "SophiaAgent/0.1.0"},
                timeout=8.0,
                follow_redirects=True,
            )
            if resp.status_code == 429:
                checks.append(DoctorCheck(name, "warn", "Endpoint is reachable but rate limited.", {"status": 429}))
            elif resp.is_success:
                checks.append(DoctorCheck(name, "pass", "Endpoint is reachable.", {"status": resp.status_code}))
            else:
                checks.append(DoctorCheck(name, "warn", "Endpoint returned a non-success status.", {"status": resp.status_code}))
        except Exception as exc:
            checks.append(DoctorCheck(
                name,
                "warn",
                f"Endpoint is not reachable now: {type(exc).__name__}: {exc}",
            ))
    return checks


def _detect_ollama() -> Optional[Dict[str, Any]]:
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1.0):
            pass
        resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        models = []
        if resp.is_success:
            models = [item.get("name", "") for item in resp.json().get("models", [])]
        model = models[0] if models else "llama3.1"
        return {"base_url": "http://127.0.0.1:11434/v1", "model": model}
    except Exception:
        return None


def _write_ollama_hint_config(config: Config, ollama: Dict[str, Any]) -> None:
    config.model.provider = "openai-compat"
    config.model.base_url = ollama["base_url"]
    config.model.api_key = "ollama"
    config.model.name = ollama["model"]


def _save_config(config: Config, config_path: Optional[str]) -> None:
    import yaml

    path = Path(
        config_path
        or os.environ.get("SOPHIA_CONFIG")
        or Path.home() / ".sophia-agent" / "config.yaml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(config.to_dict(), default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
