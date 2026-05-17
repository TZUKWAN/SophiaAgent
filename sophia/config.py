"""Configuration management for SophiaAgent."""

import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


@dataclass
class ModelConfig:
    provider: str = "openai-compat"
    name: str = "Qwen3.5-122B-A10B"
    base_url: str = ""
    api_key: str = ""
    max_turns: int = 50


@dataclass
class SessionConfig:
    db_path: str = ""
    workspace: str = ""


@dataclass
class ExportConfig:
    latex_engine: str = "xelatex"
    default_format: str = "pdf"
    citation_style: str = "gb-t-7714-2015"


@dataclass
class HookConfig:
    enabled: bool = True


@dataclass
class GoalConfig:
    enabled: bool = True
    auto_decompose: bool = False


@dataclass
class MemoryConfig:
    enabled: bool = True
    auto_inject: bool = True
    max_context_entries: int = 5


@dataclass
class GuardrailConfig:
    max_consecutive_calls: int = 5
    max_calls_per_minute: int = 60


@dataclass
class LoopConfig:
    max_concurrent: int = 3


@dataclass
class ContextConfig:
    max_messages: int = 100
    compress_threshold: float = 0.65


@dataclass
class CredentialConfig:
    pool: list = field(default_factory=list)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    # New mechanism configs
    hook: HookConfig = field(default_factory=HookConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    credential: CredentialConfig = field(default_factory=CredentialConfig)

    @staticmethod
    def _resolve_env_vars(obj):
        """Recursively resolve ${ENV_VAR} references in config values."""
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
                return os.environ.get(var_name, "")
            return obj
        elif isinstance(obj, dict):
            return {k: Config._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Config._resolve_env_vars(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        """Serialize config to a plain dict suitable for YAML."""
        return {
            "model": {
                "provider": self.model.provider,
                "name": self.model.name,
                "base_url": self.model.base_url,
                "api_key": self.model.api_key,
                "max_turns": self.model.max_turns,
            },
            "session": {
                "db_path": self.session.db_path,
                "workspace": self.session.workspace,
            },
            "export": {
                "latex_engine": self.export.latex_engine,
                "default_format": self.export.default_format,
                "citation_style": self.export.citation_style,
            },
            "hook": {"enabled": self.hook.enabled},
            "goal": {
                "enabled": self.goal.enabled,
                "auto_decompose": self.goal.auto_decompose,
            },
            "memory": {
                "enabled": self.memory.enabled,
                "auto_inject": self.memory.auto_inject,
                "max_context_entries": self.memory.max_context_entries,
            },
            "guardrail": {
                "max_consecutive_calls": self.guardrail.max_consecutive_calls,
                "max_calls_per_minute": self.guardrail.max_calls_per_minute,
            },
            "loop": {"max_concurrent": self.loop.max_concurrent},
            "context": {
                "max_messages": self.context.max_messages,
                "compress_threshold": self.context.compress_threshold,
            },
            "credential": {"pool": self.credential.pool},
        }

    @classmethod
    def create_default(cls) -> "Config":
        """Create a default config with auto-detected model settings.

        Hardcoded defaults (from ModelConfig) are preserved unless env vars override them.
        """
        config = cls()

        # Env overrides take precedence over hardcoded defaults
        if os.environ.get("OPENAI_API_KEY"):
            config.model.provider = "openai"
            config.model.name = os.environ.get("SOPHIA_MODEL", "gpt-4o-mini")
            config.model.base_url = "https://api.openai.com/v1"
            config.model.api_key = os.environ.get("OPENAI_API_KEY")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            config.model.provider = "anthropic"
            config.model.name = os.environ.get("SOPHIA_MODEL", "claude-3-haiku-20240307")
            config.model.api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif os.environ.get("SOPHIA_BASE_URL"):
            config.model.base_url = os.environ["SOPHIA_BASE_URL"]
            config.model.api_key = os.environ.get("SOPHIA_API_KEY", "")
            config.model.name = os.environ.get("SOPHIA_MODEL", config.model.name)
        elif cls._local_port_open("127.0.0.1", 11434):
            config.model.base_url = "http://127.0.0.1:11434/v1"
            config.model.api_key = "ollama"
            config.model.name = os.environ.get("SOPHIA_MODEL", "llama3.1")

        home = Path.home()
        config.session.db_path = str(home / ".sophia-agent" / "sessions.db")
        config.session.workspace = str(home / "SophiaWorkspace")
        Path(config.session.workspace).mkdir(parents=True, exist_ok=True)

        return config

    @staticmethod
    def _local_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return True
        except OSError:
            return False

    @classmethod
    def load(cls, path: Optional[str] = None, workspace: Optional[str] = None) -> "Config":
        """Load config from YAML file, with env var overrides.

        If no config file exists, auto-create a default one.
        """
        # Resolve config file path
        if path is None:
            path = os.environ.get(
                "SOPHIA_CONFIG",
                str(Path.home() / ".sophia-agent" / "config.yaml"),
            )

        # Load .env file from project root (next to config.yaml)
        if HAS_DOTENV:
            project_root = str(Path(path).resolve().parent)
            env_path = os.path.join(project_root, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path, override=False)

        config_path = Path(path)
        if not config_path.exists():
            # Auto-create default config file
            config = cls.create_default()
            requested_workspace = workspace or os.environ.get("SOPHIA_WORKSPACE")
            if requested_workspace:
                config.session.workspace = str(Path(requested_workspace).expanduser().resolve())
                Path(config.session.workspace).mkdir(parents=True, exist_ok=True)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
            except Exception:
                pass  # ignore write errors, continue with in-memory config
            return config

        config = cls()
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

            # Resolve ${ENV_VAR} references in YAML values
            data = cls._resolve_env_vars(data)

            # Parse model section
            if "model" in data:
                m = data["model"]
                config.model = ModelConfig(
                    provider=m.get("provider", config.model.provider),
                    name=m.get("name", config.model.name),
                    base_url=m.get("base_url", config.model.base_url),
                    api_key=m.get("api_key", config.model.api_key),
                    max_turns=m.get("max_turns", config.model.max_turns),
                )

            # Parse session section
            if "session" in data:
                s = data["session"]
                config.session = SessionConfig(
                    db_path=s.get("db_path", ""),
                    workspace=s.get("workspace", ""),
                )

            # Parse export section
            if "export" in data:
                e = data["export"]
                config.export = ExportConfig(
                    latex_engine=e.get("latex_engine", config.export.latex_engine),
                    default_format=e.get("default_format", config.export.default_format),
                    citation_style=e.get("citation_style", config.export.citation_style),
                )

            # Parse hook section
            if "hook" in data:
                h = data["hook"]
                config.hook = HookConfig(enabled=h.get("enabled", True))

            # Parse goal section
            if "goal" in data:
                g = data["goal"]
                config.goal = GoalConfig(
                    enabled=g.get("enabled", True),
                    auto_decompose=g.get("auto_decompose", False),
                )

            # Parse memory section
            if "memory" in data:
                m = data["memory"]
                config.memory = MemoryConfig(
                    enabled=m.get("enabled", True),
                    auto_inject=m.get("auto_inject", True),
                    max_context_entries=m.get("max_context_entries", 5),
                )

            # Parse guardrail section
            if "guardrail" in data:
                g = data["guardrail"]
                config.guardrail = GuardrailConfig(
                    max_consecutive_calls=g.get("max_consecutive_calls", 5),
                    max_calls_per_minute=g.get("max_calls_per_minute", 60),
                )

            # Parse loop section
            if "loop" in data:
                loop_data = data["loop"]
                config.loop = LoopConfig(max_concurrent=loop_data.get("max_concurrent", 3))

            # Parse context section
            if "context" in data:
                c = data["context"]
                config.context = ContextConfig(
                    max_messages=c.get("max_messages", 100),
                    compress_threshold=c.get("compress_threshold", 0.8),
                )

            # Parse credential section
            if "credential" in data:
                c = data["credential"]
                config.credential = CredentialConfig(pool=c.get("pool", []))

        # Apply env var overrides
        env_api_key = os.environ.get("SOPHIA_API_KEY")
        if env_api_key:
            config.model.api_key = env_api_key
        env_base_url = os.environ.get("SOPHIA_BASE_URL")
        if env_base_url:
            config.model.base_url = env_base_url
        env_model = os.environ.get("SOPHIA_MODEL")
        if env_model:
            config.model.name = env_model
        if (
            config.model.provider == "openai-compat"
            and (not config.model.base_url or not config.model.api_key)
            and cls._local_port_open("127.0.0.1", 11434)
        ):
            config.model.base_url = "http://127.0.0.1:11434/v1"
            config.model.api_key = "ollama"
            config.model.name = os.environ.get("SOPHIA_MODEL", config.model.name or "llama3.1")

        env_workspace = os.environ.get("SOPHIA_WORKSPACE")
        if workspace:
            config.session.workspace = workspace
        elif env_workspace:
            config.session.workspace = env_workspace

        # Expand ~ in paths and set defaults
        home = Path.home()
        if config.session.db_path:
            config.session.db_path = os.path.expanduser(config.session.db_path)
        else:
            config.session.db_path = str(home / ".sophia-agent" / "sessions.db")
        if config.session.workspace:
            config.session.workspace = os.path.expanduser(config.session.workspace)
        else:
            config.session.workspace = str(home / "SophiaWorkspace")

        # Ensure workspace directory exists
        config.session.workspace = str(Path(config.session.workspace).expanduser().resolve())
        Path(config.session.workspace).mkdir(parents=True, exist_ok=True)

        return config
