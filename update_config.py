with open('D:/SophiaAgentWork/SophiaAgent/sophia/config.py', 'r', encoding='utf-8') as f:
    config_content = f.read()

# Make create_default not crash on empty/defaults
new_create_default = """    @classmethod
    def create_default(cls) -> "Config":
        \"\"\"Create a default config with auto-detected model settings.

        When no environment variables are set and no local LLM is detected,
        model fields are left empty so that `sophia init` has a chance to
        configure them interactively.  No error is raised here.
        \"\"\"
        config = cls()

        # Reset hardcoded defaults so an unconfigured install starts clean
        config.model.name = ""
        config.model.base_url = ""
        config.model.api_key = ""

        # Env overrides take precedence
        if os.environ.get("OPENAI_API_KEY"):
            config.model.provider = "openai"
            config.model.name = os.environ.get("SOPHIA_MODEL", "gpt-4o")
            config.model.base_url = "https://api.openai.com/v1"
            config.model.api_key = os.environ.get("OPENAI_API_KEY")
        elif os.environ.get("ANTHROPIC_API_KEY"):
            config.model.provider = "anthropic"
            config.model.name = os.environ.get("SOPHIA_MODEL", "claude-3-5-sonnet-20241022")
            config.model.api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif os.environ.get("SOPHIA_BASE_URL"):
            config.model.provider = "openai-compat"
            config.model.base_url = os.environ["SOPHIA_BASE_URL"]
            config.model.api_key = os.environ.get("SOPHIA_API_KEY", "")
            config.model.name = os.environ.get("SOPHIA_MODEL", "")
        elif cls._local_port_open("127.0.0.1", 11434):
            config.model.provider = "openai-compat"
            config.model.base_url = "http://127.0.0.1:11434/v1"
            config.model.api_key = "ollama"
            config.model.name = os.environ.get("SOPHIA_MODEL", "llama3.1")

        home = Path.home()
        config.session.db_path = str(home / ".sophia-agent" / "sessions.db")
        config.session.workspace = str(home / "SophiaWorkspace")
        try:
            Path(config.session.workspace).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        return config"""

# Find create_default
start_idx = config_content.find("    @classmethod\n    def create_default(cls) -> \"Config\":")
end_idx = config_content.find("    @staticmethod\n    def _local_port_open", start_idx)
config_content = config_content[:start_idx] + new_create_default + "\n\n" + config_content[end_idx:]

with open('D:/SophiaAgentWork/SophiaAgent/sophia/config.py', 'w', encoding='utf-8') as f:
    f.write(config_content)
