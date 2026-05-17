"""SophiaAgent CLI -- Rich Terminal UI.

A Claude Code-inspired TUI with academic warm-tone aesthetics.
Uses rich for rendering, prompt_toolkit for interactive input.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

from sophia.config import Config
from sophia.agent import SophiaAgent
from sophia.lifecycle import install_process_lifecycle_hooks
from sophia.session import SessionManager


# ── Theme ───────────────────────────────────────────────────────

class SophiaTheme:
    BRAND = "color(214)"
    USER_ACCENT = "color(180)"
    TOOL_BORDER = "color(110)"
    ERROR = "color(167)"
    SUCCESS = "color(107)"
    DIM = "dim"
    HEADING = f"bold {BRAND}"

    TOOL_CATEGORIES: Dict[str, str] = {
        "literature_search": "research",
        "web_search": "research",
        "web_extract": "research",
        "doc_create": "writing",
        "doc_list": "writing",
        "doc_get": "writing",
        "doc_outline": "writing",
        "doc_write_section": "writing",
        "doc_export_markdown": "writing",
        "doc_export_latex": "writing",
        "doc_export_docx": "writing",
        "doc_export_pdf": "writing",
        "doc_pipeline_status": "writing",
        "ref_add": "citation",
        "ref_list": "citation",
        "ref_format": "citation",
        "ref_search": "citation",
        "ref_add_relation": "citation",
        "ref_network": "citation",
        "data_load": "analysis",
        "data_describe": "analysis",
        "data_visualize": "analysis",
        "code_execute": "analysis",
        "file_read": "file",
        "file_write": "file",
        "file_list": "file",
        "doc_review": "review",
        "doc_review_save": "review",
        "systematic_review": "review",
    }

    CATEGORY_COLORS: Dict[str, str] = {
        "research": "color(175)",
        "writing": "color(73)",
        "citation": "color(139)",
        "analysis": "color(150)",
        "file": "color(102)",
        "review": "color(173)",
    }

    CATEGORY_LABELS: Dict[str, str] = {
        "research": "Research",
        "writing": "Writing",
        "citation": "Citation",
        "analysis": "Analysis",
        "file": "File",
        "review": "Review",
    }

    @classmethod
    def tool_color(cls, tool_name: str) -> str:
        cat = cls.TOOL_CATEGORIES.get(tool_name, "file")
        return cls.CATEGORY_COLORS.get(cat, cls.TOOL_BORDER)

    @classmethod
    def tool_category(cls, tool_name: str) -> str:
        return cls.TOOL_CATEGORIES.get(tool_name, "file")


# ── Renderer ────────────────────────────────────────────────────

class SophiaRenderer:
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._text_parts: List[str] = []
        self._live: Optional[Live] = None
        self._last_update: float = 0

    def render_banner(self, agent: SophiaAgent, session_id: Optional[str]):
        body = Text()
        body.append("SophiaAgent ", style=f"bold {SophiaTheme.BRAND}")
        body.append("v0.1.0", style="dim")
        body.append("  |  ")
        body.append(agent.config.model.name, style=SophiaTheme.BRAND)
        body.append("\n")
        body.append(f"Workspace: {agent.workspace}", style="dim")
        body.append("\n")
        body.append(f"Session: {session_id or '(new)'}", style="dim")
        body.append("\n")
        body.append("/help commands", style=SophiaTheme.BRAND)
        body.append("  ")
        body.append("Shift+Enter multiline", style="dim")

        self.console.print(Panel(
            body,
            border_style=SophiaTheme.BRAND,
            box=box.ROUNDED,
            padding=(0, 1),
        ))
        self.console.print()

    def render_user_message(self, text: str):
        self.console.print()
        msg = Text()
        msg.append("You", style=f"bold {SophiaTheme.USER_ACCENT}")
        msg.append(f": {text}")
        self.console.print(msg)
        self.console.print()

    def start_streaming(self):
        self._text_parts = []
        md = Markdown("")
        self._live = Live(md, console=self.console, refresh_per_second=8, transient=False)
        self._live.__enter__()

    def append_token(self, chunk: str):
        self._text_parts.append(chunk)
        now = time.monotonic()
        if now - self._last_update > 0.04 and self._live:
            self._last_update = now
            full = "".join(self._text_parts)
            self._live.update(Markdown(full))

    def flush_text(self):
        if self._live:
            if self._text_parts:
                full = "".join(self._text_parts)
                self._live.update(Markdown(full))
            self._live.__exit__(None, None, None)
            self._live = None
        self._text_parts = []

    def render_tool_call(self, name: str, arguments: Dict):
        self.flush_text()
        color = SophiaTheme.tool_color(name)

        title = Text()
        title.append(f"[{name}]", style=f"bold {color}")

        body_lines = []
        for key, val in arguments.items():
            val_str = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
            if len(val_str) > 120:
                val_str = val_str[:120] + "..."
            body_lines.append(f"  {key}: {val_str}")

        body_text = "\n".join(body_lines) if body_lines else "  (no arguments)"

        self.console.print(Panel(
            Text(body_text, style="dim"),
            title=title,
            border_style=color,
            box=box.ROUNDED,
            padding=(0, 1),
        ))

    def render_tool_result(self, name: str, result: str):
        color = SophiaTheme.tool_color(name)
        preview = result[:400]
        if len(result) > 400:
            preview += f"\n  ... ({len(result)} chars total)"

        self.console.print(Panel(
            Text(preview, style="dim"),
            title=Text("  result", style=color),
            border_style=color,
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        ))

    def render_error(self, exc: Exception):
        self.flush_text()
        self.console.print(Panel(
            f"{type(exc).__name__}: {exc}",
            title="[error]",
            border_style=SpinnerTheme.ERROR,
            box=box.ROUNDED,
        ))

    def render_sessions_table(self, sessions: List[Dict], current_id: Optional[str] = None):
        if not sessions:
            self.console.print(Text("  No sessions.", style="dim"))
            return
        table = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {SophiaTheme.BRAND}")
        table.add_column("ID", style=SophiaTheme.BRAND, width=10)
        table.add_column("Title", width=36)
        table.add_column("Model", style="dim", width=22)
        table.add_column("Msgs", justify="right", width=5)
        table.add_column("Updated", style="dim", width=20)
        for s in sessions:
            style = f"bold {SophiaTheme.SUCCESS}" if s["id"] == current_id else None
            table.add_row(
                s["id"],
                s["title"][:36],
                s.get("model", ""),
                str(s.get("message_count", 0)),
                s.get("updated_at", "")[:19],
                style=style,
            )
        self.console.print(table)

    def render_tools_table(self, tools: List[str]):
        table = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {SophiaTheme.BRAND}")
        table.add_column("Category", width=12)
        table.add_column("Tool", width=28)
        cats: Dict[str, List[str]] = {}
        for t in tools:
            cat = SophiaTheme.tool_category(t)
            cats.setdefault(cat, []).append(t)
        for cat_name in ["research", "writing", "citation", "analysis", "file", "review"]:
            if cat_name not in cats:
                continue
            color = SophiaTheme.CATEGORY_COLORS.get(cat_name, "")
            label = SophiaTheme.CATEGORY_LABELS.get(cat_name, cat_name)
            for i, tool in enumerate(cats[cat_name]):
                table.add_row(
                    label if i == 0 else "",
                    tool,
                    style=color,
                )
        self.console.print(table)

    def render_command_list(self):
        table = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {SophiaTheme.BRAND}")
        table.add_column("Command", style=SophiaTheme.BRAND, width=16)
        table.add_column("Description")
        commands = [
            ("/help", "Show available commands"),
            ("/sessions", "List all sessions"),
            ("/resume", "Select and resume a previous session"),
            ("/checkpoint [label]", "Save a checkpoint"),
            ("/checkpoints", "List checkpoints for current session"),
            ("/tools", "List all tools by category"),
            ("/model", "Show current model"),
            ("/clear", "Clear terminal"),
            ("/quit", "Exit SophiaAgent"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        self.console.print(table)


# ── Status Bar ──────────────────────────────────────────────────

class StatusBarRenderer:
    """Persistent status bar showing session info + real-time token usage."""

    def __init__(self, console: Console):
        self.console = console
        self.version = "0.1.0"
        self.model = ""
        self.workspace = ""
        self.session_id = ""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.progress_pct = 0.0

    def update_session(self, model: str, workspace: str, session_id: str):
        self.model = model
        self.workspace = workspace
        self.session_id = session_id or "(new)"

    def update_usage(self, usage: dict):
        self.prompt_tokens = usage.get("prompt_tokens", 0)
        self.completion_tokens = usage.get("completion_tokens", 0)
        self.total_tokens = usage.get("total_tokens", 0)

    def update_progress(self, pct: float):
        self.progress_pct = max(0.0, min(1.0, pct))

    def render(self):
        left = Text()
        left.append("SophiaAgent ", style=f"bold {SophiaTheme.BRAND}")
        left.append(f"v{self.version}", style="dim")
        left.append(" | ")
        left.append(self.model, style=SophiaTheme.BRAND)
        left.append(" | ")
        left.append(f"WS: {_shorten_path(self.workspace)}", style="dim")
        left.append(" | ")
        left.append(f"Session: {self.session_id}", style="dim")
        left.append(" | ")
        left.append("/help", style=SophiaTheme.BRAND)

        right = Text()
        if self.total_tokens > 0:
            right.append(f"Tokens: {self.total_tokens:,}", style="dim")
            right.append(f" (in:{self.prompt_tokens:,} out:{self.completion_tokens:,})", style="dim")
        else:
            right.append("Tokens: --", style="dim")
        right.append(" | ")
        bar_width = 10
        filled = int(self.progress_pct * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        right.append(f"Progress: {bar} {self.progress_pct:.0%}", style="dim")

        body = Text()
        body.append(left)
        body.append("   ")
        body.append(right)

        self.console.print(Panel(
            body,
            border_style=SophiaTheme.BRAND,
            box=box.ROUNDED,
            padding=(0, 1),
        ))


def _shorten_path(path: str, max_len: int = 30) -> str:
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3):]


# Avoid name collision with SophiaTheme.ERROR
class SpinnerTheme:
    ERROR = SophiaTheme.ERROR


# ── Slash Commands ──────────────────────────────────────────────

class SlashCommandManager:
    def __init__(self, console: Console, renderer: SophiaRenderer):
        self.console = console
        self.renderer = renderer
        self.commands: Dict[str, Dict[str, Any]] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.commands = {
            "/help": {"desc": "Show commands", "handler": self._cmd_help},
            "/sessions": {"desc": "List sessions", "handler": self._cmd_sessions},
            "/resume": {"desc": "Resume session", "handler": self._cmd_resume},
            "/checkpoint": {"desc": "Save checkpoint", "handler": self._cmd_checkpoint},
            "/checkpoints": {"desc": "List checkpoints", "handler": self._cmd_checkpoints},
            "/tools": {"desc": "List tools", "handler": self._cmd_tools},
            "/model": {"desc": "Show model", "handler": self._cmd_model},
            "/clear": {"desc": "Clear screen", "handler": self._cmd_clear},
            "/quit": {"desc": "Exit", "handler": None},
            "/exit": {"desc": "Exit", "handler": None},
        }

    def execute(self, text: str, ctx: Dict) -> Optional[bool]:
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if cmd in ("/quit", "/exit"):
            return False
        entry = self.commands.get(cmd)
        if not entry:
            self.console.print(Text(f"  Unknown: {cmd}  Type /help", style=SophiaTheme.ERROR))
            return True
        handler = entry.get("handler")
        if handler:
            handler(args, ctx)
        return True

    def get_completer(self) -> WordCompleter:
        return WordCompleter(list(self.commands.keys()), ignore_case=True)

    def _cmd_help(self, args, ctx):
        self.renderer.render_command_list()

    def _cmd_sessions(self, args, ctx):
        mgr: SessionManager = ctx["session_mgr"]
        sid = ctx.get("session_id")
        sessions = mgr.list_sessions()
        self.renderer.render_sessions_table(sessions, sid)

    def _cmd_resume(self, args, ctx):
        mgr: SessionManager = ctx["session_mgr"]
        sessions = mgr.list_sessions()

        if not sessions:
            self.console.print(Text("  No sessions available.", style="dim"))
            return

        # Show numbered session list
        table = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {SophiaTheme.BRAND}")
        table.add_column("#", width=4)
        table.add_column("ID", style=SophiaTheme.BRAND, width=10)
        table.add_column("Title", width=36)
        table.add_column("Model", style="dim", width=22)
        table.add_column("Msgs", justify="right", width=5)
        table.add_column("Updated", style="dim", width=20)
        for i, s in enumerate(sessions, 1):
            table.add_row(
                str(i), s["id"], s["title"][:36], s.get("model", ""),
                str(s.get("message_count", 0)), s.get("updated_at", "")[:19],
            )
        self.console.print(table)
        self.console.print()

        # User selects by number
        try:
            choice = input(f"  Select session (1-{len(sessions)}, 0 to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return

        if not choice.isdigit() or int(choice) == 0:
            self.console.print(Text("  Cancelled.", style="dim"))
            return

        idx = int(choice) - 1
        if idx < 0 or idx >= len(sessions):
            self.console.print(Text("  Invalid selection.", style=SophiaTheme.ERROR))
            return

        target = sessions[idx]
        target_id = target["id"]

        existing = mgr.get_session(target_id)
        if not existing:
            self.console.print(Text(f"  Session not found: {target_id}", style=SophiaTheme.ERROR))
            return

        messages = existing.get("messages", [])

        # Rebuild history from stored messages
        history: List[Dict] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant", "system"):
                history.append({"role": role, "content": content})
            elif role == "tool":
                history.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content,
                })

        # Reset agent token counters for new session
        agent: SophiaAgent = ctx["agent"]
        agent.reset_session_tokens()

        ctx["session_id"] = target_id
        ctx["history"] = history

        self.console.print(Text(
            f"  Resumed: {target_id} ({len(messages)} messages)",
            style=SophiaTheme.SUCCESS,
        ))
        self.console.print(Text(f"  Title: {existing.get('title', '')}", style="dim"))

    def _cmd_checkpoint(self, args, ctx):
        mgr: SessionManager = ctx["session_mgr"]
        sid = ctx.get("session_id")
        if not sid:
            self.console.print(Text("  No active session", style="dim"))
            return
        label = args.strip() or "manual"
        cp_id = mgr.save_checkpoint(sid, label)
        self.console.print(Text(f"  Checkpoint #{cp_id} saved: {label}", style=SophiaTheme.SUCCESS))

    def _cmd_checkpoints(self, args, ctx):
        mgr: SessionManager = ctx["session_mgr"]
        sid = ctx.get("session_id")
        if not sid:
            self.console.print(Text("  No active session", style="dim"))
            return
        cps = mgr.list_checkpoints(sid)
        if not cps:
            self.console.print(Text("  No checkpoints", style="dim"))
            return
        table = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {SophiaTheme.BRAND}")
        table.add_column("#", width=5)
        table.add_column("Label", width=20)
        table.add_column("Created", style="dim")
        for cp in cps:
            table.add_row(str(cp["id"]), cp["label"], cp["created_at"][:19])
        self.console.print(table)

    def _cmd_tools(self, args, ctx):
        agent: SophiaAgent = ctx["agent"]
        self.renderer.render_tools_table(agent.tools.list_tools())

    def _cmd_model(self, args, ctx):
        agent: SophiaAgent = ctx["agent"]
        self.console.print(Text(f"  {agent.config.model.name}", style=SophiaTheme.BRAND))
        self.console.print(Text(f"  Provider: {agent.config.model.provider}", style="dim"))

    def _cmd_clear(self, args, ctx):
        self.console.clear()


# ── Prompt ──────────────────────────────────────────────────────

class SophiaPromptSession:
    def __init__(self, slash_mgr: SlashCommandManager):
        self.slash_mgr = slash_mgr
        self._has_prompt_toolkit = sys.stdin.isatty()
        if self._has_prompt_toolkit:
            try:
                self.session = PromptSession(
                    multiline=False,
                    completer=slash_mgr.get_completer(),
                    complete_while_typing=True,
                )
            except Exception:
                self._has_prompt_toolkit = False

    def prompt(self) -> Optional[str]:
        if self._has_prompt_toolkit:
            try:
                text = self.session.prompt(
                    [("class:prompt", "sophia> ")],
                    style=_prompt_style(),
                )
                return text.strip() if text else None
            except (EOFError, KeyboardInterrupt):
                return None
        else:
            try:
                text = input("sophia> ")
                return text.strip() if text else None
            except (EOFError, KeyboardInterrupt):
                return None


def _prompt_style():
    from prompt_toolkit.styles import Style
    return Style.from_dict({
        "prompt": "bold #ffaf00",
    })


def _default_workspace_override() -> str:
    """Use the caller's current directory as the CLI default workspace."""
    return os.getcwd()


# ── Chat Command ────────────────────────────────────────────────

def cmd_chat(args):
    install_process_lifecycle_hooks()
    config = Config.load(args.config, workspace=args.workspace)
    if args.model:
        config.model.name = args.model
    agent = SophiaAgent(config)
    session_mgr = SessionManager(config.session.db_path)

    console = Console()
    renderer = SophiaRenderer(console)
    status_bar = StatusBarRenderer(console)
    status_bar.update_session(config.model.name, config.session.workspace, None)
    slash_mgr = SlashCommandManager(console, renderer)
    prompt_sess = SophiaPromptSession(slash_mgr)

    session_id = getattr(args, "session", None)
    history: List[Dict] = []

    if session_id:
        existing = session_mgr.get_session(session_id)
        if existing:
            history = existing.get("messages", [])
            status_bar.update_session(config.model.name, config.session.workspace, session_id)
            console.print(Text(
                f"  Resumed: {session_id} ({len(history)} messages)",
                style=SophiaTheme.SUCCESS,
            ))
        else:
            console.print(Text(f"  Session not found: {session_id}", style="dim"))
            session_id = None

    status_bar.render()

    ctx = {
        "agent": agent,
        "session_mgr": session_mgr,
        "session_id": session_id,
        "history": history,
        "status_bar": status_bar,
    }

    while True:
        user_input = prompt_sess.prompt()
        if user_input is None:
            console.print(Text("\n  Goodbye.", style=SophiaTheme.BRAND))
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            result = slash_mgr.execute(user_input, ctx)
            if result is False:
                console.print(Text("\n  Goodbye.", style=SophiaTheme.BRAND))
                break
            # Re-render status bar after slash commands (session may have changed)
            status_bar.update_session(
                config.model.name, config.session.workspace,
                ctx.get("session_id"),
            )
            status_bar.render()
            continue

        # Create session on first message
        if not ctx["session_id"]:
            ctx["session_id"] = session_mgr.create_session(
                title=user_input[:50], model=config.model.name,
            )
            status_bar.update_session(
                config.model.name, config.session.workspace, ctx["session_id"],
            )

        renderer.render_user_message(user_input)

        try:
            final_response = ""
            for event in agent.run_stream(user_input, history=ctx["history"]):
                etype = event.get("type")

                if etype == "token":
                    if not renderer._live:
                        renderer.start_streaming()
                    renderer.append_token(event["content"])

                elif etype == "tool_call":
                    renderer.render_tool_call(
                        event["name"],
                        event.get("arguments", {}),
                    )

                elif etype == "tool_result":
                    renderer.render_tool_result(
                        event["name"],
                        event.get("result", ""),
                    )

                elif etype == "done":
                    renderer.flush_text()
                    final_response = event.get("response", "")
                    ctx["history"] = event.get("history", ctx["history"])
                    if event.get("usage"):
                        status_bar.update_usage(event["usage"])

            session_mgr.add_messages_batch(ctx["session_id"], [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": final_response},
            ])

            # Re-render status bar after each round
            status_bar.render()
            console.print()

        except KeyboardInterrupt:
            renderer.flush_text()
            console.print(Text("\n  Interrupted.", style=SophiaTheme.ERROR))
            console.print()
        except Exception as e:
            renderer.render_error(e)
            logging.exception("Chat error")
            console.print()


# ── Agent Factory ───────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _create_agent(config_path=None, model=None, workspace=None) -> SophiaAgent:
    config = Config.load(config_path, workspace=workspace)
    if model:
        config.model.name = model
    return SophiaAgent(config)


MCP_SOPHIA_ASK_TOOL = {
    "name": "sophia_ask",
    "description": (
        "Delegate a natural-language research, writing, review, data analysis, "
        "or workflow request to SophiaAgent. SophiaAgent will decide whether "
        "to run its automatic swarm internally and return one final answer."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The user's full request for SophiaAgent.",
            }
        },
        "required": ["prompt"],
    },
}


def _mcp_tools_for_agent(agent: SophiaAgent) -> List[Dict[str, Any]]:
    """Return MCP-compatible tools, including the natural-language Sophia entrypoint."""
    tools = [MCP_SOPHIA_ASK_TOOL]
    for schema in agent.tools.get_schemas():
        fn = schema["function"]
        if fn["name"] == MCP_SOPHIA_ASK_TOOL["name"]:
            continue
        tools.append({
            "name": fn["name"],
            "description": fn["description"],
            "inputSchema": fn["parameters"],
        })
    return tools


def _call_mcp_tool(agent: SophiaAgent, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == MCP_SOPHIA_ASK_TOOL["name"]:
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return {"error": "sophia_ask requires a non-empty prompt"}
        return {"response": agent.run(prompt), "tool": name}

    result_str = agent.tools.dispatch(name, arguments)
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        return {"text": result_str}


def _mcp_prompts_list() -> List[Dict[str, Any]]:
    return [
        {
            "name": "sophia",
            "description": "Delegate a request to SophiaAgent",
            "arguments": [
                {"name": "request", "description": "Request for SophiaAgent", "required": True}
            ],
        },
        {
            "name": "research",
            "description": "Run a SophiaAgent research workflow",
            "arguments": [{"name": "topic", "description": "Research topic", "required": True}],
        },
        {
            "name": "paper",
            "description": "Create an academic paper with SophiaAgent",
            "arguments": [
                {"name": "title", "description": "Paper title", "required": True},
                {"name": "type", "description": "Document type", "required": False},
            ],
        },
    ]


def _mcp_prompt_messages(name: str, arguments: Dict[str, Any]) -> List[Dict[str, str]]:
    if name == "sophia":
        request = arguments.get("request", "")
        content = (
            "Use the SophiaAgent MCP tool `sophia_ask` to handle this request "
            f"end-to-end, and return SophiaAgent's final answer:\n\n{request}"
        )
    elif name == "research":
        topic = arguments.get("topic", "")
        content = (
            "Use the SophiaAgent MCP tool `sophia_ask` to research this topic, "
            f"collect credible sources, synthesize findings, and cite limitations:\n\n{topic}"
        )
    elif name == "paper":
        title = arguments.get("title", "")
        doc_type = arguments.get("type", "paper")
        content = (
            "Use the SophiaAgent MCP tool `sophia_ask` to create a "
            f"{doc_type} titled `{title}` with outline, argument structure, "
            "and academically appropriate prose."
        )
    else:
        return []
    return [{"role": "user", "content": content}]


# ── Exec Command ────────────────────────────────────────────────

def cmd_exec(args):
    install_process_lifecycle_hooks()
    agent = _create_agent(args.config, args.model, args.workspace)
    prompt = args.prompt
    if prompt == "-" or (not prompt and not sys.stdin.isatty()):
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Error: no prompt provided", file=sys.stderr)
        sys.exit(1)
    try:
        if args.json:
            result = _exec_json(agent, prompt, args.max_turns)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            response = agent.run(prompt)
            print(response)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
        else:
            print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


def _exec_json(agent: SophiaAgent, prompt: str, max_turns: int = 50) -> Dict:
    from sophia.prompts.system import build_system_prompt
    system_prompt = build_system_prompt(agent.workspace)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    tool_trace = []
    tool_schemas = agent.tools.get_schemas() or None
    for turn in range(max_turns):
        response = agent.provider.chat(messages=messages, tools=tool_schemas)
        messages.append(response.to_dict())
        if not response.tool_calls:
            return {
                "response": response.content or "",
                "tool_calls": tool_trace,
                "turns": turn + 1,
            }
        for tc in response.tool_calls:
            result_str = agent.tools.dispatch(tc.name, tc.arguments)
            tool_trace.append({
                "turn": turn + 1,
                "tool": tc.name,
                "arguments": tc.arguments,
                "result_preview": result_str[:500],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })
    return {
        "response": messages[-1].get("content", ""),
        "tool_calls": tool_trace,
        "turns": max_turns,
        "truncated": True,
    }


# ── Tools Command ───────────────────────────────────────────────

def cmd_tools_list(args):
    install_process_lifecycle_hooks()
    agent = _create_agent(args.config, args.model, args.workspace)
    if args.json:
        schemas = agent.tools.get_schemas()
        print(json.dumps({
            "tools": [
                {
                    "name": s["function"]["name"],
                    "description": s["function"]["description"],
                    "parameters": s["function"]["parameters"],
                }
                for s in schemas
            ]
        }, ensure_ascii=False, indent=2))
    else:
        console = Console()
        renderer = SophiaRenderer(console)
        renderer.render_tools_table(agent.tools.list_tools())


def cmd_tools_call(args):
    install_process_lifecycle_hooks()
    agent = _create_agent(args.config, args.model, args.workspace)
    tool_name = args.tool_name
    if tool_name not in agent.tools._tools:
        print(f"Error: unknown tool '{tool_name}'", file=sys.stderr)
        print(f"Available: {', '.join(agent.tools.list_tools())}", file=sys.stderr)
        sys.exit(1)
    if args.args_json:
        try:
            tool_args = json.loads(args.args_json)
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON arguments: {e}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        try:
            tool_args = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON from stdin: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        tool_args = {}
    result = agent.tools.dispatch(tool_name, tool_args)
    print(result)


# ── Serve Command ───────────────────────────────────────────────

def cmd_serve(args):
    if args.stdio:
        _serve_stdio(args)
    else:
        _serve_http(args)


def cmd_integrate(args):
    """Install SophiaAgent integrations for local coding agents."""
    from sophia.integrations import install_all, install_claude_code, install_codex

    if args.target == "claude":
        results = [install_claude_code(force=args.force)]
    elif args.target == "codex":
        results = [install_codex(force=args.force)]
    else:
        results = install_all(force=args.force)

    exit_code = 0
    for result in results:
        status = "installed" if result.installed else "skipped"
        detected = "detected" if result.detected else "not detected"
        print(f"{result.client}: {status} ({detected})")
        for path in result.paths:
            print(f"  wrote: {path}")
        for message in result.messages:
            print(f"  {message}")
        for error in result.errors:
            print(f"  error: {error}", file=sys.stderr)
            exit_code = 1
    if exit_code:
        sys.exit(exit_code)


def cmd_doctor(args):
    """Run installation and runtime health checks."""
    from sophia.doctor import render_report, run_doctor

    config = Config.load(args.config, workspace=args.workspace)
    report = run_doctor(
        config,
        network=args.network,
        fix=args.fix,
        config_path=args.config,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_report(report))
    if not report.ok and args.strict:
        sys.exit(1)


def cmd_web(args):
    """Start SophiaAgent web UI server."""
    install_process_lifecycle_hooks()
    import uvicorn
    from sophia.web import create_app
    config = Config.load(args.config, workspace=args.workspace)
    app = create_app(config)
    port = args.port or 8080
    host = args.host or "0.0.0.0"

    console = Console()
    body = Text()
    body.append("SophiaAgent Web", style=f"bold {SophiaTheme.BRAND}")
    body.append("\n")
    body.append("  Model: ", style="dim")
    body.append(config.model.name, style=SophiaTheme.BRAND)
    body.append("\n")
    body.append(f"  Workspace: {config.session.workspace}", style="dim")
    body.append("\n")
    body.append(f"  URL: http://{host}:{port}", style=SophiaTheme.BRAND)
    console.print(Panel(
        body,
        border_style=SophiaTheme.BRAND,
        box=box.ROUNDED,
        padding=(0, 1),
    ))
    console.print()

    uvicorn.run(app, host=host, port=port, log_level="info")


def _serve_http(args):
    install_process_lifecycle_hooks()
    import uvicorn
    from sophia.web import create_app
    config = Config.load(args.config, workspace=args.workspace)
    app = create_app(config)
    port = args.port or 8080
    host = args.host or "0.0.0.0"
    print("\nSophiaAgent Web Server")
    print(f"  Model: {config.model.name}")
    print(f"  Workspace: {config.session.workspace}")
    print(f"  URL: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


def _serve_stdio(args):
    install_process_lifecycle_hooks()
    agent = _create_agent(args.config, args.model, args.workspace)
    _initialized = False

    def _send(obj: Dict):
        line = json.dumps(obj, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _send_error(req_id, code, message):
        _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

    def _handle_initialize(params):
        nonlocal _initialized
        _initialized = True
        return {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
                "logging": {},
            },
            "serverInfo": {"name": "SophiaAgent", "version": "0.1.0"},
        }

    def _handle_tools_list(params):
        return {"tools": _mcp_tools_for_agent(agent)}

    def _handle_tools_call(params):
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        result_data = _call_mcp_tool(agent, name, arguments)
        return {
            "content": [{"type": "text", "text": json.dumps(result_data, ensure_ascii=False)}],
            "isError": "error" in result_data if isinstance(result_data, dict) else False,
        }

    def _handle_resources_list(params):
        return {
            "resources": [
                {"uri": "sophia://config", "name": "Configuration",
                 "description": "Current config", "mimeType": "application/json"},
                {"uri": "sophia://tools", "name": "Tools",
                 "description": "Available tools", "mimeType": "application/json"},
            ]
        }

    def _handle_resources_read(params):
        uri = params.get("uri", "")
        if uri == "sophia://config":
            content = json.dumps({
                "model": agent.config.model.name,
                "provider": agent.config.model.provider,
                "workspace": agent.workspace,
                "tools": [tool["name"] for tool in _mcp_tools_for_agent(agent)],
            }, ensure_ascii=False, indent=2)
        elif uri == "sophia://tools":
            content = json.dumps({
                "tools": _mcp_tools_for_agent(agent)
            }, ensure_ascii=False, indent=2)
        else:
            return {"contents": []}
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": content}]}

    def _handle_prompts_list(params):
        return {"prompts": _mcp_prompts_list()}

    def _handle_prompts_get(params):
        name = params.get("name", "")
        a = params.get("arguments", {})
        msgs = _mcp_prompt_messages(name, a)
        return {"description": f"Prompt: {name}", "messages": msgs}

    METHOD_HANDLERS = {
        "initialize": _handle_initialize,
        "tools/list": _handle_tools_list,
        "tools/call": _handle_tools_call,
        "resources/list": _handle_resources_list,
        "resources/read": _handle_resources_read,
        "prompts/list": _handle_prompts_list,
        "prompts/get": _handle_prompts_get,
    }
    NOTIFICATION_METHODS = {"notifications/initialized", "notifications/cancelled"}
    print("SophiaAgent MCP Server (stdio) ready", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            _send_error(None, -32700, f"Parse error: {e}")
            continue
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})
        if method in NOTIFICATION_METHODS:
            continue
        if method not in METHOD_HANDLERS:
            _send_error(req_id, -32601, f"Method not found: {method}")
            continue
        try:
            result = METHOD_HANDLERS[method](params)
            _send({"jsonrpc": "2.0", "id": req_id, "result": result})
        except Exception as e:
            logging.exception("Handler error for %s", method)
            _send_error(req_id, -32603, f"Internal error: {type(e).__name__}: {e}")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SophiaAgent -- Humanities & Social Science Research Assistant",
        prog="sophia",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument(
        "--workspace",
        default=_default_workspace_override(),
        help="Override workspace directory. Use '.' to bind Sophia to the current project.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_chat = subparsers.add_parser("chat", help="Interactive REPL")
    p_chat.add_argument("--session", "-s", help="Resume session by ID")
    p_chat.set_defaults(func=cmd_chat)

    p_exec = subparsers.add_parser("exec", help="Single-shot execution")
    p_exec.add_argument("prompt", nargs="?", help='Prompt (use "-" for stdin)')
    p_exec.add_argument("--json", action="store_true", help="JSON output")
    p_exec.add_argument("--max-turns", type=int, default=50)
    p_exec.set_defaults(func=cmd_exec)

    p_tools = subparsers.add_parser("tools", help="Tool management")
    tools_sub = p_tools.add_subparsers(dest="tools_command")
    p_list = tools_sub.add_parser("list", help="List all tools")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_tools_list)
    p_call = tools_sub.add_parser("call", help="Call a tool")
    p_call.add_argument("tool_name")
    p_call.add_argument("--args", dest="args_json")
    p_call.set_defaults(func=cmd_tools_call)

    p_serve = subparsers.add_parser("serve", help="Start server")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--stdio", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_integrate = subparsers.add_parser(
        "integrate",
        help="Auto-register SophiaAgent with Claude Code and Codex",
    )
    p_integrate.add_argument(
        "--target",
        choices=["auto", "claude", "codex"],
        default="auto",
        help="Client integration to install",
    )
    p_integrate.add_argument(
        "--force",
        action="store_true",
        help="Write integration files even if the client command is not detected",
    )
    p_integrate.add_argument(
        "--auto",
        action="store_true",
        help="Alias for the default automatic detection mode",
    )
    p_integrate.set_defaults(func=cmd_integrate)

    p_doctor = subparsers.add_parser(
        "doctor",
        help="Check SophiaAgent installation, configuration, tools, and network",
    )
    p_doctor.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    p_doctor.add_argument("--network", action="store_true", help="Check external network services")
    p_doctor.add_argument("--fix", action="store_true", help="Apply safe automatic fixes when possible")
    p_doctor.add_argument("--strict", action="store_true", help="Exit non-zero when a required check fails")
    p_doctor.set_defaults(func=cmd_doctor)

    p_web = subparsers.add_parser("web", help="Start web UI")
    p_web.add_argument("--port", type=int, default=8080)
    p_web.add_argument("--host", default="0.0.0.0")
    p_web.set_defaults(func=cmd_web)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        args.func = cmd_chat

    args.func(args)


if __name__ == "__main__":
    main()
