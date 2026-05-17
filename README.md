# SophiaAgent

SophiaAgent is an AI research assistant for humanities and social science work. It combines a
ChatGPT-style web UI, a terminal CLI, research-method tools, academic writing/export tools,
citation management, review engines, memory, loops, checkpoints, and an automatic multi-agent
swarm system.

The project is designed for research workflows where a single prompt may require multiple kinds
of expertise: literature search, methodology, statistics, writing, critique, citation management,
and final synthesis.

## Highlights

- **Automatic Swarm Orchestration**: complex requests are automatically decomposed into specialist
  sub-agents. Users do not need to manually start the swarm.
- **Role-Based SubAgents**: built-in roles include literature searcher, data analyst, writer,
  reviewer, methodologist, critic, synthesizer, and citation manager.
- **Inter-Agent Collaboration**: sub-agents communicate through a thread-safe SwarmBus and later
  stages can read earlier expert outputs.
- **Tool Whitelisting**: each sub-agent only sees the tools assigned to its role.
- **Academic Research Tools**: statistics, causal inference, survey analysis, qualitative coding,
  meta-analysis, machine learning, visualization, and method discovery.
- **Empirical Workflow Orchestrator**: applied-economics, epidemiology/public-health, and
  ML-causal workflows now start from a single auditable 8-step plan, then run available stages on
  real workspace data with ResultStore lineage and explicit missing-input warnings.
- **Writing and Export**: create papers, reports, monographs, grant templates, Markdown, LaTeX,
  PDF, and DOCX outputs.
- **Citation Management**: BibTeX-like reference storage, formatting, relation tracking, and
  citation network support.
- **Web and CLI Interfaces**: use either a browser UI or an interactive terminal workflow.
- **Session Safety**: sessions, checkpoints, memory, snapshots, recovery hooks, and process
  lifecycle guards are built in.
- **OpenAI-Compatible Provider Support**: use OpenAI-compatible APIs, local vLLM/Ollama-style
  endpoints, or Anthropic with the optional extra.

## Repository

```text
https://github.com/TZUKWAN/SophiaAgent
```

## Requirements

- Python 3.10, 3.11, or 3.12
- A model provider with an OpenAI-compatible API, or Anthropic if using the `anthropic` extra
- Optional: XeLaTeX for PDF export from LaTeX
- Optional: additional scientific Python libraries for advanced methods

## Installation

### Install from GitHub

Core install:

```bash
pip install "git+https://github.com/TZUKWAN/SophiaAgent.git"
```

Recommended full install:

```bash
pip install "git+https://github.com/TZUKWAN/SophiaAgent.git#egg=sophia-agent[all]"
```

Install from a local checkout:

```bash
git clone https://github.com/TZUKWAN/SophiaAgent.git
cd SophiaAgent
pip install -e ".[all]"
```

Developer install:

```bash
pip install -e ".[dev,analysis,export,citation,data,ml,advanced,empirical]"
```

Optional empirical-method extras:

```bash
pip install -e ".[empirical]"
```

The built-in empirical workflow works without those optional packages. The extra adds richer
classical econometrics/reporting libraries such as `statsmodels`, `linearmodels`, `pyfixest`,
`stargazer`, `tableone`, and `lifelines` where the local Python environment supports them.

For heavier causal-inference stacks:

```bash
pip install -e ".[empirical-full]"
```

That optional bundle adds RD, DML, causal forest, meta-learner, fairness, and conformal-causal
libraries such as `rdrobust`, `rddensity`, `econml`, `causalml`, `doubleml`, `fairlearn`, and
`mapie`. Sophia detects these packages at runtime and reports missing pieces through
`empirical_capability_audit`.

## Configuration

SophiaAgent can be configured with environment variables or a YAML config file.

### Environment Variables

PowerShell:

```powershell
$env:SOPHIA_API_KEY="your-api-key"
$env:SOPHIA_BASE_URL="https://api.openai.com/v1"
$env:SOPHIA_MODEL="gpt-4o"
```

Bash:

```bash
export SOPHIA_API_KEY="your-api-key"
export SOPHIA_BASE_URL="https://api.openai.com/v1"
export SOPHIA_MODEL="gpt-4o"
```

For an OpenAI-compatible self-hosted or third-party endpoint, change `SOPHIA_BASE_URL` and
`SOPHIA_MODEL` to match that service.

### Config File

You can also use `config.yaml`:

```yaml
model:
  provider: openai-compat
  name: gpt-4o
  base_url: https://api.openai.com/v1
  api_key: ${SOPHIA_API_KEY}
  max_turns: 50

session:
  db_path: ~/.sophia-agent/sessions.db
  workspace: ~/SophiaWorkspace

export:
  latex_engine: xelatex
  default_format: pdf
  citation_style: gb-t-7714-2015
```

Use a custom config:

```bash
sophia --config path/to/config.yaml chat
```

## Quick Start

Start the interactive CLI:

```bash
sophia chat
```

Start the web UI:

```bash
sophia web --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

Run a one-shot prompt:

```bash
sophia exec "帮我设计一个关于数字经济与城市创新的研究方案"
```

Run a one-shot prompt from stdin:

```bash
echo "写一份关于平台经济劳动者保障的文献综述大纲" | sophia exec -
```

## Command Line Usage

```bash
sophia --help
```

Available commands:

```text
sophia chat      Interactive terminal session
sophia exec      Single-shot prompt execution
sophia tools     List or call tools
sophia serve     Start server mode
sophia integrate Auto-register with Claude Code and Codex
sophia doctor    Check installation, tools, config, and network
sophia web       Start the web UI
```

Useful examples:

```bash
sophia chat
sophia chat --model gpt-4o
sophia --workspace . chat
sophia chat --session SESSION_ID
sophia exec --json "分析这个研究问题适合什么方法"
sophia tools list --json
sophia doctor
sophia doctor --network
sophia integrate --auto
sophia --workspace . web --host 127.0.0.1 --port 8080
sophia web --host 127.0.0.1 --port 8080
```

## Health Check

After installation, run:

```bash
sophia doctor
```

This checks Python version, required dependencies, workspace read/write access,
core file/document tools, model configuration, and local Codex/Claude command
availability. Network checks are optional because external services may rate
limit or be blocked by local proxy/firewall settings:

```bash
sophia doctor --network
```

`sophia doctor --fix` applies safe fixes when possible, such as using a detected
local Ollama/OpenAI-compatible endpoint. If no model endpoint or API key exists,
SophiaAgent can still use local tools, read workspace documents, and run doctor,
but chat generation requires a real model provider. In that case SophiaAgent now
returns a clear setup message instead of crashing on first use.

## Codex and Claude Code Integration

SophiaAgent can be registered as a local MCP server for Codex and Claude Code.
After installation on any computer, run:

```bash
sophia integrate --auto
```

The command detects whether `claude` and/or `codex` are installed on `PATH`.
When Claude Code is detected, SophiaAgent registers a user-scope MCP server and
writes:

- `~/.claude/commands/sophia.md` for `/sophia ...`
- `~/.claude/skills/sophia/SKILL.md` so Claude Code can choose SophiaAgent for suitable tasks

When Codex is detected, SophiaAgent writes a local Codex plugin with MCP and skill
metadata under `~/.agents/plugins/`, and updates the local plugin marketplace.

Force writing integration files even when a client command is not currently on
`PATH`:

```bash
sophia integrate --target claude --force
sophia integrate --target codex --force
```

Project-level integration files are also included in this repository:

- `.mcp.json` exposes the `sophia` MCP server.
- `.claude/commands/sophia.md` provides `/sophia`.
- `.claude/skills/sophia/SKILL.md` describes when Claude Code should call SophiaAgent.
- `plugins/sophia/` contains the Codex plugin manifest, MCP config, and skill.
- `.agents/plugins/marketplace.json` advertises the repo-local Codex plugin.

The MCP server exposes all Sophia tools plus a unified `sophia_ask` tool. External
agents should call `sophia_ask` with the full user request; SophiaAgent then
decides internally whether to run normally or launch its automatic swarm.

By default, the generated MCP server command includes `--workspace .`, so Codex
and Claude Code bind SophiaAgent to the project directory where the coding agent
starts the MCP server. The CLI also uses the shell's current directory as the
default workspace, so `cd /path/to/project && sophia chat` binds SophiaAgent to
that project. For explicit overrides, pass `--workspace` or set `SOPHIA_WORKSPACE`:

```bash
sophia --workspace . chat
SOPHIA_WORKSPACE=/path/to/project sophia tools list
```

When a request explicitly says it is based on workspace papers, documents, or
references, SophiaAgent now performs an automatic local pre-read before calling
the model. It scans supported local files (`.pdf`, `.docx`, `.md`, `.txt`,
`.tex`, `.csv`), injects the extracted evidence into the prompt, and warns the
model not to invent citations. If the request asks for a paper or article,
SophiaAgent also saves the final answer as Markdown under
`.sophia/generated_documents/`.

### Slash Commands in Chat

Inside `sophia chat`:

| Command | Purpose |
| --- | --- |
| `/help` | Show chat commands |
| `/sessions` | List saved sessions |
| `/resume` | Resume a previous session |
| `/checkpoint [label]` | Save a checkpoint |
| `/checkpoints` | List checkpoints |
| `/tools` | List available tools |
| `/model` | Show current model |
| `/clear` | Clear terminal |
| `/quit` or `/exit` | Exit |

## Web UI

The web interface provides:

- Streaming chat output
- Markdown rendering
- KaTeX math rendering
- Code highlighting
- Session list and session history
- Settings panel for provider/model/workspace
- Token usage display
- Tool cards for tool calls and swarm lifecycle events
- Dark/light theme

Start it with:

```bash
sophia web --host 127.0.0.1 --port 8080
```

The web server runs in the foreground. Closing the terminal or stopping the process will also
clean up Sophia-managed child processes through the lifecycle guard.

## Automatic Swarm System

SophiaAgent includes an automatic swarm orchestration system. The user does not manually enable it.
When a request is simple, SophiaAgent uses the normal single-agent loop. When a request is complex,
SophiaAgent automatically starts a role-based swarm.

Examples that normally trigger the swarm:

```text
帮我写一篇关于数字经济的文献综述，要包含研究脉络、方法比较和评审意见
```

```text
分析这份数据，做描述统计、回归分析、可视化，并写成一段论文结果
```

```text
给我设计一个关于平台经济劳动者权益保障的混合研究方案，并指出风险
```

### Built-In Roles

| Role ID | Role |
| --- | --- |
| `literature_searcher` | Literature search expert |
| `data_analyst` | Statistical and data analysis expert |
| `writer` | Academic writing expert |
| `reviewer` | Academic review and quality-control expert |
| `methodologist` | Research methodology expert |
| `critic` | Logic and argumentation critic |
| `synthesizer` | Final synthesis expert |
| `citation_manager` | Citation and reference manager |

### Swarm Lifecycle

The swarm system performs:

1. Task analysis
2. Role selection
3. Task decomposition
4. Parallel or pipeline execution
5. Bus-based inter-agent communication
6. Result synthesis
7. A single final answer to the user

The web UI can show lifecycle events such as task analysis, plan creation, stage start/end,
agent completion, and synthesis.

### Manual Swarm Tools

Manual tools exist for advanced use and debugging, but they are not required for normal users:

- `swarm_delegate`
- `swarm_delegate_batch`
- `swarm_list`

Legacy sub-agent tool names are still available:

- `subagent_delegate`
- `subagent_delegate_batch`
- `subagent_list`

## Research Capabilities

SophiaAgent includes tools for:

- Research design advice
- Descriptive statistics
- T-tests, ANOVA, correlations, nonparametric tests
- Regression and causal inference
- Difference-in-differences
- Regression discontinuity
- Instrumental variables
- Propensity score matching
- Synthetic control
- Survey reliability and item analysis
- Qualitative coding and thematic analysis
- Meta-analysis and publication-bias checks
- Machine learning training, tuning, evaluation, comparison, and feature importance
- LLM evaluation and prompt testing
- Visualization and dashboards
- Method discovery and safe dependency handling

## Writing, Review, and Export

SophiaAgent can help create and manage academic documents:

- Document creation
- Outlines and section writing
- Paper/report/monograph/grant templates
- Markdown export
- LaTeX export
- PDF export
- DOCX export
- Automated academic review
- Revision from review feedback

Review dimensions include logic, citation quality, language, statistical rigor, authenticity, and
ethics-related checks.

## Citation Management

Citation tools support:

- Adding references
- Searching references
- Listing references
- Formatting references
- Adding citation relations
- Building citation networks

Supported styles include GB/T 7714 and APA-oriented workflows.

## Sessions, Memory, and Safety

SophiaAgent includes:

- SQLite-backed sessions
- Checkpoints
- Memory store and recall
- Context compression
- Tool guardrails
- Recovery hooks
- Credential failover support
- Snapshots
- Trajectory recording
- Process lifecycle cleanup for CLI and web server modes

## Project Layout

```text
sophia/
  agent.py                  Core agent loop and automatic swarm entry
  swarm/                    Automatic multi-agent swarm system
  tools/                    Tool registry and tool implementations
  research/                 Research-method engines
  review/                   Academic review engines
  exporters/                DOCX, LaTeX, PDF, OMML export support
  web/                      FastAPI web UI
  prompts/                  System prompts
  skills/                   Skill management and evolution
  lifecycle.py              Process lifecycle guards
tests/                      Test suite
config.yaml                 Example runtime configuration
pyproject.toml              Package metadata and optional extras
```

## Development

Clone and install:

```bash
git clone https://github.com/TZUKWAN/SophiaAgent.git
cd SophiaAgent
pip install -e ".[dev,analysis,export,citation,data,ml,advanced]"
```

Run tests:

```bash
pytest tests/ -v
```

Run a focused swarm test set:

```bash
pytest -q -k swarm
```

Run syntax compilation:

```bash
python -m compileall sophia tests
```

Run lint on the newest swarm/lifecycle code:

```bash
ruff check sophia/swarm sophia/lifecycle.py tests/test_swarm_*.py tests/test_lifecycle.py \
  --select E,F,W --ignore E501,E731
```

## Docker

Build and run:

```bash
docker compose up -d
```

Then open:

```text
http://localhost:8080
```

Set API credentials through environment variables or your deployment environment.

## Security Notes

- Do not commit real `.env` files or API keys.
- Use `${SOPHIA_API_KEY}` in config files instead of hardcoding secrets.
- Tool execution and method discovery include safety checks, but users should still review generated
  code, data transformations, and research conclusions.
- SophiaAgent can make mistakes. Verify important factual, legal, medical, financial, and academic
  claims before relying on them.

## Current Status

The current repository version includes the automatic swarm system and process lifecycle guards.
The local verification suite passed with:

```text
1661 passed, 6 skipped
```

## License

MIT. See `LICENSE`.
