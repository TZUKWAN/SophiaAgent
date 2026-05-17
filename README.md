# SophiaAgent

AI research assistant for humanities and social sciences.

## Features

- **Literature Search** -- Semantic Scholar, arXiv, Crossref
- **Academic Writing** -- Papers, reports, monographs, grant proposals (NSFC/NSSFC/MOE)
- **Citation Management** -- BibTeX library, GB/T 7714 & APA formatting, citation network graph
- **Data Analysis** -- Pandas + Matplotlib sandbox with CSV/Excel/SPSS/Stata support
- **Document Export** -- LaTeX (.tex), PDF (via XeLaTeX), Word (.docx), Markdown
- **Peer Review** -- 5-dimension weighted scoring (PRISMA 2020 systematic review)
- **Multi-Provider** -- OpenAI-compatible APIs (DeepSeek, Ollama, vLLM) + Anthropic Claude

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Configure API
cp config.yaml.example config.yaml
# Edit config.yaml with your API key and base URL

# CLI mode
sophia chat

# Web mode
sophia serve --port 8080
```

## Configuration

Create `config.yaml` (or use environment variables):

```yaml
model:
  provider: openai-compat
  name: your-model-name
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  max_turns: 50

session:
  workspace: ~/SophiaWorkspace
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SOPHIA_API_KEY` | LLM API key |
| `SOPHIA_BASE_URL` | LLM API base URL |
| `SOPHIA_MODEL` | Model name |

### Pre-configured Providers

Switch providers in `config.yaml`:

```yaml
model:
  providers:
    deepseek:
      provider: openai-compat
      name: deepseek-chat
      base_url: https://api.deepseek.com/v1
      api_key: ${DEEPSEEK_API_KEY}
    anthropic:
      provider: anthropic
      name: claude-sonnet-4-6
      api_key: ${ANTHROPIC_API_KEY}
    ollama:
      provider: openai-compat
      name: qwen2.5:7b
      base_url: http://localhost:11434/v1
      api_key: ollama
```

## CLI Commands

```bash
sophia chat                          # Interactive session
sophia chat --model deepseek-chat    # Specify model
sophia chat --session SESSION_ID     # Resume session
sophia serve --port 8080             # Web interface
sophia config set model.name MODEL   # Update config
```

### CLI Slash Commands (in chat)

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/sessions` | List saved sessions |
| `/checkpoint` | Save current checkpoint |
| `/quit` | Exit |

## Web Interface

Access at `http://localhost:8080` after running `sophia serve`.

Features: streaming output, Markdown/KaTeX rendering, dark mode, file drag-and-drop upload, citation network visualization, session management with checkpoints.

## Tool List (28 tools)

| Category | Tools |
|----------|-------|
| File | `file_read`, `file_write`, `file_list` |
| Research | `literature_search` |
| Citation | `ref_add`, `ref_list`, `ref_format`, `ref_search`, `ref_add_relation`, `ref_network` |
| Writing | `doc_create`, `doc_list`, `doc_get`, `doc_outline`, `doc_write_section`, `doc_export_markdown`, `doc_export_latex`, `doc_export_docx`, `doc_export_pdf`, `doc_pipeline_status` |
| Analysis | `data_load`, `data_describe`, `data_visualize`, `code_execute` |
| Web | `web_search`, `web_extract` |
| Review | `doc_review`, `doc_review_save`, `systematic_review` |

## Docker

```bash
docker compose up -d
```

Web interface available at `http://localhost:8080`.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check sophia/ tests/ --select E,F,W
```

## License

MIT
