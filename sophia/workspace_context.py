"""Workspace evidence collection and generated document persistence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

TEXT_SUFFIXES = {".md", ".txt", ".tex", ".rst"}
TABULAR_SUFFIXES = {".csv"}
DOCX_SUFFIXES = {".docx"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | TABULAR_SUFFIXES | DOCX_SUFFIXES | PDF_SUFFIXES
IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}
DEFAULT_PROMPT_BUDGET_CHARS = 90_000
DEFAULT_EXCERPT_CHARS = 2_200


@dataclass
class WorkspaceEvidence:
    path: str
    kind: str
    chars: int
    content: str
    warning: str = ""


@dataclass
class WorkspaceContext:
    requested: bool
    evidences: List[WorkspaceEvidence]
    skipped: List[str]
    total_candidates: int = 0

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidences)

    def to_prompt_block(
        self,
        *,
        user_message: str = "",
        max_chars: int = DEFAULT_PROMPT_BUDGET_CHARS,
        max_excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
    ) -> str:
        if not self.requested:
            return ""

        lines = [
            "[Sophia workspace pre-read result]",
            "The user asked Sophia to answer based on workspace materials. The following materials are local workspace files, not web search results.",
            "All supported files were scanned and read when possible. File count is not limited.",
            "To stay within the model context window, Sophia injects the complete file manifest plus bounded relevant excerpts instead of dumping every full document into every request.",
            "Use these workspace materials and later real tool results as evidence. Do not fabricate authors, years, DOI values, journals, page numbers, or statistics.",
            "If the workspace excerpts are insufficient for a conclusion or citation, state the gap explicitly instead of inventing evidence.",
        ]

        if not self.evidences:
            lines.append("No usable workspace paper or literature text was read.")
        else:
            lines.append(f"Read {len(self.evidences)} workspace files.")
            lines.append("\n[Workspace file manifest]")
            for idx, item in enumerate(self.evidences, 1):
                warning = f"; warning={item.warning}" if item.warning else ""
                lines.append(f"{idx}. {item.path} ({item.kind}, {item.chars} chars{warning})")

            lines.append("\n[Relevant workspace excerpts]")
            used = len("\n".join(lines))
            omitted = 0
            for idx, item in enumerate(self.evidences, 1):
                header = f"\n[Material {idx}] {item.path} ({item.kind}, {item.chars} chars)"
                excerpt_budget = min(
                    max_excerpt_chars,
                    max(0, max_chars - used - len(header) - 200),
                )
                if excerpt_budget <= 200:
                    omitted += 1
                    continue
                excerpt = _select_excerpt(item.content, user_message, excerpt_budget)
                block = [header]
                if item.warning:
                    block.append(f"Read note: {item.warning}")
                block.append(excerpt)
                block_text = "\n".join(block)
                if used + len(block_text) > max_chars:
                    omitted += 1
                    continue
                lines.append(block_text)
                used += len(block_text)
            if omitted:
                lines.append(
                    f"\n[Context budget note] {omitted} read files are included in the manifest but not embedded as excerpts in this model request."
                )

        if self.skipped:
            lines.append("\n[Skipped or unread files]")
            lines.extend(f"- {entry}" for entry in self.skipped[:20])
            if len(self.skipped) > 20:
                lines.append(f"- ... {len(self.skipped) - 20} more skipped files")
        return "\n".join(lines)


def needs_workspace_context(user_message: str) -> bool:
    text = user_message.lower()
    workspace_terms = [
        "工作空间",
        "workspace",
        "本地",
        "当前目录",
        "目录中",
        "文件中",
    ]
    source_terms = [
        "论文",
        "文献",
        "资料",
        "材料",
        "参考文献",
        "文件",
        "仔细阅读",
        "基于",
    ]
    return any(term in text for term in workspace_terms) and any(term in text for term in source_terms)


def asks_for_paper_document(user_message: str) -> bool:
    text = user_message.lower()
    write_terms = ["写", "撰写", "生成", "起草", "成文", "write", "draft", "generate"]
    paper_terms = ["论文", "文章", "研究报告", "综述", "paper", "article", "manuscript", "review"]
    if any(term in text for term in write_terms) and any(term in text for term in paper_terms):
        return True
    return any(term in text for term in ["写", "生成", "撰写", "成文", "write", "draft"]) and any(
        term in text for term in ["论文", "文章", "paper", "article", "文稿"]
    )


def collect_workspace_context(
    workspace: str,
    user_message: str,
) -> WorkspaceContext:
    final_context = WorkspaceContext(requested=False, evidences=[], skipped=[])
    for event in iter_workspace_context_events(workspace, user_message):
        if event["type"] == "workspace_context_complete":
            final_context = event["context"]
    return final_context


def iter_workspace_context_events(
    workspace: str,
    user_message: str,
) -> Generator[Dict[str, Any], None, None]:
    requested = needs_workspace_context(user_message)
    if not requested:
        yield {
            "type": "workspace_context_complete",
            "context": WorkspaceContext(requested=False, evidences=[], skipped=[]),
        }
        return

    root = Path(workspace).expanduser().resolve()
    candidates = list(_candidate_files(root))
    ranked = sorted(candidates, key=lambda path: _rank_file(path, user_message))
    evidences: List[WorkspaceEvidence] = []
    skipped: List[str] = []
    yield {
        "type": "workspace_scan_start",
        "workspace": str(root),
        "total_files": len(ranked),
    }
    for index, path in enumerate(ranked, 1):
        rel_path = _rel(path, root)
        yield {
            "type": "workspace_file_start",
            "index": index,
            "total": len(ranked),
            "path": rel_path,
            "suffix": path.suffix.lower(),
        }
        evidence = _read_evidence(path, root)
        if evidence.content:
            evidences.append(evidence)
            yield {
                "type": "workspace_file_done",
                "index": index,
                "total": len(ranked),
                "path": rel_path,
                "status": "read",
                "chars": evidence.chars,
                "warning": evidence.warning,
            }
        else:
            skipped.append(f"{_rel(path, root)}: {evidence.warning or 'empty or unsupported'}")
            yield {
                "type": "workspace_file_done",
                "index": index,
                "total": len(ranked),
                "path": rel_path,
                "status": "skipped",
                "chars": 0,
                "warning": evidence.warning or "empty or unsupported",
            }
    context = WorkspaceContext(
        requested=True,
        evidences=evidences,
        skipped=skipped,
        total_candidates=len(candidates),
    )
    yield {
        "type": "workspace_context_complete",
        "context": context,
        "read_files": len(evidences),
        "skipped_files": len(skipped),
        "total_files": len(ranked),
    }


def save_generated_markdown(workspace: str, user_message: str, content: str) -> Optional[str]:
    if not content.strip() or not asks_for_paper_document(user_message):
        return None
    title = _extract_title(user_message) or _extract_title(content) or "sophia_generated_paper"
    filename = _safe_filename(title) + ".md"
    output_dir = Path(workspace).expanduser().resolve() / ".sophia" / "generated_documents"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def _candidate_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def _rank_file(path: Path, user_message: str) -> tuple:
    name = path.name.lower()
    text = user_message.lower()
    score = 0
    for token in [
        "生成式",
        "人工智能",
        "中华文化",
        "国际传播",
        "文化传播",
        "ai",
        "文化",
    ]:
        if token.lower() in text and token.lower() in name:
            score -= 5
    if path.suffix.lower() in {".pdf", ".docx"}:
        score -= 2
    if path.suffix.lower() in TEXT_SUFFIXES:
        score -= 1
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return (score, -size, str(path))


def _read_evidence(path: Path, root: Path) -> WorkspaceEvidence:
    suffix = path.suffix.lower()
    warning = ""
    try:
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            kind = suffix[1:]
        elif suffix in TABULAR_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            kind = "csv"
        elif suffix in DOCX_SUFFIXES:
            text = _read_docx(path)
            kind = "docx"
        elif suffix in PDF_SUFFIXES:
            text, warning = _read_pdf(path)
            kind = "pdf"
        else:
            return WorkspaceEvidence(_rel(path, root), suffix, 0, "", "unsupported suffix")
    except Exception as exc:
        return WorkspaceEvidence(_rel(path, root), suffix, 0, "", f"{type(exc).__name__}: {exc}")

    text = _clean_text(text)
    original_len = len(text)
    return WorkspaceEvidence(_rel(path, root), kind, original_len, text, warning)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        return ""
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_pdf(path: Path) -> tuple[str, str]:
    try:
        import fitz  # type: ignore
    except ImportError:
        return "", "PDF reader dependency is not installed; install PyMuPDF to read PDFs."
    parts: List[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
        warning = ""
    return "\n".join(parts), warning


def _select_excerpt(text: str, query: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    tokens = _query_tokens(query)
    if not tokens:
        return text[:max_chars] + f"\n...[excerpted from {len(text)} chars]"

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return text[:max_chars] + f"\n...[excerpted from {len(text)} chars]"

    scored = []
    for index, paragraph in enumerate(paragraphs):
        lower = paragraph.lower()
        score = sum(1 for token in tokens if token in lower)
        if score:
            scored.append((score, -index, paragraph))
    selected = [item[2] for item in sorted(scored, reverse=True)]
    if not selected:
        selected = paragraphs

    chunks: List[str] = []
    used = 0
    for paragraph in selected:
        if used >= max_chars:
            break
        remaining = max_chars - used
        piece = paragraph[:remaining]
        chunks.append(piece)
        used += len(piece) + 2
    return "\n\n".join(chunks).strip() + f"\n...[excerpted from {len(text)} chars]"


def _query_tokens(query: str) -> List[str]:
    lowered = query.lower()
    latin = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", lowered)
    chinese_terms = [
        term
        for term in [
            "生成式人工智能",
            "人工智能",
            "中华文化",
            "国际传播",
            "文化传播",
            "机遇",
            "挑战",
            "路径",
            "论文",
            "文献",
        ]
        if term in query
    ]
    return list(dict.fromkeys(latin + chinese_terms))


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_title(text: str) -> str:
    quoted = re.search(r"《([^《》]{4,80})》", text)
    if quoted:
        return quoted.group(1)
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9，、：（）()\s]{8,80})这个论文", text)
    if match:
        return match.group(1).strip(" ，、：:")
    first_heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if first_heading:
        return first_heading.group(1).strip()
    return ""


def _safe_filename(title: str) -> str:
    title = re.sub(r"[\\/:*?\"<>|]+", "_", title).strip()
    title = re.sub(r"\s+", "_", title)
    return title[:80] or "sophia_generated_paper"
