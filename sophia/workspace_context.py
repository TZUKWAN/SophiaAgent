"""Workspace evidence collection and generated document persistence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


TEXT_SUFFIXES = {".md", ".txt", ".tex", ".rst"}
TABULAR_SUFFIXES = {".csv"}
DOCX_SUFFIXES = {".docx"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | TABULAR_SUFFIXES | DOCX_SUFFIXES | PDF_SUFFIXES
IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}


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

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidences)

    def to_prompt_block(self) -> str:
        if not self.requested:
            return ""
        lines = [
            "【Sophia 自动工作空间预读结果】",
            "用户要求基于工作空间材料回答。下面内容来自本机工作空间文件，不是网络检索结果。",
            "必须只基于这些已读取材料和后续真实工具结果写作；不得编造作者、年份、DOI、期刊或统计数字。",
            "如果材料不足以支撑某个结论或参考文献，请明确说明不足，不要补造。",
        ]
        if not self.evidences:
            lines.append("未读取到可用论文/文献文本。")
        else:
            lines.append(f"已读取 {len(self.evidences)} 个文件：")
            for idx, item in enumerate(self.evidences, 1):
                lines.append(f"\n[材料 {idx}] {item.path} ({item.kind}, {item.chars} chars)")
                if item.warning:
                    lines.append(f"读取提示：{item.warning}")
                lines.append(item.content)
        if self.skipped:
            lines.append("\n未读取或跳过的文件：")
            lines.extend(f"- {entry}" for entry in self.skipped[:20])
        return "\n".join(lines)


def needs_workspace_context(user_message: str) -> bool:
    text = user_message.lower()
    workspace_terms = ["工作空间", "workspace", "本地", "当前目录", "目录中", "文件中"]
    source_terms = ["论文", "文献", "资料", "材料", "参考文献", "文件", "仔细阅读", "基于"]
    return any(term in text for term in workspace_terms) and any(term in text for term in source_terms)


def asks_for_paper_document(user_message: str) -> bool:
    text = user_message.lower()
    return any(term in text for term in ["写", "生成", "撰写", "成文"]) and any(
        term in text for term in ["论文", "文章", "paper", "文稿"]
    )


def collect_workspace_context(
    workspace: str,
    user_message: str,
    *,
    max_files: int = 8,
    max_chars_per_file: int = 12000,
) -> WorkspaceContext:
    requested = needs_workspace_context(user_message)
    if not requested:
        return WorkspaceContext(requested=False, evidences=[], skipped=[])

    root = Path(workspace).expanduser().resolve()
    candidates = _candidate_files(root)
    ranked = sorted(candidates, key=lambda path: _rank_file(path, user_message))
    evidences: List[WorkspaceEvidence] = []
    skipped: List[str] = []
    for path in ranked:
        if len(evidences) >= max_files:
            break
        evidence = _read_evidence(path, root, max_chars_per_file)
        if evidence.content:
            evidences.append(evidence)
        else:
            skipped.append(f"{_rel(path, root)}: {evidence.warning or 'empty or unsupported'}")
    return WorkspaceContext(requested=True, evidences=evidences, skipped=skipped)


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
    for token in ["生成式", "人工智能", "中华文化", "国际传播", "文化传播", "ai", "文化"]:
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


def _read_evidence(path: Path, root: Path, max_chars: int) -> WorkspaceEvidence:
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
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[已截断，原文 {original_len} 字符]"
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
        for page in doc[:20]:
            parts.append(page.get_text("text"))
        warning = "Only the first 20 PDF pages were read." if len(doc) > 20 else ""
    return "\n".join(parts), warning


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_title(text: str) -> str:
    quoted = re.search(r"《([^》]{4,80})》", text)
    if quoted:
        return quoted.group(1)
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9，、：:（）()\s]{8,80})这个论文", text)
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
