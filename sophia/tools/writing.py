"""Document writing tool for SophiaAgent.

Manages academic document projects: papers, reports, monographs, grants.
Documents stored as JSON in workspace/.sophia/documents/.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sophia.paper_quality import (
    MIN_BODY_CHARS,
    MIN_FIGURES,
    MIN_REFERENCES,
    MIN_TABLES,
    append_quality_report_if_needed,
    quality_report_dict,
)

logger = logging.getLogger(__name__)

DOC_TYPES = ["paper", "report", "monograph", "grant-nsfc", "grant-nssfc", "grant-moe"]

DEFAULT_OUTLINES = {
    "paper": [
        "摘要",
        "引言",
        "文献综述",
        "研究方法",
        "研究结果",
        "讨论",
        "结论",
        "参考文献",
    ],
    "report": [
        "摘要",
        "研究背景",
        "研究设计",
        "数据分析",
        "主要发现",
        "政策建议",
        "附录",
    ],
    "monograph": [
        "序言",
        "第一章 导论",
        "第二章 文献综述与理论基础",
        "第三章 研究方法",
        "第四章 实证分析",
        "第五章 讨论",
        "第六章 结论",
        "参考文献",
        "索引",
    ],
    "grant-nsfc": [
        "立项依据（研究意义、国内外研究现状及发展动态）",
        "项目的研究内容、研究目标及拟解决的关键科学问题",
        "拟采取的研究方案及可行性分析",
        "本项目的特色与创新之处",
        "年度研究计划及预期研究结果",
        "研究基础与工作条件",
        "经费预算",
        "参考文献",
    ],
    "grant-nssfc": [
        "课题研究的理论意义和现实意义",
        "国内外相关研究的学术史梳理及研究动态",
        "课题研究的主要内容、基本思路、研究方法",
        "课题研究的创新之处",
        "预期成果及社会效益",
        "参考文献",
    ],
    "grant-moe": [
        "本课题国内外研究现状述评及研究意义",
        "研究的主要内容、基本思路和方法",
        "研究的重点与难点",
        "本课题的创新之处",
        "预期成果",
        "参考文献",
    ],
}


def _docs_dir(workspace: str) -> str:
    return os.path.join(workspace, ".sophia", "documents")


def _doc_path(workspace: str, doc_id: str) -> str:
    return os.path.join(_docs_dir(workspace), f"{doc_id}.json")


def _ensure_dirs(workspace: str) -> None:
    os.makedirs(_docs_dir(workspace), exist_ok=True)


def _load_doc(workspace: str, doc_id: str) -> Optional[Dict]:
    path = _doc_path(workspace, doc_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_doc(workspace: str, doc: Dict) -> None:
    _ensure_dirs(workspace)
    doc["updated_at"] = datetime.now().isoformat()
    path = _doc_path(workspace, doc["id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def doc_create(args: Dict[str, Any], workspace: str) -> str:
    """Create a new document project.

    Args: {title: str, doc_type: str, authors: str, abstract: str}
    """
    title = args.get("title", "")
    doc_type = args.get("doc_type", "paper")
    if not title:
        return json.dumps({"error": "title is required"}, ensure_ascii=False)
    if doc_type not in DOC_TYPES:
        return json.dumps(
            {"error": f"Invalid doc_type. Must be one of: {DOC_TYPES}"},
            ensure_ascii=False,
        )

    doc_id = args.get("id", uuid.uuid4().hex[:8])
    if _load_doc(workspace, doc_id):
        return json.dumps({"error": f"Document '{doc_id}' already exists"}, ensure_ascii=False)

    outline = args.get("outline", DEFAULT_OUTLINES.get(doc_type, DEFAULT_OUTLINES["paper"]))

    sections = {}
    for i, section_title in enumerate(outline):
        sections[str(i + 1)] = {
            "title": section_title,
            "content": "",
            "status": "pending",
        }

    doc = {
        "id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "authors": args.get("authors", ""),
        "abstract": args.get("abstract", ""),
        "keywords": args.get("keywords", []),
        "outline": outline,
        "sections": sections,
        "references": [],
        "quality_requirements": {
            "min_body_chars_excluding_references": MIN_BODY_CHARS,
            "min_references": MIN_REFERENCES,
            "min_tables": MIN_TABLES,
            "min_figures": MIN_FIGURES,
            "style": (
                "formal academic prose, progressive logic, short direct sentences, "
                "no fabricated citations or data"
            ),
        } if doc_type == "paper" else {},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    _save_doc(workspace, doc)
    return json.dumps({
        "action": "created",
        "id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "sections": list(outline),
    }, ensure_ascii=False)


def doc_list(args: Dict[str, Any], workspace: str) -> str:
    """List all document projects."""
    docs_dir = _docs_dir(workspace)
    if not os.path.exists(docs_dir):
        return json.dumps({"total": 0, "documents": []}, ensure_ascii=False)

    results = []
    for fname in os.listdir(docs_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(docs_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            results.append({
                "id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "doc_type": doc.get("doc_type", ""),
                "authors": doc.get("authors", ""),
                "updated_at": doc.get("updated_at", ""),
                "sections_total": len(doc.get("sections", {})),
                "sections_completed": sum(
                    1 for s in doc.get("sections", {}).values()
                    if s.get("status") == "completed"
                ),
            })
        except Exception:
            continue

    return json.dumps({"total": len(results), "documents": results}, ensure_ascii=False)


def doc_get(args: Dict[str, Any], workspace: str) -> str:
    """Get full document details.

    Args: {id: str}
    """
    doc_id = args.get("id", "")
    if not doc_id:
        return json.dumps({"error": "id is required"}, ensure_ascii=False)

    doc = _load_doc(workspace, doc_id)
    if not doc:
        return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

    return json.dumps(doc, ensure_ascii=False)


def doc_outline(args: Dict[str, Any], workspace: str) -> str:
    """Set or update document outline.

    Args: {id: str, outline: list[str], mode: "replace"|"append"|"insert"}
    """
    doc_id = args.get("id", "")
    if not doc_id:
        return json.dumps({"error": "id is required"}, ensure_ascii=False)

    doc = _load_doc(workspace, doc_id)
    if not doc:
        return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

    new_outline = args.get("outline", [])
    mode = args.get("mode", "replace")

    if mode == "replace":
        doc["outline"] = new_outline
        sections = {}
        for i, section_title in enumerate(new_outline):
            key = str(i + 1)
            if key in doc.get("sections", {}):
                sections[key] = doc["sections"][key]
                sections[key]["title"] = section_title
            else:
                sections[key] = {"title": section_title, "content": "", "status": "pending"}
        doc["sections"] = sections
    elif mode == "append":
        existing = doc.get("outline", [])
        start = len(existing) + 1
        for i, section_title in enumerate(new_outline):
            key = str(start + i)
            doc["sections"][key] = {"title": section_title, "content": "", "status": "pending"}
            existing.append(section_title)
        doc["outline"] = existing
    elif mode == "insert":
        after_section = args.get("after_section", "0")
        existing = doc.get("outline", [])
        insert_idx = int(after_section)
        for i, section_title in enumerate(new_outline):
            existing.insert(insert_idx + i, section_title)
        doc["outline"] = existing
        sections = {}
        for i, section_title in enumerate(existing):
            key = str(i + 1)
            old_sections = doc.get("sections", {})
            matching = [
                s for s in old_sections.values()
                if s.get("title") == section_title and s.get("content")
            ]
            if matching:
                sections[key] = {
                    "title": section_title,
                    "content": matching[0]["content"],
                    "status": matching[0]["status"],
                }
            else:
                sections[key] = {"title": section_title, "content": "", "status": "pending"}
        doc["sections"] = sections

    _save_doc(workspace, doc)
    return json.dumps({
        "action": "outline_updated",
        "id": doc_id,
        "outline": doc["outline"],
    }, ensure_ascii=False)


def doc_write_section(args: Dict[str, Any], workspace: str) -> str:
    """Write content to a specific section.

    Args: {id: str, section: str, content: str, status: str}
    """
    doc_id = args.get("id", "")
    section_key = args.get("section", "")
    content = args.get("content", "")

    if not doc_id:
        return json.dumps({"error": "id is required"}, ensure_ascii=False)
    if not section_key:
        return json.dumps(
            {"error": "section is required (section number or title)"},
            ensure_ascii=False,
        )

    doc = _load_doc(workspace, doc_id)
    if not doc:
        return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

    sections = doc.get("sections", {})
    # Try numeric key first, then match by title
    resolved_key = section_key
    if resolved_key not in sections:
        for k, s in sections.items():
            if s.get("title", "") == section_key or s.get("title", "").startswith(section_key):
                resolved_key = k
                break
    if resolved_key not in sections:
        available = {k: s.get("title", "") for k, s in sections.items()}
        return json.dumps(
            {"error": f"Section '{section_key}' not found. "
                     f"Available: {available}"},
            ensure_ascii=False,
        )

    sections[resolved_key]["content"] = content
    sections[resolved_key]["status"] = args.get("status", "completed")

    word_count = len(content)
    _save_doc(workspace, doc)

    return json.dumps({
        "action": "section_written",
        "id": doc_id,
        "section": resolved_key,
        "section_title": sections[resolved_key]["title"],
        "word_count": word_count,
        "status": sections[resolved_key]["status"],
    }, ensure_ascii=False)


def doc_export_markdown(args: Dict[str, Any], workspace: str) -> str:
    """Export document as Markdown text.

    Args: {id: str}
    """
    doc_id = args.get("id", "")
    if not doc_id:
        return json.dumps({"error": "id is required"}, ensure_ascii=False)

    doc = _load_doc(workspace, doc_id)
    if not doc:
        return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

    lines = []
    lines.append(f"# {doc['title']}")
    lines.append("")

    if doc.get("authors"):
        lines.append(f"**作者**: {doc['authors']}")
        lines.append("")

    if doc.get("abstract"):
        lines.append("## 摘要")
        lines.append("")
        lines.append(doc["abstract"])
        lines.append("")

    if doc.get("keywords"):
        lines.append(f"**关键词**: {', '.join(doc['keywords'])}")
        lines.append("")

    lines.append("---")
    lines.append("")

    sections = doc.get("sections", {})
    has_abstract_section = any(
        s.get("title", "").startswith("摘要") for s in sections.values()
    )
    for key in sorted(sections.keys(), key=lambda x: int(x)):
        section = sections[key]
        # Skip abstract section if we already rendered it from metadata
        if section["title"].startswith("摘要") and doc.get("abstract") and has_abstract_section:
            lines.append(f"## {section['title']}")
            lines.append("")
            if section.get("content"):
                lines.append(section["content"])
            else:
                lines.append(doc["abstract"])
            lines.append("")
            continue
        lines.append(f"## {section['title']}")
        lines.append("")
        if section.get("content"):
            lines.append(section["content"])
        else:
            lines.append("*(待撰写)*")
        lines.append("")

    if doc.get("references"):
        lines.append("## 参考文献")
        lines.append("")
        for i, ref in enumerate(doc["references"], 1):
            lines.append(f"[{i}] {ref}")
        lines.append("")

    markdown = "\n".join(lines)
    quality_report = quality_report_dict(markdown) if doc.get("doc_type") == "paper" else None
    if doc.get("doc_type") == "paper":
        markdown = append_quality_report_if_needed(f"write paper {doc.get('title', '')}", markdown)

    _ensure_dirs(workspace)
    export_path = os.path.join(_docs_dir(workspace), f"{doc_id}.md")
    with open(export_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    return json.dumps({
        "action": "exported",
        "id": doc_id,
        "format": "markdown",
        "path": export_path,
        "chars": len(markdown),
        "quality_report": quality_report,
    }, ensure_ascii=False)


def doc_quality_check(args: Dict[str, Any], workspace: str) -> str:
    """Check whether a paper document meets the minimum generation contract."""
    doc_id = args.get("id", "")
    content = args.get("content", "")
    if doc_id:
        doc = _load_doc(workspace, doc_id)
        if not doc:
            return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)
        exported = json.loads(doc_export_markdown({"id": doc_id}, workspace))
        if exported.get("path") and os.path.exists(exported["path"]):
            with open(exported["path"], "r", encoding="utf-8") as f:
                content = f.read()
    if not content:
        return json.dumps({"error": "id or content is required"}, ensure_ascii=False)
    return json.dumps(quality_report_dict(content), ensure_ascii=False, indent=2)


def register_writing_tools(registry, workspace: str, result_store=None):
    """Register document writing tools."""
    from functools import partial

    from sophia.exporters.latex_export import export_latex
    from sophia.exporters.docx_export import export_docx
    from sophia.exporters.pdf_export import export_pdf

    registry.register(
        name="doc_create",
        description=(
            "Create a new academic document project. "
            "Supported types: paper, report, monograph, grant-nsfc, grant-nssfc, grant-moe. "
            "Each type has a default outline structure."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Document title"},
                "doc_type": {
                    "type": "string",
                    "description": "Document type",
                    "default": "paper",
                    "enum": DOC_TYPES,
                },
                "authors": {"type": "string", "description": "Author names"},
                "abstract": {"type": "string", "description": "Abstract or summary"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords",
                },
                "id": {
                    "type": "string",
                    "description": "Custom document ID (auto-generated if omitted)",
                },
                "outline": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom outline (uses type default if omitted)",
                },
            },
            "required": ["title"],
        },
        handler=partial(doc_create, workspace=workspace),
    )

    registry.register(
        name="doc_list",
        description="List all document projects with summary info.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=partial(doc_list, workspace=workspace),
    )

    registry.register(
        name="doc_get",
        description="Get full details of a document project including all sections.",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
            },
            "required": ["id"],
        },
        handler=partial(doc_get, workspace=workspace),
    )

    registry.register(
        name="doc_outline",
        description=(
            "Set or update document outline. "
            "Modes: 'replace' (default), 'append' (add sections at end), "
            "'insert' (insert after a section number)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
                "outline": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New outline sections",
                },
                "mode": {
                    "type": "string",
                    "default": "replace",
                    "enum": ["replace", "append", "insert"],
                },
                "after_section": {
                    "type": "string",
                    "description": "Section number to insert after (for insert mode)",
                },
            },
            "required": ["id", "outline"],
        },
        handler=partial(doc_outline, workspace=workspace),
    )

    registry.register(
        name="doc_write_section",
        description=(
            "Write content to a document section. "
            "Section is identified by number (e.g. '1', '2'). "
            "Status can be 'draft', 'completed', or 'pending'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
                "section": {
                    "type": "string",
                    "description": (
                        "Section number (e.g. '1', '2') "
                        "or section title (e.g. '引言')"
                    ),
                },
                "content": {"type": "string", "description": "Section content (Markdown)"},
                "status": {
                    "type": "string",
                    "default": "completed",
                    "enum": ["pending", "draft", "completed"],
                },
            },
            "required": ["id", "section", "content"],
        },
        handler=partial(doc_write_section, workspace=workspace),
    )

    registry.register(
        name="doc_quality_check",
        description=(
            "Check whether an academic paper meets Sophia's minimum quality contract: "
            "6500 body characters excluding references, 20 references, 5 tables, "
            "8 figures/diagrams, and banned lazy style patterns."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document id to check"},
                "content": {
                    "type": "string",
                    "description": "Raw Markdown paper content to check if no document id is available",
                },
            },
            "required": [],
        },
        handler=partial(doc_quality_check, workspace=workspace),
    )

    registry.register(
        name="doc_export_markdown",
        description="Export a document project as a Markdown file.",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
            },
            "required": ["id"],
        },
        handler=partial(doc_export_markdown, workspace=workspace),
    )

    def _doc_export_latex(args: Dict[str, Any], workspace: str) -> str:
        doc_id = args.get("id", "")
        if not doc_id:
            return json.dumps({"error": "id is required"}, ensure_ascii=False)
        doc = _load_doc(workspace, doc_id)
        if not doc:
            return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)
        output_dir = os.path.join(_docs_dir(workspace), "export")
        try:
            result = export_latex(doc, output_dir, compile_pdf=False)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _doc_export_docx(args: Dict[str, Any], workspace: str, store=None) -> str:
        doc_id = args.get("id", "")
        if not doc_id:
            return json.dumps({"error": "id is required"}, ensure_ascii=False)
        doc = _load_doc(workspace, doc_id)
        if not doc:
            return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)
        output_dir = os.path.join(_docs_dir(workspace), "export")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{doc_id}.docx")
        try:
            result = export_docx(doc, output_path, result_store=store)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _doc_export_pdf(args: Dict[str, Any], workspace: str) -> str:
        doc_id = args.get("id", "")
        if not doc_id:
            return json.dumps({"error": "id is required"}, ensure_ascii=False)
        doc = _load_doc(workspace, doc_id)
        if not doc:
            return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)
        output_dir = os.path.join(_docs_dir(workspace), "export")
        try:
            result = export_pdf(doc, output_dir)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    registry.register(
        name="doc_export_latex",
        description="Export a document project as a LaTeX (.tex) file.",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
            },
            "required": ["id"],
        },
        handler=partial(_doc_export_latex, workspace=workspace),
    )

    registry.register(
        name="doc_export_docx",
        description="Export a document project as a Word (.docx) file.",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
            },
            "required": ["id"],
        },
        handler=partial(_doc_export_docx, workspace=workspace, store=result_store),
    )

    registry.register(
        name="doc_export_pdf",
        description=(
            "Export a document project as PDF. "
            "Requires xelatex (TeX Live or MiKTeX) to be installed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
            },
            "required": ["id"],
        },
        handler=partial(_doc_export_pdf, workspace=workspace),
    )

    def _doc_pipeline_status(args: Dict[str, Any], workspace: str) -> str:
        """Get or set the current pipeline stage for a document."""
        doc_id = args.get("id", "")
        if not doc_id:
            return json.dumps({"error": "id is required"}, ensure_ascii=False)
        doc = _load_doc(workspace, doc_id)
        if not doc:
            return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

        stage = args.get("stage", "")
        if stage:
            valid_stages = ["outline", "draft", "refine", "completed"]
            if stage not in valid_stages:
                return json.dumps(
                    {"error": f"Invalid stage. Must be: {valid_stages}"},
                    ensure_ascii=False,
                )
            doc["pipeline_stage"] = stage
            _save_doc(workspace, doc)

        return json.dumps({
            "id": doc_id,
            "pipeline_stage": doc.get("pipeline_stage", "outline"),
            "sections_total": len(doc.get("sections", {})),
            "sections_completed": sum(
                1 for s in doc.get("sections", {}).values() if s.get("status") == "completed"
            ),
        }, ensure_ascii=False)

    registry.register(
        name="doc_pipeline_status",
        description=(
            "Get or set the writing pipeline stage for a document. "
            "Stages: outline (planning), draft (writing, external tools disabled), "
            "refine (review and polish), completed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
                "stage": {
                    "type": "string",
                    "enum": ["outline", "draft", "refine", "completed"],
                    "description": "Set pipeline stage (omit to query current stage)",
                },
            },
            "required": ["id"],
        },
        handler=partial(_doc_pipeline_status, workspace=workspace),
    )

    # ------------------------------------------------------------------
    # Paper assembly & pipeline
    # ------------------------------------------------------------------

    def _doc_assemble(args: Dict[str, Any], workspace: str) -> str:
        """Assemble Methods/Results from ResultStore result_ids embedded in doc.

        Args: {id: str}
        """
        doc_id = args.get("id", "")
        if not doc_id:
            return json.dumps({"error": "id is required"}, ensure_ascii=False)
        doc = _load_doc(workspace, doc_id)
        if not doc:
            return json.dumps({"error": f"Document '{doc_id}' not found"}, ensure_ascii=False)

        try:
            from sophia.research.result_store import ResultStore
            from sophia.pipeline.assembler import PaperAssembler

            store = ResultStore(workspace)
            assembler = PaperAssembler(result_store=store)
            doc = assembler.assemble(doc)
            _save_doc(workspace, doc)

            assembled_sections = []
            for key, sec in doc.get("sections", {}).items():
                if sec.get("status") == "completed" and sec.get("content"):
                    assembled_sections.append(sec["title"])

            return json.dumps({
                "action": "assembled",
                "id": doc_id,
                "assembled_sections": assembled_sections,
                "tables": len(doc.get("_assembled_tables", [])),
            }, ensure_ascii=False)
        except Exception as e:
            logger.exception("Assembly failed")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    registry.register(
        name="doc_assemble",
        description=(
            "Auto-assemble Methods and Results sections from research result_ids "
            "embedded in the document. Requires ResultStore with stored results. "
            "Only fills empty sections to avoid overwriting user content."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
            },
            "required": ["id"],
        },
        handler=partial(_doc_assemble, workspace=workspace),
    )

    def _doc_pipeline_run(args: Dict[str, Any], workspace: str) -> str:
        """Run full paper pipeline: assemble -> review -> revise -> export DOCX.

        Args: {id: str, citation_style: str, max_iterations: int}
        """
        doc_id = args.get("id", "")
        if not doc_id:
            return json.dumps({"error": "id is required"}, ensure_ascii=False)

        try:
            from sophia.research.result_store import ResultStore
            from sophia.pipeline.loop import PaperPipeline

            store = ResultStore(workspace)
            pipeline = PaperPipeline(workspace=workspace, result_store=store)
            citation_style = args.get("citation_style", "apa7")
            result = pipeline.run(doc_id, citation_style=citation_style)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.exception("Pipeline run failed")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    registry.register(
        name="doc_pipeline_run",
        description=(
            "Run the full automated paper pipeline on a document: "
            "1) assemble Methods/Results from ResultStore, "
            "2) run six-dimension auto-review, "
            "3) apply automated fixes, "
            "4) iterate review->revise up to 3 times, "
            "5) export final DOCX. "
            "Returns full history and export path."
        ),
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Document ID"},
                "citation_style": {
                    "type": "string",
                    "default": "apa7",
                    "enum": ["apa7", "gb-t-7714-2015"],
                    "description": "Citation style",
                },
            },
            "required": ["id"],
        },
        handler=partial(_doc_pipeline_run, workspace=workspace),
    )
