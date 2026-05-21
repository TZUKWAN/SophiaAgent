"""Reading and notes tools for SophiaAgent.

Registers tools for paper reading, annotation extraction, note management,
and literature graph analysis.
"""

import json
import logging
from typing import Any, Dict

from sophia.research.literature_graph import LiteratureGraph
from sophia.research.notes import ZettelkastenStore
from sophia.research.reader import PaperReader

logger = logging.getLogger(__name__)


def _json_str(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


# =====================================================================
# PaperReader tools
# =====================================================================

def paper_extract_elements(args: Dict[str, Any]) -> str:
    """Extract key elements from paper text.

    Args: {text: str}
    """
    text = args.get("text", "")
    if not text:
        return _json_str({"error": "text is required"})

    reader = PaperReader()
    result = reader.extract_key_elements(text)
    return _json_str(result)


def paper_extract_annotations(args: Dict[str, Any]) -> str:
    """Extract annotations from a PDF file.

    Args: {pdf_path: str}
    """
    pdf_path = args.get("pdf_path", "")
    if not pdf_path:
        return _json_str({"error": "pdf_path is required"})

    reader = PaperReader()
    result = reader.extract_annotations(pdf_path)
    return _json_str(result)


def paper_compare(args: Dict[str, Any]) -> str:
    """Compare multiple papers by their extracted elements.

    Args: {elements_list: list[dict]}
    """
    elements_list = args.get("elements_list", [])
    if not elements_list:
        return _json_str({"error": "elements_list is required"})

    reader = PaperReader()
    result = reader.compare_papers(elements_list)
    return _json_str(result)


# =====================================================================
# ZettelkastenStore tools
# =====================================================================

def note_create(args: Dict[str, Any], workspace: str) -> str:
    """Create a new note card.

    Args: {title, content, note_type, tags, links, source_type, source_id}
    """
    title = args.get("title", "")
    content = args.get("content", "")
    if not title or not content:
        return _json_str({"error": "title and content are required"})

    store = ZettelkastenStore(workspace)
    result = store.create(
        title=title,
        content=content,
        note_type=args.get("note_type", "concept"),
        tags=args.get("tags"),
        links=args.get("links"),
        source_type=args.get("source_type", ""),
        source_id=args.get("source_id", ""),
    )
    return _json_str(result)


def note_search(args: Dict[str, Any], workspace: str) -> str:
    """Search notes by query, tags, linked_to, or note_type.

    Args: {query, tags, linked_to, note_type}
    """
    store = ZettelkastenStore(workspace)
    results = store.search(
        query=args.get("query", ""),
        tags=args.get("tags"),
        linked_to=args.get("linked_to"),
        note_type=args.get("note_type"),
    )
    return _json_str({"count": len(results), "results": results})


def note_link(args: Dict[str, Any], workspace: str) -> str:
    """Add or update links between notes.

    Args: {note_id, links: list[str]}
    """
    note_id = args.get("note_id", "")
    links = args.get("links", [])
    if not note_id:
        return _json_str({"error": "note_id is required"})

    store = ZettelkastenStore(workspace)
    result = store.update(note_id=note_id, links=links)
    return _json_str(result)


def note_graph(args: Dict[str, Any], workspace: str) -> str:
    """Get the note link graph as {nodes, edges}.

    Args: {}
    """
    store = ZettelkastenStore(workspace)
    result = store.get_link_graph()
    return _json_str(result)


def note_from_paper(args: Dict[str, Any], workspace: str) -> str:
    """Auto-generate an evidence note from paper extraction results.

    Args: {elements: dict, paper_title, paper_id}
    """
    elements = args.get("elements", {})
    if not elements:
        return _json_str({"error": "elements is required"})

    store = ZettelkastenStore(workspace)
    result = store.from_paper_elements(
        elements=elements,
        paper_title=args.get("paper_title", ""),
        paper_id=args.get("paper_id", ""),
    )
    return _json_str(result)


# =====================================================================
# LiteratureGraph tools
# =====================================================================

def literature_graph_build(args: Dict[str, Any], workspace: str) -> str:
    """Build a literature citation graph.

    Args: {literature_ids: list[str] (optional)}
    """
    graph_mgr = LiteratureGraph(workspace)
    try:
        G = graph_mgr.build_graph(literature_ids=args.get("literature_ids"))
        stats = graph_mgr.graph_stats(G)
        stats["visualization_mermaid"] = graph_mgr.visualize(G, format="mermaid")
        return _json_str(stats)
    except Exception as e:
        logger.exception("literature_graph_build failed")
        return _json_str({"error": str(e)})


def literature_graph_visualize(args: Dict[str, Any], workspace: str) -> str:
    """Visualize the literature graph in mermaid, tikz, or dot.

    Args: {format: str, literature_ids: list[str] (optional)}
    """
    fmt = args.get("format", "mermaid")
    graph_mgr = LiteratureGraph(workspace)
    try:
        G = graph_mgr.build_graph(literature_ids=args.get("literature_ids"))
        viz = graph_mgr.visualize(G, format=fmt)
        return _json_str({"format": fmt, "visualization": viz})
    except Exception as e:
        logger.exception("literature_graph_visualize failed")
        return _json_str({"error": str(e)})


def literature_graph_clusters(args: Dict[str, Any], workspace: str) -> str:
    """Detect clusters in the literature graph.

    Args: {literature_ids: list[str] (optional)}
    """
    graph_mgr = LiteratureGraph(workspace)
    try:
        G = graph_mgr.build_graph(literature_ids=args.get("literature_ids"))
        clusters = graph_mgr.detect_clusters(G)
        key_papers = graph_mgr.find_key_papers(G)
        return _json_str({
            "clusters": clusters,
            "key_papers": key_papers[:10],
            "stats": graph_mgr.graph_stats(G),
        })
    except Exception as e:
        logger.exception("literature_graph_clusters failed")
        return _json_str({"error": str(e)})


# =====================================================================
# Registration
# =====================================================================

def register_reading_tools(registry, workspace: str):
    """Register all reading, notes, and literature graph tools."""
    from functools import partial

    # PaperReader tools
    registry.register(
        name="paper_extract_elements",
        description=(
            "Extract key elements from academic paper text: research questions, "
            "core arguments, methods, data sources, main findings, limitations, "
            "theoretical framework, and sample size. Uses LLM if available, "
            "otherwise keyword-based fallback."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Full or partial paper text to analyze",
                },
            },
            "required": ["text"],
        },
        handler=paper_extract_elements,
    )

    registry.register(
        name="paper_extract_annotations",
        description=(
            "Extract highlights, underlines, and text annotations from a PDF. "
            "Returns annotation list with page, rect, content, type, "
            "surrounding context, and color grouping. Also exports Markdown."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the PDF file",
                },
            },
            "required": ["pdf_path"],
        },
        handler=paper_extract_annotations,
    )

    registry.register(
        name="paper_compare",
        description=(
            "Compare multiple papers by their extracted elements. "
            "Returns cross-paper comparison matrix and auto-detected "
            "consensus/controversies across dimensions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "elements_list": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of extracted element dicts from paper_extract_elements",
                },
            },
            "required": ["elements_list"],
        },
        handler=paper_compare,
    )

    # ZettelkastenStore tools
    registry.register(
        name="note_create",
        description=(
            "Create an atomic note card in the Zettelkasten store. "
            "Types: concept (definition+example+links), "
            "evidence (source+data+method+conclusion), "
            "comment (personal insight+argument+critique)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title"},
                "content": {"type": "string", "description": "Note content (Markdown supported, use [[note_id]] for links)"},
                "note_type": {
                    "type": "string",
                    "enum": ["concept", "evidence", "comment"],
                    "default": "concept",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "links": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Note IDs to link to",
                },
                "source_type": {"type": "string", "description": "Source type (e.g. paper, book, web)"},
                "source_id": {"type": "string", "description": "Source identifier"},
            },
            "required": ["title", "content"],
        },
        handler=partial(note_create, workspace=workspace),
    )

    registry.register(
        name="note_search",
        description=(
            "Search notes by text query, tags, linked_to note, or note_type. "
            "Returns ranked results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text search query"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags",
                },
                "linked_to": {"type": "string", "description": "Filter by notes linked to this note ID"},
                "note_type": {
                    "type": "string",
                    "enum": ["concept", "evidence", "comment"],
                    "description": "Filter by note type",
                },
            },
            "required": [],
        },
        handler=partial(note_search, workspace=workspace),
    )

    registry.register(
        name="note_link",
        description=(
            "Add or update bidirectional links between notes. "
            "Auto-maintains backlink table."
        ),
        parameters={
            "type": "object",
            "properties": {
                "note_id": {"type": "string", "description": "Note ID to update links for"},
                "links": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of target note IDs",
                },
            },
            "required": ["note_id", "links"],
        },
        handler=partial(note_link, workspace=workspace),
    )

    registry.register(
        name="note_graph",
        description=(
            "Get the note link graph as nodes and edges. "
            "Includes forward links and backlinks."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=partial(note_graph, workspace=workspace),
    )

    registry.register(
        name="note_from_paper",
        description=(
            "Auto-generate an evidence note from paper extraction results. "
            "Takes the output of paper_extract_elements and creates a note card."
        ),
        parameters={
            "type": "object",
            "properties": {
                "elements": {
                    "type": "object",
                    "description": "Extracted elements dict from paper_extract_elements",
                },
                "paper_title": {"type": "string", "description": "Paper title"},
                "paper_id": {"type": "string", "description": "Paper identifier"},
            },
            "required": ["elements"],
        },
        handler=partial(note_from_paper, workspace=workspace),
    )

    # LiteratureGraph tools
    registry.register(
        name="literature_graph_build",
        description=(
            "Build a literature citation graph from references.bib and "
            "citation_relations.json. Returns graph statistics and a "
            "Mermaid visualization."
        ),
        parameters={
            "type": "object",
            "properties": {
                "literature_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of BibTeX keys to include",
                },
            },
            "required": [],
        },
        handler=partial(literature_graph_build, workspace=workspace),
    )

    registry.register(
        name="literature_graph_visualize",
        description=(
            "Visualize the literature graph in Mermaid, TikZ, or DOT format. "
            "Supports directed edges with relation type labels."
        ),
        parameters={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["mermaid", "tikz", "dot"],
                    "default": "mermaid",
                },
                "literature_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of BibTeX keys to include",
                },
            },
            "required": [],
        },
        handler=partial(literature_graph_visualize, workspace=workspace),
    )

    registry.register(
        name="literature_graph_clusters",
        description=(
            "Detect clusters/communities in the literature graph using Louvain "
            "algorithm, and find key papers by PageRank and degree centrality."
        ),
        parameters={
            "type": "object",
            "properties": {
                "literature_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of BibTeX keys to include",
                },
            },
            "required": [],
        },
        handler=partial(literature_graph_clusters, workspace=workspace),
    )
