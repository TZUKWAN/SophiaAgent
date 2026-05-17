"""Citation management tool for SophiaAgent.

Manages a per-project BibTeX reference library.
Formats citations in GB/T 7714, APA, MLA, Chicago styles.
"""

import json
import logging
import os
from typing import Any, Dict, List

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

logger = logging.getLogger(__name__)


def _bib_db_path(workspace: str) -> str:
    return os.path.join(workspace, ".sophia", "references.bib")


def _relations_path(workspace: str) -> str:
    return os.path.join(workspace, ".sophia", "citation_relations.json")


def _ensure_bib_dir(workspace: str) -> None:
    d = os.path.join(workspace, ".sophia")
    os.makedirs(d, exist_ok=True)


def _load_relations(workspace: str) -> List[Dict]:
    path = _relations_path(workspace)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_relations(workspace: str, relations: List[Dict]) -> None:
    _ensure_bib_dir(workspace)
    path = _relations_path(workspace)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)


def _load_bib(workspace: str) -> List[Dict]:
    """Load all BibTeX entries from the reference library using bibtexparser."""
    path = _bib_db_path(workspace)
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if not content.strip():
        return []

    try:
        parser = BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        bib_db = bibtexparser.loads(content, parser=parser)

        entries = []
        for entry in bib_db.entries:
            fields = dict(entry)
            fields["_key"] = fields.pop("ID", "")
            fields["_type"] = fields.pop("ENTRYTYPE", "article")
            entries.append(fields)
        return entries
    except Exception as e:
        logger.warning("Failed to parse BibTeX: %s", e)
        return []


def _save_bib(workspace: str, entries: List[Dict]) -> None:
    """Save entries back to BibTeX file using bibtexparser."""
    _ensure_bib_dir(workspace)
    path = _bib_db_path(workspace)

    db = BibDatabase()
    bib_entries = []
    for e in entries:
        entry = dict(e)
        entry["ID"] = entry.pop("_key", "unknown")
        entry["ENTRYTYPE"] = entry.pop("_type", "article")
        bib_entries.append(entry)
    db.entries = bib_entries

    writer = BibTexWriter()
    writer.indent = "  "
    with open(path, "w", encoding="utf-8") as f:
        f.write(writer.write(db))


def _format_gbt7714(entry: Dict) -> str:
    """Format citation in GB/T 7714-2015 style."""
    authors = entry.get("author", "")
    title = entry.get("title", "")
    journal = entry.get("journal", entry.get("booktitle", ""))
    year = entry.get("year", "")
    volume = entry.get("volume", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")
    entry_type = entry.get("_type", "").lower()

    type_markers = {
        "article": "[J]", "inproceedings": "[C]", "conference": "[C]",
        "book": "[M]", "inbook": "[M]", "incollection": "[M]",
        "phdthesis": "[D]", "mastersthesis": "[D]",
        "techreport": "[R]", "misc": "[EB/OL]",
    }
    marker = type_markers.get(entry_type, "[J]" if journal else "[M]")

    parts = []
    if authors:
        parts.append(authors)
    if year:
        parts.append(f"{year}.")
    if title:
        parts.append(f"{title}{marker}.")
    if journal:
        jpart = journal
        if volume:
            jpart += f", {volume}"
        if pages:
            jpart += f": {pages}"
        parts.append(jpart + ".")
    if doi:
        parts.append(f"DOI: {doi}")

    return " ".join(parts)


def _format_apa(entry: Dict) -> str:
    """Format citation in APA 7th style."""
    authors = entry.get("author", "")
    year = entry.get("year", "")
    title = entry.get("title", "")
    journal = entry.get("journal", entry.get("booktitle", ""))
    volume = entry.get("volume", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")

    parts = []
    if authors:
        parts.append(authors)
    if year:
        parts.append(f"({year}).")
    if title:
        parts.append(f"{title}.")
    if journal:
        jpart = f"*{journal}*"
        if volume:
            jpart += f", *{volume}*"
        if pages:
            jpart += f", {pages}"
        parts.append(jpart + ".")
    if doi:
        parts.append(f"https://doi.org/{doi}")

    return " ".join(parts)


_STYLE_FORMATTERS = {
    "gb-t-7714-2015": _format_gbt7714,
    "gb-t-7714": _format_gbt7714,
    "apa": _format_apa,
    "apa-7th": _format_apa,
}


def ref_add(args: Dict[str, Any], workspace: str) -> str:
    """Add a reference to the library.

    Args: {key: str, type: str, fields: dict}
    """
    key = args.get("key", "")
    if not key:
        return json.dumps({"error": "key is required"}, ensure_ascii=False)

    entry_type = args.get("type", "article")
    fields = args.get("fields", {})
    fields["_key"] = key
    fields["_type"] = entry_type

    entries = _load_bib(workspace)

    # Check for duplicate key
    for i, e in enumerate(entries):
        if e.get("_key") == key:
            entries[i] = fields
            _save_bib(workspace, entries)
            return json.dumps({"action": "updated", "key": key}, ensure_ascii=False)

    entries.append(fields)
    _save_bib(workspace, entries)
    return json.dumps({"action": "added", "key": key, "total": len(entries)}, ensure_ascii=False)


def ref_list(args: Dict[str, Any], workspace: str) -> str:
    """List all references in the library."""
    entries = _load_bib(workspace)
    result = []
    for e in entries:
        result.append({
            "key": e.get("_key", ""),
            "title": e.get("title", ""),
            "author": e.get("author", ""),
            "year": e.get("year", ""),
        })
    return json.dumps({"total": len(result), "references": result}, ensure_ascii=False)


def ref_format(args: Dict[str, Any], workspace: str) -> str:
    """Format references in a citation style.

    Args: {style: str, keys: list[str]}
    """
    style = args.get("style", "gb-t-7714-2015")
    requested_keys = args.get("keys", [])

    entries = _load_bib(workspace)
    if not entries:
        return json.dumps({"error": "No references in library"}, ensure_ascii=False)

    formatter = _STYLE_FORMATTERS.get(style, _format_gbt7714)

    results = []
    for e in entries:
        key = e.get("_key", "")
        if requested_keys and key not in requested_keys:
            continue
        results.append({
            "key": key,
            "formatted": formatter(e),
        })

    return json.dumps({"style": style, "citations": results}, ensure_ascii=False)


def ref_search(args: Dict[str, Any], workspace: str) -> str:
    """Search references by keyword in title/author.

    Args: {query: str}
    """
    query = args.get("query", "").lower()
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)

    entries = _load_bib(workspace)
    results = []
    for e in entries:
        title = e.get("title", "").lower()
        author = e.get("author", "").lower()
        if query in title or query in author:
            results.append({
                "key": e.get("_key", ""),
                "title": e.get("title", ""),
                "author": e.get("author", ""),
                "year": e.get("year", ""),
            })

    return json.dumps(
        {"query": query, "found": len(results), "references": results},
        ensure_ascii=False,
    )


def register_citation_tools(registry, workspace: str):
    """Register citation management tools."""
    from functools import partial

    registry.register(
        name="ref_add",
        description=(
            "Add a reference to the project bibliography. "
            "Provide a unique key, entry type (article/book/incollection), and fields dict "
            "(title, author, year, journal, volume, pages, doi)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Unique citation key (e.g. 'Smith2024')",
                },
                "type": {
                    "type": "string",
                    "description": "BibTeX entry type",
                    "default": "article",
                    "enum": [
                        "article", "book",
                        "incollection", "inproceedings", "misc",
                    ],
                },
                "fields": {
                    "type": "object",
                    "description": (
                        "BibTeX fields: title, author, year, "
                        "journal, volume, pages, doi, etc."
                    ),
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "year": {"type": "string"},
                        "journal": {"type": "string"},
                        "volume": {"type": "string"},
                        "pages": {"type": "string"},
                        "doi": {"type": "string"},
                    },
                },
            },
            "required": ["key", "fields"],
        },
        handler=partial(ref_add, workspace=workspace),
    )

    registry.register(
        name="ref_list",
        description="List all references in the project bibliography.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=partial(ref_list, workspace=workspace),
    )

    registry.register(
        name="ref_format",
        description=(
            "Format references in a citation style. "
            "Supported: gb-t-7714-2015 (Chinese standard), apa-7th. "
            "Omit 'keys' to format all references."
        ),
        parameters={
            "type": "object",
            "properties": {
                "style": {
                    "type": "string",
                    "default": "gb-t-7714-2015",
                    "enum": ["gb-t-7714-2015", "apa-7th"],
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific reference keys to format (omit for all)",
                },
            },
            "required": [],
        },
        handler=partial(ref_format, workspace=workspace),
    )

    registry.register(
        name="ref_search",
        description="Search references by keyword in title or author.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keyword"},
            },
            "required": ["query"],
        },
        handler=partial(ref_search, workspace=workspace),
    )

    def _ref_add_relation(args: Dict[str, Any], workspace: str) -> str:
        from_key = args.get("from_key", "")
        to_key = args.get("to_key", "")
        rel_type = args.get("relation", "")
        if not from_key or not to_key or not rel_type:
            return json.dumps(
                {"error": "from_key, to_key, and relation are required"},
                ensure_ascii=False,
            )

        valid_types = [
            "builds-on", "contradicts", "parallel",
            "supersedes", "applies", "critiques",
        ]
        if rel_type not in valid_types:
            return json.dumps(
                {"error": f"Invalid relation type. Must be: {valid_types}"},
                ensure_ascii=False,
            )

        relations = _load_relations(workspace)
        relations.append({"from": from_key, "to": to_key, "type": rel_type})
        _save_relations(workspace, relations)

        return json.dumps({
            "action": "relation_added",
            "from": from_key,
            "to": to_key,
            "type": rel_type,
            "total_relations": len(relations),
        }, ensure_ascii=False)

    def _ref_network(args: Dict[str, Any], workspace: str) -> str:
        entries = _load_bib(workspace)
        relations = _load_relations(workspace)

        nodes = []
        for e in entries:
            nodes.append({
                "id": e.get("_key", ""),
                "label": e.get("title", "")[:60],
                "author": e.get("author", ""),
                "year": e.get("year", ""),
                "citations": sum(
                    1 for r in relations
                    if r.get("to") == e.get("_key", "") or r.get("from") == e.get("_key", "")
                ),
            })

        edges = []
        type_colors = {
            "builds-on": "#22c55e",
            "contradicts": "#ef4444",
            "parallel": "#3b82f6",
            "supersedes": "#f59e0b",
            "applies": "#8b5cf6",
            "critiques": "#ec4899",
        }
        for r in relations:
            edges.append({
                "from": r.get("from", ""),
                "to": r.get("to", ""),
                "type": r.get("type", ""),
                "color": type_colors.get(r.get("type", ""), "#999"),
            })

        return json.dumps({
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }, ensure_ascii=False)

    registry.register(
        name="ref_add_relation",
        description=(
            "Add a typed citation relationship between two references. "
            "Types: builds-on (基于...发展), contradicts (与...矛盾), "
            "parallel (平行研究), supersedes (取代), applies (应用...方法), "
            "critiques (批评)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "from_key": {"type": "string", "description": "Source reference key"},
                "to_key": {"type": "string", "description": "Target reference key"},
                "relation": {
                    "type": "string",
                    "enum": [
                        "builds-on", "contradicts",
                        "parallel", "supersedes",
                        "applies", "critiques",
                    ],
                    "description": "Relationship type",
                },
            },
            "required": ["from_key", "to_key", "relation"],
        },
        handler=partial(_ref_add_relation, workspace=workspace),
    )

    registry.register(
        name="ref_network",
        description="Get the citation network as nodes and edges for visualization.",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=partial(_ref_network, workspace=workspace),
    )
