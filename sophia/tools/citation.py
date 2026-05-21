"""Citation management tool for SophiaAgent.

Manages a per-project BibTeX reference library.
Formats citations in GB/T 7714, APA, MLA, Chicago styles.
Supports 8 literature types: journal, book, chapter, thesis, web, report, policy, conference.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 8 literature types mapping
# ---------------------------------------------------------------------------
_LITERATURE_TYPES = {
    "article": "journal",
    "inproceedings": "conference",
    "conference": "conference",
    "book": "book",
    "inbook": "chapter",
    "incollection": "chapter",
    "phdthesis": "thesis",
    "mastersthesis": "thesis",
    "techreport": "report",
    "misc": "web",
    "online": "web",
    "unpublished": "report",
    "manual": "report",
}


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
            fields["_lit_type"] = _LITERATURE_TYPES.get(fields["_type"].lower(), "journal")
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
        entry.pop("_lit_type", None)
        bib_entries.append(entry)
    db.entries = bib_entries

    writer = BibTexWriter()
    writer.indent = "  "
    with open(path, "w", encoding="utf-8") as f:
        f.write(writer.write(db))


# ---------------------------------------------------------------------------
# Formatters for 4 styles x 8 literature types
# ---------------------------------------------------------------------------

def _parse_authors_apa(authors: str) -> str:
    """Parse author string into APA 7th format (Last, F. M., & Last, F. M.)."""
    if not authors:
        return ""
    names = [n.strip() for n in authors.replace(" and ", ",").split(",") if n.strip()]
    formatted = []
    for name in names:
        parts = name.split()
        if len(parts) >= 2:
            last = parts[-1]
            initials = "".join(p[0] + "." for p in parts[:-1] if p)
            formatted.append(f"{last}, {initials}")
        else:
            formatted.append(name)
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + ", & " + formatted[-1]


def _parse_authors_mla(authors: str) -> str:
    """Parse author string into MLA 9th format."""
    if not authors:
        return ""
    names = [n.strip() for n in authors.replace(" and ", ",").split(",") if n.strip()]
    if not names:
        return ""
    first = names[0]
    parts = first.split()
    if len(parts) >= 2:
        last = parts[-1]
        rest = " ".join(parts[:-1])
        first_fmt = f"{last}, {rest}"
    else:
        first_fmt = first
    if len(names) == 1:
        return first_fmt
    if len(names) == 2:
        return f"{first_fmt}, and {names[1]}"
    return f"{first_fmt}, et al."


def _parse_authors_chicago(authors: str) -> str:
    """Parse author string into Chicago 17th (notes-bib) format."""
    if not authors:
        return ""
    names = [n.strip() for n in authors.replace(" and ", ",").split(",") if n.strip()]
    formatted = []
    for name in names:
        parts = name.split()
        if len(parts) >= 2:
            last = parts[-1]
            first = " ".join(parts[:-1])
            formatted.append(f"{last}, {first}")
        else:
            formatted.append(name)
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return ", ".join(formatted[:-1]) + ", and " + formatted[-1]


def _format_apa7(entry: Dict) -> str:
    """Format citation in APA 7th edition style."""
    lit_type = entry.get("_lit_type", "journal")
    authors = _parse_authors_apa(entry.get("author", ""))
    year = entry.get("year", "n.d.")
    title = entry.get("title", "")
    journal = entry.get("journal", "")
    booktitle = entry.get("booktitle", "")
    volume = entry.get("volume", "")
    issue = entry.get("number", entry.get("issue", ""))
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")
    url = entry.get("url", "")
    publisher = entry.get("publisher", "")
    edition = entry.get("edition", "")
    editors = entry.get("editor", "")
    city = entry.get("address", entry.get("location", ""))

    parts = []
    if authors:
        parts.append(authors)
    parts.append(f"({year}).")

    if lit_type == "journal":
        parts.append(f"{title}.")
        jpart = f"*{journal}*"
        if volume:
            jpart += f", *{volume}*"
        if issue:
            jpart += f"({issue})"
        if pages:
            jpart += f", {pages}"
        parts.append(jpart + ".")
        if doi:
            parts.append(f"https://doi.org/{doi}")
        elif url:
            parts.append(url)
    elif lit_type == "book":
        parts.append(f"*{title}*.")
        if edition:
            parts.append(f"({edition} ed.).")
        if publisher:
            parts.append(f"{publisher}.")
        if doi:
            parts.append(f"https://doi.org/{doi}")
    elif lit_type == "chapter":
        parts.append(f"{title}.")
        if editors:
            parts.append(f"In {_parse_authors_apa(editors)} (Eds.),")
        parts.append(f"*{booktitle}*")
        if edition:
            parts[-1] += f" ({edition} ed.)"
        parts[-1] += "."
        if pages:
            parts.append(f"pp. {pages}.")
        if publisher:
            parts.append(f"{publisher}.")
        if doi:
            parts.append(f"https://doi.org/{doi}")
    elif lit_type == "thesis":
        parts.append(f"*{title}* [Doctoral dissertation or Master's thesis].")
        if publisher:
            parts.append(f"{publisher}.")
        if url:
            parts.append(url)
    elif lit_type == "web":
        parts.append(f"{title}.")
        if publisher:
            parts.append(f"{publisher}.")
        if url:
            parts.append(url)
    elif lit_type == "report":
        parts.append(f"*{title}*.")
        if publisher:
            parts.append(f"{publisher}.")
        if url:
            parts.append(url)
    elif lit_type == "policy":
        parts.append(f"*{title}*.")
        if publisher:
            parts.append(f"{publisher}.")
        if url:
            parts.append(url)
    elif lit_type == "conference":
        parts.append(f"{title}.")
        if booktitle:
            parts.append(f"In *{booktitle}*")
            if pages:
                parts[-1] += f", pp. {pages}"
            parts[-1] += "."
        if publisher:
            parts.append(f"{publisher}.")
        if doi:
            parts.append(f"https://doi.org/{doi}")
    else:
        parts.append(f"{title}.")
        if journal:
            parts.append(f"*{journal}*.")
        if doi:
            parts.append(f"https://doi.org/{doi}")

    return " ".join(parts)


def _format_mla(entry: Dict) -> str:
    """Format citation in MLA 9th edition style."""
    lit_type = entry.get("_lit_type", "journal")
    authors = _parse_authors_mla(entry.get("author", ""))
    title = entry.get("title", "")
    journal = entry.get("journal", "")
    booktitle = entry.get("booktitle", "")
    volume = entry.get("volume", "")
    issue = entry.get("number", entry.get("issue", ""))
    year = entry.get("year", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")
    url = entry.get("url", "")
    publisher = entry.get("publisher", "")
    edition = entry.get("edition", "")

    parts = []
    if authors:
        parts.append(f"{authors}.")

    if lit_type == "journal":
        parts.append(f"\"{title}.\"")
        jpart = f"*{journal}*"
        if volume:
            jpart += f", vol. {volume}"
        if issue:
            jpart += f", no. {issue}"
        if year:
            jpart += f", {year}"
        if pages:
            jpart += f", pp. {pages}"
        parts.append(jpart + ".")
        if doi:
            parts.append(f"doi:{doi}.")
        elif url:
            parts.append(f"{url}.")
    elif lit_type == "book":
        parts.append(f"*{title}*.")
        if edition:
            parts.append(f"{edition} ed.,")
        if publisher:
            parts.append(f"{publisher},")
        if year:
            parts.append(f"{year}.")
    elif lit_type == "chapter":
        parts.append(f"\"{title}.\"")
        if booktitle:
            parts.append(f"*{booktitle}*,")
        if publisher:
            parts.append(f"{publisher},")
        if year:
            parts.append(f"{year},")
        if pages:
            parts.append(f"pp. {pages}.")
    elif lit_type == "thesis":
        parts.append(f"*{title}*.")
        if year:
            parts.append(f"{year}.")
        if publisher:
            parts.append(f"{publisher},")
    elif lit_type in ("web", "report", "policy"):
        parts.append(f"\"{title}.\"")
        if publisher:
            parts.append(f"{publisher},")
        if year:
            parts.append(f"{year}.")
        if url:
            parts.append(f"{url}.")
    elif lit_type == "conference":
        parts.append(f"\"{title}.\"")
        if booktitle:
            parts.append(f"*{booktitle}*,")
        if publisher:
            parts.append(f"{publisher},")
        if year:
            parts.append(f"{year},")
        if pages:
            parts.append(f"pp. {pages}.")
    else:
        parts.append(f"\"{title}.\"")
        if journal:
            parts.append(f"*{journal}*.")

    return " ".join(parts)


def _format_chicago(entry: Dict) -> str:
    """Format citation in Chicago 17th edition (notes-bibliography) style."""
    lit_type = entry.get("_lit_type", "journal")
    authors = _parse_authors_chicago(entry.get("author", ""))
    title = entry.get("title", "")
    journal = entry.get("journal", "")
    booktitle = entry.get("booktitle", "")
    volume = entry.get("volume", "")
    issue = entry.get("number", entry.get("issue", ""))
    year = entry.get("year", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")
    url = entry.get("url", "")
    publisher = entry.get("publisher", "")
    edition = entry.get("edition", "")
    city = entry.get("address", entry.get("location", ""))

    parts = []
    if authors:
        parts.append(f"{authors}.")

    if lit_type == "journal":
        parts.append(f"\"{title}.\"")
        jpart = f"*{journal}*"
        if volume:
            jpart += f" {volume}"
        if issue:
            jpart += f", no. {issue}"
        if year:
            jpart += f" ({year})"
        if pages:
            jpart += f": {pages}"
        parts.append(jpart + ".")
        if doi:
            parts.append(f"doi:{doi}.")
        elif url:
            parts.append(f"{url}.")
    elif lit_type == "book":
        parts.append(f"*{title}*.")
        if edition:
            parts.append(f"{edition} ed.")
        if city and publisher:
            parts.append(f"{city}: {publisher},")
        elif publisher:
            parts.append(f"{publisher},")
        if year:
            parts.append(f"{year}.")
    elif lit_type == "chapter":
        parts.append(f"\"{title}.\"")
        if booktitle:
            parts.append(f"In *{booktitle}*,")
        if pages:
            parts.append(f"{pages}.")
        if city and publisher:
            parts.append(f"{city}: {publisher},")
        elif publisher:
            parts.append(f"{publisher},")
        if year:
            parts.append(f"{year}.")
    elif lit_type == "thesis":
        parts.append(f"\"{title}.\"")
        if publisher:
            parts.append(f"{publisher},")
        if year:
            parts.append(f"{year}.")
    elif lit_type in ("web", "report", "policy"):
        parts.append(f"\"{title}.\"")
        if publisher:
            parts.append(f"{publisher}.")
        if year:
            parts.append(f"Last modified {year}.")
        if url:
            parts.append(f"{url}.")
    elif lit_type == "conference":
        parts.append(f"\"{title}.\"")
        if booktitle:
            parts.append(f"Paper presented at {booktitle},")
        if year:
            parts.append(f"{year}.")
        if pages:
            parts.append(f"{pages}.")
    else:
        parts.append(f"\"{title}.\"")
        if journal:
            parts.append(f"*{journal}*.")

    return " ".join(parts)


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
    publisher = entry.get("publisher", "")
    url = entry.get("url", "")
    city = entry.get("address", entry.get("location", ""))

    type_markers = {
        "article": "[J]", "inproceedings": "[C]", "conference": "[C]",
        "book": "[M]", "inbook": "[M]", "incollection": "[M]",
        "phdthesis": "[D]", "mastersthesis": "[D]",
        "techreport": "[R]", "misc": "[EB/OL]", "online": "[EB/OL]",
        "unpublished": "[R]", "manual": "[R]",
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
    if publisher:
        parts.append(f"{publisher}.")
    if doi:
        parts.append(f"DOI: {doi}")
    elif url:
        parts.append(url)

    return " ".join(parts)


_STYLE_FORMATTERS = {
    "gb-t-7714-2015": _format_gbt7714,
    "gb-t-7714": _format_gbt7714,
    "apa": _format_apa7,
    "apa-7th": _format_apa7,
    "mla": _format_mla,
    "chicago": _format_chicago,
    "chicago-17th": _format_chicago,
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
                    "enum": ["gb-t-7714-2015", "apa-7th", "apa7", "apa",
                             "chicago", "chicago-17th", "mla"],
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

def _convert_citation_style(args: Dict[str, Any], workspace: str) -> str:
    from_style = args.get("from_style", "")
    to_style = args.get("to_style", "")
    document_text = args.get("document_text", "")

    if not from_style or not to_style:
        return json.dumps(
            {"error": "from_style and to_style are required"},
            ensure_ascii=False,
        )

    valid_styles = ["apa7", "apa-7th", "apa", "chicago", "chicago-17th",
                    "mla", "gb-t-7714-2015", "gb-t-7714"]
    if to_style not in valid_styles:
        return json.dumps(
            {"error": f"Invalid to_style. Must be one of: {valid_styles}"},
            ensure_ascii=False,
        )

    entries = _load_bib(workspace)
    if not entries:
        return json.dumps(
            {"error": "No references found in workspace"},
            ensure_ascii=False,
        )

    formatter = _STYLE_FORMATTERS.get(to_style, _format_gbt7714)

    converted = []
    for e in entries:
        converted.append({
            "key": e.get("_key", ""),
            "old_formatted": _STYLE_FORMATTERS.get(from_style, _format_gbt7714)(e),
            "new_formatted": formatter(e),
            "lit_type": e.get("_lit_type", "journal"),
        })

    # Update in-text citations in document_text if provided
    updated_text = document_text
    if updated_text:
        # Simple heuristic: replace (Author, Year) patterns based on entries
        for e in entries:
            author = e.get("author", "")
            year = e.get("year", "")
            key = e.get("_key", "")
            if not author or not year:
                continue
            first_author_last = author.split()[-1] if author.split() else author
            # Match patterns like (Smith, 2020) or (Smith et al., 2020)
            old_patterns = [
                rf'\({re.escape(first_author_last)}\s+et\s+al\.?,\s*{re.escape(year)}\)',
                rf'\({re.escape(first_author_last)},\s*{re.escape(year)}\)',
            ]
            if to_style in ("apa7", "apa-7th", "apa"):
                new_citation = f"({first_author_last}, {year})"
            elif to_style in ("mla",):
                new_citation = f"({first_author_last} {year})"
            elif to_style in ("chicago", "chicago-17th"):
                new_citation = f"{first_author_last} ({year})"
            else:
                new_citation = f"[{key}]"
            for pat in old_patterns:
                updated_text = re.sub(pat, new_citation, updated_text, flags=re.IGNORECASE)

    return json.dumps({
        "action": "style_converted",
        "from_style": from_style,
        "to_style": to_style,
        "total_references": len(converted),
        "converted_references": converted,
        "document_updated": bool(document_text),
        "updated_text_preview": updated_text[:500] if updated_text else "",
    }, ensure_ascii=False)


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
                    "enum": ["gb-t-7714-2015", "apa-7th", "apa7", "apa",
                             "chicago", "chicago-17th", "mla"],
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

    registry.register(
        name="citation_style_convert",
        description=(
            "Convert all references from one citation style to another. "
            "Supports APA7, Chicago 17th, MLA 9th, and GB/T 7714-2015. "
            "Optionally updates in-text citations in a provided document text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "from_style": {
                    "type": "string",
                    "description": "Original citation style",
                    "enum": ["apa7", "apa-7th", "apa", "chicago", "chicago-17th",
                             "mla", "gb-t-7714-2015", "gb-t-7714"],
                },
                "to_style": {
                    "type": "string",
                    "description": "Target citation style",
                    "enum": ["apa7", "apa-7th", "apa", "chicago", "chicago-17th",
                             "mla", "gb-t-7714-2015", "gb-t-7714"],
                },
                "document_text": {
                    "type": "string",
                    "description": "Optional document text to update in-text citations",
                },
            },
            "required": ["from_style", "to_style"],
        },
        handler=partial(_convert_citation_style, workspace=workspace),
    )
