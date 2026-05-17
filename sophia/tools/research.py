"""Literature search tool for SophiaAgent.

Integrates: Semantic Scholar API, arXiv API, Crossref API.
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any, Dict, List, Tuple

import httpx

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 20.0
MAX_NETWORK_ATTEMPTS = 2

# arXiv Atom namespace
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class _HTMLStripper(HTMLParser):
    """Strip HTML tags from text, keeping inner content."""

    def __init__(self):
        super().__init__()
        self._parts: List[str] = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    stripper = _HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()


def _get_with_retry(url: str, **kwargs) -> httpx.Response:
    last_error = None
    for attempt in range(MAX_NETWORK_ATTEMPTS):
        try:
            resp = httpx.get(url, **kwargs)
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < MAX_NETWORK_ATTEMPTS - 1:
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 1.5 + attempt
                time.sleep(min(wait, 5.0))
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_error = exc
            if attempt < MAX_NETWORK_ATTEMPTS - 1:
                time.sleep(1.0 + attempt)
    raise last_error


def _search_semantic_scholar(
    query: str,
    max_results: int = 10,
    api_key: str = "",
) -> Tuple[List[Dict], str]:
    """Search Semantic Scholar API."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,authors,year,abstract,url,externalIds,citationCount",
    }

    try:
        resp = _get_with_retry(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for paper in data.get("data", []):
            authors = ", ".join(
                a.get("name", "") for a in (paper.get("authors") or [])[:5]
            )
            results.append({
                "title": paper.get("title", ""),
                "authors": authors,
                "year": paper.get("year"),
                "abstract": (paper.get("abstract") or "")[:300],
                "url": paper.get("url", ""),
                "doi": (paper.get("externalIds") or {}).get("DOI", ""),
                "arxiv": (paper.get("externalIds") or {}).get("ArXiv", ""),
                "citations": paper.get("citationCount", 0),
                "source": "semantic_scholar",
            })
        return results, ""
    except Exception as e:
        logger.warning("Semantic Scholar search failed: %s", e)
        return [], f"semantic_scholar failed: {type(e).__name__}: {e}"


def _search_arxiv(query: str, max_results: int = 10) -> Tuple[List[Dict], str]:
    """Search arXiv API."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(max_results, 50),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        resp = _get_with_retry(
            "https://export.arxiv.org/api/query",
            params=params,
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ATOM_NS)

        results = []
        for entry in entries[:max_results]:
            title_el = entry.find("atom:title", ATOM_NS)
            summary_el = entry.find("atom:summary", ATOM_NS)
            published_el = entry.find("atom:published", ATOM_NS)
            id_el = entry.find("atom:id", ATOM_NS)
            author_els = entry.findall("atom:author/atom:name", ATOM_NS)

            title = (
                (title_el.text or "").strip().replace("\n", " ")
                if title_el is not None else ""
            )
            summary = (
                (summary_el.text or "").strip().replace("\n", " ")[:300]
                if summary_el is not None else ""
            )
            year_str = (
                (published_el.text or "")[:4]
                if published_el is not None else None
            )
            arxiv_id = ""
            if id_el is not None and id_el.text:
                raw_id = id_el.text.strip()
                if "abs/" in raw_id:
                    arxiv_id = raw_id.split("abs/")[-1]

            authors = ", ".join(
                (a.text or "").strip() for a in author_els[:5]
            )

            results.append({
                "title": title,
                "authors": authors,
                "year": int(year_str) if year_str and year_str.isdigit() else None,
                "abstract": summary,
                "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "doi": "",
                "arxiv": arxiv_id,
                "citations": None,
                "source": "arxiv",
            })
        return results, ""
    except Exception as e:
        logger.warning("arXiv search failed: %s", e)
        return [], f"arxiv failed: {type(e).__name__}: {e}"


def _search_crossref(query: str, max_results: int = 10) -> Tuple[List[Dict], str]:
    """Search Crossref API."""
    params = {
        "query": query,
        "rows": min(max_results, 100),
        "sort": "relevance",
    }

    try:
        resp = _get_with_retry(
            "https://api.crossref.org/works",
            params=params,
            timeout=HTTP_TIMEOUT,
            headers={
                "User-Agent": (
                    "SophiaAgent/0.1.0 "
                    "(mailto:research@sophia-agent.org)"
                ),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("message", {}).get("items", [])

        results = []
        for item in items[:max_results]:
            authors = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in (item.get("author") or [])[:5]
            )
            title_list = item.get("title", [])
            abstract = item.get("abstract", "")
            if isinstance(abstract, str):
                abstract = _strip_html(abstract)[:300]

            results.append({
                "title": title_list[0] if title_list else "",
                "authors": authors,
                "year": (
                    item.get("published-print", {})
                    .get("date-parts", [[None]])[0][0]
                    or item.get("published-online", {})
                    .get("date-parts", [[None]])[0][0]
                ),
                "abstract": abstract,
                "url": item.get("URL", ""),
                "doi": item.get("DOI", ""),
                "arxiv": "",
                "citations": item.get("is-referenced-by-count", 0),
                "source": "crossref",
            })
        return results, ""
    except Exception as e:
        logger.warning("Crossref search failed: %s", e)
        return [], f"crossref failed: {type(e).__name__}: {e}"


def literature_search(args: Dict[str, Any]) -> str:
    """Search academic literature across multiple databases.

    Args:
        args: {query: str, max_results: int, sources: list[str]}
    """
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)

    max_results = args.get("max_results", 10)
    sources = args.get("sources") or ["crossref", "semantic_scholar", "arxiv"]

    all_results = []
    seen_titles = set()
    warnings = []

    if "crossref" in sources:
        results, warning = _search_crossref(query, max_results)
        if warning:
            warnings.append(warning)
        for r in results:
            if r["title"] not in seen_titles:
                all_results.append(r)
                seen_titles.add(r["title"])

    if "semantic_scholar" in sources:
        results, warning = _search_semantic_scholar(query, max_results)
        if warning:
            warnings.append(warning)
        for r in results:
            if r["title"] not in seen_titles:
                all_results.append(r)
                seen_titles.add(r["title"])

    if "arxiv" in sources:
        results, warning = _search_arxiv(query, max_results)
        if warning:
            warnings.append(warning)
        for r in results:
            if r["title"] not in seen_titles:
                all_results.append(r)
                seen_titles.add(r["title"])

    all_results = all_results[:max_results * 2]  # Allow some extra

    return json.dumps({
        "query": query,
        "total": len(all_results),
        "sources": sources,
        "warnings": warnings,
        "papers": all_results,
    }, ensure_ascii=False)


def register_research_tools(registry):
    """Register literature search tools."""
    registry.register(
        name="literature_search",
        description=(
            "Search academic literature across databases. "
            "Returns paper titles, authors, years, abstracts, and citation counts. "
            "Sources: semantic_scholar, arxiv, crossref."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords, paper title, author, etc.)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results per source",
                    "default": 10,
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "semantic_scholar",
                            "arxiv",
                            "crossref",
                        ],
                    },
                    "description": "Databases to search",
                    "default": ["crossref", "semantic_scholar", "arxiv"],
                },
            },
            "required": ["query"],
        },
        handler=literature_search,
    )
