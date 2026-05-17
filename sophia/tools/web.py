"""Web search and content extraction tool for SophiaAgent."""

import json
import logging
from html.parser import HTMLParser
from typing import Any, Dict, List
from urllib.parse import unquote

import httpx

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 30.0


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
    return stripper.get_text().strip()


def web_search(args: Dict[str, Any]) -> str:
    """Search the web using DuckDuckGo HTML (no API key required).

    Args: {query: str, max_results: int}
    """
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "query is required"}, ensure_ascii=False)

    max_results = args.get("max_results", 10)

    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; SophiaAgent/0.1.0)"},
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()

        results = []
        from html.parser import HTMLParser

        class _ResultParser(HTMLParser):
            """Parse DDG HTML to extract search results."""
            def __init__(self):
                super().__init__()
                self.results = []
                self._in_result_a = False
                self._in_snippet = False
                self._current_title = ""
                self._current_snippet = ""
                self._urls = []

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                classes = attrs_dict.get("class", "")
                if tag == "a" and "result__a" in classes:
                    self._in_result_a = True
                    self._current_title = ""
                    href = attrs_dict.get("href", "")
                    if "uddg=" in href:
                        uddg = href.split("uddg=")[-1].split("&")[0]
                        self._urls.append(unquote(uddg))
                elif tag == "a" and "result__snippet" in classes:
                    self._in_snippet = True
                    self._current_snippet = ""

            def handle_data(self, data):
                if self._in_result_a:
                    self._current_title += data
                elif self._in_snippet:
                    self._current_snippet += data

            def handle_endtag(self, tag):
                if self._in_result_a and tag == "a":
                    self._in_result_a = False
                    self.results.append({
                        "title": self._current_title.strip(),
                        "snippet": "",
                        "url": (
                            self._urls[len(self.results)]
                            if len(self._urls) > len(self.results)
                            else ""
                        ),
                    })
                elif self._in_snippet and tag == "a":
                    self._in_snippet = False
                    if self.results:
                        self.results[-1]["snippet"] = self._current_snippet.strip()[:300]

        parser = _ResultParser()
        parser.feed(resp.text)

        for r in parser.results[:max_results]:
            results.append(r)

        return json.dumps({
            "query": query,
            "total": len(results),
            "results": results,
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return json.dumps({"error": f"Search failed: {e}"}, ensure_ascii=False)


def web_extract(args: Dict[str, Any]) -> str:
    """Extract text content from a URL.

    Args: {url: str, max_length: int}
    """
    url = args.get("url", "")
    if not url:
        return json.dumps({"error": "url is required"}, ensure_ascii=False)

    max_length = args.get("max_length", 5000)

    try:
        resp = httpx.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.5",
            },
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()

        class _ContentExtractor(HTMLParser):
            """Extract visible text from HTML, skipping script/style."""
            def __init__(self):
                super().__init__()
                self._parts = []
                self._skip = False
                self.title = ""

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "noscript"):
                    self._skip = True
                elif tag == "title":
                    self._in_title = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "noscript"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    self._parts.append(data)

        extractor = _ContentExtractor()
        extractor.feed(resp.text)

        text = " ".join(" ".join(p.split()) for p in extractor._parts)
        text = text.strip()

        if len(text) > max_length:
            text = text[:max_length] + "..."

        # Extract title
        title = ""
        title_start = resp.text.lower().find("<title")
        if title_start != -1:
            title_open_end = resp.text.find(">", title_start)
            title_close = resp.text.find("</title>", title_open_end)
            if title_open_end != -1 and title_close != -1:
                title = resp.text[title_open_end + 1:title_close].strip()

        return json.dumps({
            "url": url,
            "title": title,
            "content": text,
            "length": len(text),
        }, ensure_ascii=False)
    except Exception as e:
        logger.warning("Web extract failed: %s", e)
        return json.dumps({"error": f"Extraction failed: {e}"}, ensure_ascii=False)


def register_web_tools(registry):
    """Register web search tools."""
    registry.register(
        name="web_search",
        description=(
            "Search the web using DuckDuckGo. "
            "Returns titles, snippets, and URLs for matching pages."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
        handler=web_search,
    )

    registry.register(
        name="web_extract",
        description=(
            "Extract text content from a web page URL. "
            "Returns the page title and cleaned text content."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to extract content from"},
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length in characters",
                    "default": 5000,
                },
            },
            "required": ["url"],
        },
        handler=web_extract,
    )
