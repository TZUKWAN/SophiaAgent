import json

import httpx


def test_literature_search_defaults_to_multiple_sources(monkeypatch):
    from sophia.tools import research

    calls = []

    def fake_crossref(query, max_results):
        calls.append("crossref")
        return ([{"title": "A", "authors": "Author", "year": 2024, "abstract": "", "url": "", "doi": "", "arxiv": "", "citations": 1, "source": "crossref"}], "")

    def fake_semantic(query, max_results, api_key=""):
        calls.append("semantic_scholar")
        return ([], "semantic_scholar failed: rate limited")

    def fake_arxiv(query, max_results):
        calls.append("arxiv")
        return ([], "")

    monkeypatch.setattr(research, "_search_crossref", fake_crossref)
    monkeypatch.setattr(research, "_search_semantic_scholar", fake_semantic)
    monkeypatch.setattr(research, "_search_arxiv", fake_arxiv)

    result = json.loads(research.literature_search({"query": "test", "max_results": 2}))

    assert calls == ["crossref", "semantic_scholar", "arxiv"]
    assert result["total"] == 1
    assert result["warnings"] == ["semantic_scholar failed: rate limited"]


def test_web_search_returns_actionable_error(monkeypatch):
    from sophia.tools import web

    def fail(*args, **kwargs):
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(web.httpx, "get", fail)

    result = json.loads(web.web_search({"query": "test"}))

    assert "error" in result
    assert "sophia doctor --network" in result["suggestion"]
