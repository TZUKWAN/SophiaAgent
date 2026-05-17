"""Tests for MethodSearcher: find methods from external sources."""
import json
import pytest

from sophia.research.discovery.method_catalog import MethodCatalog
from sophia.research.discovery.method_searcher import MethodSearcher


@pytest.fixture
def searcher(tmp_path):
    """Create a MethodSearcher with a fresh catalog."""
    db_path = str(tmp_path / "test_search.db")
    catalog = MethodCatalog(db_path)
    return MethodSearcher(catalog)


@pytest.fixture
def searcher_with_catalog(tmp_path):
    """Create a MethodSearcher and also return the catalog for inspection."""
    db_path = str(tmp_path / "test_search2.db")
    catalog = MethodCatalog(db_path)
    searcher = MethodSearcher(catalog)
    return searcher, catalog


class TestMethodSearcherBasic:
    def test_search_returns_json_string(self, searcher):
        result = searcher.search("t-test")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_search_empty_description(self, searcher):
        result = searcher.search("")
        parsed = json.loads(result)
        assert parsed["found"] is False

    def test_search_whitespace_description(self, searcher):
        result = searcher.search("   ")
        parsed = json.loads(result)
        assert parsed["found"] is False


class TestMethodSearcherCatalogHit:
    def test_finds_installed_method(self, searcher):
        result = searcher.search("t-test")
        parsed = json.loads(result)
        assert parsed["found"] is True
        assert parsed["source"] == "catalog"
        assert parsed["status"] == "installed"

    def test_finds_anova(self, searcher):
        result = searcher.search("ANOVA")
        parsed = json.loads(result)
        assert parsed["found"] is True
        methods = parsed.get("methods", [])
        assert any("ANOVA" in m.get("name", "") for m in methods)

    def test_finds_correlation(self, searcher):
        result = searcher.search("correlation")
        parsed = json.loads(result)
        assert parsed["found"] is True

    def test_search_with_category_filter(self, searcher):
        result = searcher.search("test", category="statistics")
        parsed = json.loads(result)
        if parsed.get("methods"):
            for m in parsed["methods"]:
                assert m["category"] == "statistics"


class TestMethodSearcherExternalCandidates:
    def test_irt_generates_girth_candidate(self, searcher):
        result = searcher.search("irt analysis")
        parsed = json.loads(result)
        # Should either find in catalog or generate candidates
        assert parsed["found"] is True
        candidates = parsed.get("candidates", [])
        assert any(c.get("library") == "girth" for c in candidates)

    def test_sem_generates_semopy_candidate(self, searcher):
        result = searcher.search("structural equation modeling")
        parsed = json.loads(result)
        assert parsed["found"] is True
        candidates = parsed.get("candidates", [])
        assert any(c.get("library") == "semopy" for c in candidates)

    def test_survival_generates_lifelines_candidate(self, searcher):
        result = searcher.search("survival analysis")
        parsed = json.loads(result)
        assert parsed["found"] is True
        candidates = parsed.get("candidates", [])
        assert any(c.get("library") == "lifelines" for c in candidates)

    def test_no_match_returns_not_found(self, searcher):
        # Use very short words so the name heuristic fallback won't produce a candidate
        result = searcher.search("a b c")
        parsed = json.loads(result)
        assert parsed["found"] is False

    def test_candidate_has_importable_field(self, searcher):
        # Search for something that maps to an external library
        result = searcher.search("item response theory")
        parsed = json.loads(result)
        if parsed.get("candidates"):
            for c in parsed["candidates"]:
                assert "importable" in c


class TestMethodSearcherCategoryInference:
    def test_infer_statistics(self, searcher):
        cat = searcher._infer_category("t-test for comparing two groups")
        assert cat == "statistics"

    def test_infer_causal(self, searcher):
        cat = searcher._infer_category("causal effect estimation")
        assert cat == "causal"

    def test_infer_ml(self, searcher):
        cat = searcher._infer_category("train a machine learning model")
        assert cat == "ml"

    def test_infer_default(self, searcher):
        cat = searcher._infer_category("something completely unrelated")
        assert cat == "uncategorized"


class TestMethodSearcherValidation:
    def test_validate_existing_library(self, searcher):
        candidate = {"library": "json"}  # json is always available
        assert searcher._validate_candidate(candidate) is True

    def test_validate_nonexistent_library(self, searcher):
        candidate = {"library": "xyzzy_nonexistent_package_12345"}
        assert searcher._validate_candidate(candidate) is False

    def test_validate_empty_library(self, searcher):
        candidate = {"library": ""}
        assert searcher._validate_candidate(candidate) is False

    def test_validate_missing_library_key(self, searcher):
        candidate = {}
        assert searcher._validate_candidate(candidate) is False


class MockProvider:
    """Mock LLM provider that returns predefined responses."""
    def __init__(self, response_text: str):
        self.response_text = response_text

    def chat(self, messages, tools=None):
        from sophia.providers.base import ProviderResponse
        return ProviderResponse(content=self.response_text)


class TestMethodSearcherLLMPath:
    def test_llm_candidate_generation(self, tmp_path):
        db_path = str(tmp_path / "test_llm.db")
        catalog = MethodCatalog(db_path)
        # Use a library NOT in the keyword map to avoid deduplication
        llm_response = (
            '[{"library": "orbit-ml", "pip_name": "orbit-ml", '
            '"description": "Bayesian time series with orbit"}]'
        )
        provider = MockProvider(llm_response)
        searcher = MethodSearcher(catalog, provider=provider)

        # Use a description that does NOT match any keyword in KEYWORD_LIBRARY_MAP
        result = searcher.search("orbit bayesian structural time series")
        parsed = json.loads(result)
        assert parsed["found"] is True
        candidates = parsed.get("candidates", [])
        assert any(c.get("library") == "orbit-ml" for c in candidates)
        assert any(c.get("source") == "llm_suggestion" for c in candidates)

    def test_llm_bad_json_falls_back_gracefully(self, tmp_path):
        db_path = str(tmp_path / "test_llm2.db")
        catalog = MethodCatalog(db_path)
        provider = MockProvider("not valid json")
        searcher = MethodSearcher(catalog, provider=provider)

        # Should still find something via keyword fallback or catalog
        result = searcher.search("bayesian")
        parsed = json.loads(result)
        assert parsed["found"] is True
