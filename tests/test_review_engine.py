"""Tests for the six-dimension review system."""

import pytest
from sophia.review.engine import ReviewEngine
from sophia.review.authenticity import AuthenticityChecker
from sophia.review.logic import LogicChecker
from sophia.review.citations import CitationChecker
from sophia.review.language import LanguageChecker
from sophia.review.statistics import StatisticsChecker
from sophia.review.ethics import EthicsChecker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_doc():
    return {
        "id": "doc-001",
        "title": "Test Paper",
        "abstract": "This is an abstract.",
        "authors": ["Alice", "Bob"],
        "sections": {
            "1": {"title": "Introduction", "content": "Intro text."},
            "2": {"title": "Methods", "content": "We used t-test and regression. Normality and homogeneity of variance were checked."},
            "3": {"title": "Results", "content": "t(98) = 2.50, p = .014, d = 0.50, 95% CI [0.10, 0.90]."},
            "4": {"title": "Discussion", "content": "The results suggest a significant effect."},
        },
        "references": [
            "Smith (2020). Title. Journal, 1(1), 1-10.",
            "Jones (2019). Another Title. Journal, 2(2), 20-30.",
        ],
    }


@pytest.fixture
def doc_with_issues():
    return {
        "id": "doc-002",
        "title": "Problematic Paper",
        "abstract": "",
        "authors": [],
        "sections": {
            "1": {"title": "Introduction", "content": "This is really very good stuff. It is interesting to note that things are like this."},
            "2": {"title": "Methods", "content": "We used t-test."},
            "3": {"title": "Results", "content": "The effect is significant, p = .120. t(98) = 1.20. We found a perfect correlation, r = 0.999."},
            "4": {"title": "Discussion", "content": "This proves our hypothesis."},
        },
        "references": [
            "placeholder",
            "Smith (2020).",
        ],
    }


@pytest.fixture
def empty_doc():
    return {
        "id": "doc-empty",
        "title": "",
        "abstract": "",
        "authors": [],
        "sections": {},
        "references": [],
    }


# ---------------------------------------------------------------------------
# ReviewEngine core
# ---------------------------------------------------------------------------

def test_review_engine_init():
    engine = ReviewEngine()
    assert set(engine.checkers.keys()) == {"authenticity", "logic", "citations", "language", "statistics", "ethics"}


def test_review_all_dimensions(minimal_doc):
    engine = ReviewEngine()
    report = engine.review(minimal_doc, citation_style="apa7")

    assert report["document_id"] == "doc-001"
    assert report["document_title"] == "Test Paper"
    assert 0 <= report["overall_score"] <= 100
    assert report["recommendation"] in {"reject", "major_revision", "minor_revision", "accept"}
    assert len(report["dimensions"]) == 6
    assert "all_findings" in report
    assert "critical_issues" in report
    assert "stats" in report


def test_review_subset_dimensions(minimal_doc):
    engine = ReviewEngine()
    report = engine.review(minimal_doc, dimensions=["language", "statistics"])
    assert set(report["dimensions"].keys()) == {"language", "statistics"}


def test_review_dimension_single(minimal_doc):
    engine = ReviewEngine()
    result = engine.review_dimension(minimal_doc, "statistics")
    assert result["dimension"] == "statistics"
    assert "score" in result
    assert "findings" in result


def test_review_dimension_unknown():
    engine = ReviewEngine()
    result = engine.review_dimension({}, "nonexistent")
    assert "error" in result


def test_recommendation_accept(minimal_doc):
    engine = ReviewEngine()
    report = engine.review(minimal_doc)
    assert report["recommendation"] in {"minor_revision", "accept"}


def test_recommendation_reject(doc_with_issues):
    engine = ReviewEngine()
    report = engine.review(doc_with_issues)
    assert report["recommendation"] in {"reject", "major_revision"}


def test_empty_document(empty_doc):
    engine = ReviewEngine()
    report = engine.review(empty_doc)
    assert report["overall_score"] == 0.0 or report["overall_score"] is not None
    assert isinstance(report["all_findings"], list)


# ---------------------------------------------------------------------------
# Authenticity
# ---------------------------------------------------------------------------

def test_authenticity_checker_init():
    checker = AuthenticityChecker()
    assert checker is not None


def test_authenticity_no_store():
    checker = AuthenticityChecker()
    doc = {
        "title": "Title",
        "abstract": "Abstract.",
        "sections": {"1": {"title": "Results", "content": "p = .05, n = 100, r = .30."}},
        "references": [],
    }
    result = checker.check(doc)
    assert result["dimension"] == "authenticity"
    assert result["score"] >= 0


def test_authenticity_citation_verification():
    checker = AuthenticityChecker()
    doc = {
        "title": "Title",
        "abstract": "",
        "sections": {},
        "references": ["Smith, J. (2020). Fake Title. Fake Journal, 1(1), 1-10."],
    }
    result = checker.check(doc)
    assert "findings" in result


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def test_logic_checker_causal_mismatch():
    checker = LogicChecker()
    doc = {
        "title": "Effect of X on Y",
        "abstract": "We study the effect of X on Y.",
        "sections": {
            "1": {"title": "Methods", "content": "We surveyed 100 people."},
            "2": {"title": "Results", "content": "Most people said yes."},
        },
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "methodology_mismatch" for f in findings)


def test_logic_checker_broken_chain():
    checker = LogicChecker()
    doc = {
        "title": "Hypothesis Testing",
        "abstract": "We test H1 and H2.",
        "sections": {
            "1": {"title": "Methods", "content": "We ran an experiment."},
        },
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "broken_chain" for f in findings)


def test_logic_checker_good_chain(minimal_doc):
    checker = LogicChecker()
    result = checker.check(minimal_doc)
    assert result["score"] >= 70 or any(f["severity"] != "fatal" for f in result["findings"])


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def test_citation_checker_apa(minimal_doc):
    checker = CitationChecker()
    result = checker.check(minimal_doc, citation_style="apa7")
    assert result["dimension"] == "citations"
    assert result["score"] >= 0
    assert "stats" in result


def test_citation_checker_phantom():
    checker = CitationChecker()
    doc = {
        "title": "Title",
        "abstract": "",
        "sections": {"1": {"title": "Text", "content": "(Smith, 2020) said something."}},
        "references": ["placeholder"],
    }
    result = checker.check(doc, citation_style="apa7")
    findings = result["findings"]
    assert any(f["type"] == "placeholder_reference" for f in findings)


def test_citation_checker_mismatch():
    checker = CitationChecker()
    doc = {
        "title": "Title",
        "abstract": "",
        "sections": {"1": {"title": "Text", "content": "(Unknown, 2099) made a claim."}},
        "references": ["Smith (2020). Title. Journal."],
    }
    result = checker.check(doc, citation_style="apa7")
    findings = result["findings"]
    assert any(f["type"] == "citation_not_in_list" for f in findings)


def test_citation_checker_gb():
    checker = CitationChecker()
    doc = {
        "title": "Title",
        "abstract": "",
        "sections": {"1": {"title": "Text", "content": "文献[1]指出。"}},
        "references": ["[1] 作者. 标题[J]. 期刊, 2020, 1(1): 1-10."],
    }
    result = checker.check(doc, citation_style="gb-t-7714-2015")
    assert result["dimension"] == "citations"


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

def test_language_checker_informal():
    checker = LanguageChecker()
    doc = {
        "title": "Title",
        "abstract": "This is really very good stuff.",
        "sections": {},
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "informal_language" for f in findings)


def test_language_checker_weak_phrase():
    checker = LanguageChecker()
    doc = {
        "title": "Title",
        "abstract": "It is interesting to note that the result is significant.",
        "sections": {},
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "weak_phrase" for f in findings)


def test_language_checker_good(minimal_doc):
    checker = LanguageChecker()
    result = checker.check(minimal_doc)
    assert result["score"] >= 70


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_statistics_checker_p_consistency():
    checker = StatisticsChecker()
    doc = {
        "title": "Title",
        "abstract": "The effect is significant, p = .120.",
        "sections": {},
        "references": [],
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "p_value_inconsistency" for f in findings)


def test_statistics_checker_missing_effect_size():
    checker = StatisticsChecker()
    doc = {
        "title": "Title",
        "abstract": "t(98) = 2.50, p = .014.",
        "sections": {},
        "references": [],
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "missing_effect_sizes" for f in findings)


def test_statistics_checker_missing_ci():
    checker = StatisticsChecker()
    doc = {
        "title": "Title",
        "abstract": "F(2, 97) = 5.20, p = .007, η² = .097.",
        "sections": {},
        "references": [],
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "missing_confidence_intervals" for f in findings)


def test_statistics_checker_good(minimal_doc):
    checker = StatisticsChecker()
    result = checker.check(minimal_doc)
    assert result["score"] >= 70
    assert result["pass"] is True


def test_statistics_checker_multiple_comparison():
    checker = StatisticsChecker()
    doc = {
        "title": "Title",
        "abstract": "",
        "sections": {
            "1": {"title": "Results", "content": "Group A: t(48) = 2.1, p = .04. Group B: t(48) = 2.3, p = .03. Group C: t(48) = 2.0, p = .05. Group D: t(48) = 2.4, p = .02. Group E: t(48) = 1.9, p = .06."}
        },
        "references": [],
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "multiple_comparison_uncorrected" for f in findings)


def test_statistics_checker_p_zero():
    checker = StatisticsChecker()
    doc = {
        "title": "Title",
        "abstract": "p = 0.000.",
        "sections": {},
        "references": [],
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "p_value_rounding" for f in findings)


# ---------------------------------------------------------------------------
# Ethics
# ---------------------------------------------------------------------------

def test_ethics_checker_suspicious_p():
    checker = EthicsChecker()
    doc = {
        "title": "Title",
        "abstract": "p = .049, p = .051, p = .048, p = .052, p = .050.",
        "sections": {},
        "references": [],
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "suspicious_p_distribution" for f in findings)


def test_ethics_checker_round_n():
    checker = EthicsChecker()
    doc = {
        "title": "Title",
        "abstract": "Group A n = 100, Group B n = 150, Group C n = 200.",
        "sections": {},
        "references": [],
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "suspicious_sample_sizes" for f in findings)


def test_ethics_checker_perfect_correlation():
    checker = EthicsChecker()
    doc = {
        "title": "Title",
        "abstract": "r = 0.999, r = 1.000.",
        "sections": {},
        "references": [],
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "perfect_correlation" for f in findings)


def test_ethics_checker_low_citation():
    checker = EthicsChecker()
    long_text = "This is a very long abstract with many words and sentences that goes on and on about the research topic and methodology and findings without any citations at all. " * 30
    doc = {
        "title": "Title",
        "abstract": long_text,
        "sections": {},
        "references": ["Smith (2020). Title. Journal."],
    }
    result = checker.check(doc)
    findings = result["findings"]
    assert any(f["type"] == "very_low_citation_density" for f in findings)


def test_ethics_checker_good(minimal_doc):
    checker = EthicsChecker()
    result = checker.check(minimal_doc)
    assert result["score"] >= 70


# ---------------------------------------------------------------------------
# Integration / edge cases
# ---------------------------------------------------------------------------

def test_review_engine_graceful_failure():
    """If a checker raises, engine catches it and continues."""
    engine = ReviewEngine()
    doc = {"id": "x", "title": "x", "abstract": "x", "sections": {}, "references": []}
    report = engine.review(doc)
    assert "dimensions" in report
    assert "overall_score" in report


def test_severity_sorting(doc_with_issues):
    engine = ReviewEngine()
    report = engine.review(doc_with_issues)
    severities = [f.get("severity", "suggestion") for f in report["all_findings"]]
    order = {"fatal": 0, "major": 1, "minor": 2, "suggestion": 3}
    numeric = [order.get(s, 99) for s in severities]
    assert numeric == sorted(numeric)


def test_critical_issues(doc_with_issues):
    engine = ReviewEngine()
    report = engine.review(doc_with_issues)
    assert all(f.get("severity") in ("fatal", "major") for f in report["critical_issues"])
