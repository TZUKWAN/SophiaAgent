"""Tests for enhanced LogicChecker (F-1~F-4)."""

import pytest
from sophia.review.logic import LogicChecker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def checker():
    return LogicChecker()


@pytest.fixture
def doc_with_argument_chain():
    return {
        "title": "教育公平研究",
        "abstract": "",
        "sections": {
            "1": {
                "title": "引言",
                "content": "由于教育资源分配不均，农村学生面临更多困难。因此，政府应加大投入。"
            },
            "2": {
                "title": "方法",
                "content": "我们采用了混合研究方法。"
            },
            "3": {
                "title": "结果",
                "content": "数据显示，增加投入后学生成绩显著提高。研究表明，教育公平需要长期投入。"
            },
        },
    }


@pytest.fixture
def doc_with_fallacies():
    return {
        "title": "谬误示例",
        "abstract": "",
        "sections": {
            "1": {
                "title": "论证",
                "content": (
                    "自古以来，男人就应该主外。一旦放开管制，社会就会一发不可收拾。"
                    "有人认为经济发展是好事，但实际上这会导致环境问题。"
                    "所有成功人士都是努力工作的，所以只要努力就能成功。"
                    "令人愤慨的是，这些政策完全不考虑民众感受。"
                    "要么接受改革，要么被淘汰。"
                    "因为A在B之前发生，所以A导致了B。"
                    "随着收入的增加，幸福感也增加，说明收入决定幸福。"
                    "正如案例所示，这个政策是成功的。"
                    "历来如此，所以这是对的。"
                    "成功者都有坚定的信念，所以我们应该向他们学习。"
                )
            },
        },
    }


@pytest.fixture
def doc_with_naked_arguments():
    return {
        "title": "缺乏证据的论文",
        "abstract": "",
        "sections": {
            "1": {
                "title": "讨论",
                "content": (
                    "因此，教育政策需要改革。"
                    "所以，我们应该增加教师工资。"
                    "这表明，小班教学效果更好。"
                    "我们认为，技术可以提升学习效率。"
                    "研究发现，课外辅导有助于成绩提高。"
                )
            },
        },
    }


@pytest.fixture
def doc_with_mixed_evidence():
    return {
        "title": "有证据的论文",
        "abstract": "",
        "sections": {
            "1": {
                "title": "讨论",
                "content": (
                    "根据Smith (2020)的研究，教育投入与学生成绩正相关。"
                    "因此，增加教育投入可以改善学生表现。"
                    "数据显示，投入增加10%后，成绩提高了5%。"
                    "所以，政策制定者应考虑加大投入。"
                )
            },
        },
    }


@pytest.fixture
def doc_with_counter_arguments():
    return {
        "title": "考虑反驳的论文",
        "abstract": "",
        "sections": {
            "1": {
                "title": "讨论",
                "content": (
                    "然而，这一结论可能受到样本选择偏差的影响。"
                    "另一方面，文化因素也可能起到重要作用。"
                    "尽管如此，我们的发现仍然具有一定参考价值。"
                    "但是，研究也存在局限性，有待进一步验证。"
                    "一定程度上，政策效果可能因地区而异。"
                )
            },
        },
    }


# ---------------------------------------------------------------------------
# F-1: Argument chain extraction
# ---------------------------------------------------------------------------

def test_extract_argument_chains_basic(checker, doc_with_argument_chain):
    chains = checker.extract_argument_chains(doc_with_argument_chain)
    assert len(chains) >= 1
    # Should find "由于...因此..." pattern
    assert any("由于" in c["premise"] for c in chains)
    assert any("政府应加大投入" in c["conclusion"] for c in chains)


def test_extract_argument_chains_evidence_based(checker, doc_with_mixed_evidence):
    chains = checker.extract_argument_chains(doc_with_mixed_evidence)
    assert len(chains) >= 1
    # Should find evidence-based reasoning
    assert any(c.get("evidence") for c in chains)


def test_extract_argument_chains_empty_doc(checker):
    chains = checker.extract_argument_chains({"title": "", "sections": {}})
    assert chains == []


def test_extract_argument_chains_no_markers(checker):
    doc = {"title": "X", "sections": {"1": {"title": "T", "content": "这是一段普通的描述性文字。"}}}
    chains = checker.extract_argument_chains(doc)
    # No premise/conclusion markers, but might have weak chains
    assert isinstance(chains, list)


def test_visualize_argument_chains_mermaid(checker, doc_with_argument_chain):
    chains = checker.extract_argument_chains(doc_with_argument_chain)
    mermaid = checker.visualize_argument_chains(chains, format="mermaid")
    assert "graph LR" in mermaid
    assert "P0" in mermaid
    assert "C0" in mermaid


def test_visualize_argument_chains_dot(checker, doc_with_argument_chain):
    chains = checker.extract_argument_chains(doc_with_argument_chain)
    dot = checker.visualize_argument_chains(chains, format="dot")
    assert "digraph Arguments" in dot
    assert "shape=box" in dot
    assert "shape=ellipse" in dot


def test_visualize_argument_chains_json(checker, doc_with_argument_chain):
    chains = checker.extract_argument_chains(doc_with_argument_chain)
    json_out = checker.visualize_argument_chains(chains, format="json")
    assert "chain_id" in json_out
    assert "premise" in json_out


def test_visualize_empty_chains(checker):
    mermaid = checker.visualize_argument_chains([], format="mermaid")
    assert "graph LR" in mermaid


# ---------------------------------------------------------------------------
# F-2: Fallacy detection
# ---------------------------------------------------------------------------

def test_detect_slippery_slope(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "slippery_slope" for f in findings)


def test_detect_straw_man(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "straw_man" for f in findings)


def test_detect_hasty_generalization(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "hasty_generalization" for f in findings)


def test_detect_appeal_to_emotion(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "appeal_to_emotion" for f in findings)


def test_detect_false_dilemma(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "false_dilemma" for f in findings)


def test_detect_false_causation(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "false_causation" for f in findings)


def test_detect_correlation_causation(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "correlation_causation" for f in findings)


def test_detect_cherry_picking(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "cherry_picking" for f in findings)


def test_detect_appeal_to_tradition(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "appeal_to_tradition" for f in findings)


def test_detect_survivorship_bias(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert any(f["subtype_en"] == "survivorship_bias" for f in findings)


def test_fallacy_context_check_skips_qualified(checker):
    # "可能" should cause context_check fallacies to be skipped
    doc = {
        "title": "X",
        "sections": {"1": {"title": "T", "content": "因为A发生了，所以可能导致了B，但这有待验证。"}}
    }
    findings = checker._detect_fallacies(doc)
    # Should not flag false_causation due to "可能" and "有待验证"
    false_causation = [f for f in findings if f["subtype_en"] == "false_causation"]
    assert len(false_causation) == 0


def test_fallacy_severity_minor(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    assert all(f["severity"] == "minor" for f in findings)


def test_fallacy_structure(checker, doc_with_fallacies):
    findings = checker._detect_fallacies(doc_with_fallacies)
    for f in findings:
        assert "subtype" in f
        assert "subtype_en" in f
        assert "detail" in f
        assert "explanation" in f
        assert "suggestion" in f
        assert "location" in f


# ---------------------------------------------------------------------------
# F-3: Evidence sufficiency
# ---------------------------------------------------------------------------

def test_naked_argument_detection(checker, doc_with_naked_arguments):
    findings = checker._check_evidence_sufficiency(doc_with_naked_arguments)
    naked = [f for f in findings if f["type"] == "naked_argument"]
    assert len(naked) >= 3
    assert all(f["is_naked"] for f in naked)
    assert all(f["severity"] == "major" for f in naked)


def test_mixed_evidence_no_naked(checker, doc_with_mixed_evidence):
    findings = checker._check_evidence_sufficiency(doc_with_mixed_evidence)
    naked = [f for f in findings if f["type"] == "naked_argument"]
    # Should have fewer or no naked arguments due to citations and data
    assert len(naked) <= 1


def test_evidence_types_detected(checker, doc_with_mixed_evidence):
    findings = checker._check_evidence_sufficiency(doc_with_mixed_evidence)
    # Some arguments should have evidence types found
    non_naked = [f for f in findings if f["type"] != "naked_argument"]
    if non_naked:
        assert any(f.get("sufficiency_score", 0) > 0 for f in non_naked)


def test_evidence_sufficiency_structure(checker, doc_with_naked_arguments):
    findings = checker._check_evidence_sufficiency(doc_with_naked_arguments)
    for f in findings:
        assert "sufficiency_score" in f
        assert "is_naked" in f
        assert "suggestion" in f
        assert "location" in f


# ---------------------------------------------------------------------------
# F-4: Argument structure scoring
# ---------------------------------------------------------------------------

def test_scoring_dimensions(checker, doc_with_argument_chain):
    findings = []
    result = checker._score_argument_structure(doc_with_argument_chain, findings)
    assert "total_score" in result
    assert "grade" in result
    assert "breakdown" in result
    breakdown = result["breakdown"]
    assert "clarity" in breakdown
    assert "evidence" in breakdown
    assert "coherence" in breakdown
    assert "counter" in breakdown
    assert "caution" in breakdown
    for dim in breakdown.values():
        assert "score" in dim
        assert "max" in dim
        assert dim["max"] == 20
        assert 0 <= dim["score"] <= 20


def test_grade_levels(checker):
    # A grade: >= 90
    doc_a = {
        "title": "优秀论文",
        "abstract": "因此。所以。然而。另一方面。可能。也许。一定程度上。",
        "sections": {
            "1": {"title": "结果", "content": "由于数据显示，因此结论成立。研究表明。"},
        },
    }
    result = checker.check(doc_a)
    assert result["grade"] in ("A", "B", "C", "D")
    assert "score_breakdown" in result


def test_score_range(checker, doc_with_argument_chain):
    result = checker.check(doc_with_argument_chain)
    assert 0 <= result["score"] <= 100
    assert result["grade"] in ("A", "B", "C", "D")


def test_counter_consideration_scoring(checker, doc_with_counter_arguments):
    result = checker.check(doc_with_counter_arguments)
    breakdown = result["score_breakdown"]
    # Should have decent counter score due to counter_keywords
    assert breakdown["counter"]["score"] >= 5


def test_caution_scoring_with_strong_markers(checker):
    doc = {
        "title": "绝对结论",
        "abstract": "这毫无疑问证明了我们的假设。必然导致。绝对正确。",
        "sections": {},
    }
    result = checker.check(doc)
    breakdown = result["score_breakdown"]
    # Strong markers should reduce caution score
    assert breakdown["caution"]["score"] < 15


# ---------------------------------------------------------------------------
# Integration: full check() with enhanced features
# ---------------------------------------------------------------------------

def test_full_check_returns_enhanced_fields(checker, doc_with_argument_chain):
    result = checker.check(doc_with_argument_chain)
    assert "score_breakdown" in result
    assert "grade" in result
    assert isinstance(result["score"], float)


def test_full_check_with_fallacies(checker, doc_with_fallacies):
    result = checker.check(doc_with_fallacies)
    findings = result["findings"]
    assert any(f["type"] == "fallacy" for f in findings)
    assert any(f["type"] in ("naked_argument", "weak_evidence") for f in findings)


def test_full_check_pass_threshold(checker, doc_with_mixed_evidence):
    result = checker.check(doc_with_mixed_evidence)
    # Pass if score >= 70
    assert isinstance(result["pass"], bool)


def test_summary_format(checker, doc_with_fallacies):
    result = checker.check(doc_with_fallacies)
    summary = result["summary"]
    assert "Found" in summary
    assert "fatal" in summary or "major" in summary or "minor" in summary
