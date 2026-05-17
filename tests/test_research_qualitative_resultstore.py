"""Tests for QualitativeEngine ↔ ResultStore integration (P1.4d).

These tests exercise the new behaviours layered onto the engine in P1.4d:

- Every public method returns a ``result_id`` when a ``ResultStore`` is
  configured (thematic, content, grounded_code, sentiment, coding_reliability).
- Errors do NOT produce ``result_id`` and do NOT persist.
- ``params`` are sanitized (no huge text lists in the SQLite blob).
- Legacy inputs still work and still get a ``result_id``.

This file deliberately exercises only the NLP fallback paths (no LLM provider
attached) so the tests stay deterministic. Full 4-pass LLM iterations are
covered in P5.1/P5.2.
"""
from __future__ import annotations

import json

import pytest

from sophia.research.qualitative import QualitativeEngine
from sophia.research.result_store import ResultStore
from sophia.research.seed import GlobalSeed


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def store(workspace):
    return ResultStore(workspace)


@pytest.fixture
def engine(store):
    # No provider -> use keyword fallback paths
    return QualitativeEngine(provider=None, store=store)


@pytest.fixture(autouse=True)
def _reset_seed():
    GlobalSeed.reset()
    yield
    GlobalSeed.reset()


def _sample_texts():
    return [
        "Online learning has many benefits, including flexibility and access to diverse courses.",
        "However, online classes often lack the personal interaction of traditional classrooms.",
        "Students report that virtual environments can feel isolating without peer support.",
        "Quality online programs incorporate interactive discussions and real-time feedback.",
        "Time management remains a crucial skill for success in online learning environments.",
        "The flexibility of online study helps balance work and personal life commitments.",
        "Some learners thrive in self-paced courses while others need structured schedules.",
        "Technology issues can disrupt learning and create frustration for both students and teachers.",
    ]


def _interview_texts():
    return [
        "I really enjoyed this product, it was excellent and worth every dollar.",
        "Terrible experience, the worst I have ever had, very disappointed.",
        "It was okay, nothing special but not bad either.",
        "Amazing quality and great customer service, I would highly recommend it.",
        "Poor build quality, broke after one week, never buying again.",
    ]


# ----------------------------------------------------------------------
# result_id round-trip
# ----------------------------------------------------------------------
class TestResultIdRoundTrip:
    def test_thematic_returns_result_id(self, engine):
        out = json.loads(engine.thematic({"texts": _sample_texts()}))
        assert "result_id" in out
        assert out["result_id"].startswith("res_")
        assert "themes" in out

    def test_content_returns_result_id(self, engine):
        out = json.loads(engine.content({"texts": _sample_texts()}))
        assert "result_id" in out
        assert "word_frequencies" in out

    def test_grounded_code_open_returns_result_id(self, engine):
        out = json.loads(engine.grounded_code({
            "texts": _sample_texts(),
            "stage": "open",
        }))
        assert "result_id" in out
        assert out["stage"] == "open"

    def test_grounded_code_axial_returns_result_id(self, engine):
        out = json.loads(engine.grounded_code({
            "texts": _sample_texts(),
            "stage": "axial",
            "existing_codes": ["learning", "flexibility", "interaction", "technology"],
        }))
        assert "result_id" in out
        assert out["stage"] == "axial"

    def test_grounded_code_selective_returns_result_id(self, engine):
        out = json.loads(engine.grounded_code({
            "texts": _sample_texts(),
            "stage": "selective",
            "existing_codes": ["learning", "flexibility", "interaction"],
        }))
        assert "result_id" in out
        assert out["stage"] == "selective"

    def test_sentiment_returns_result_id(self, engine):
        out = json.loads(engine.sentiment({"texts": _interview_texts()}))
        assert "result_id" in out
        assert "sentiments" in out

    def test_coding_reliability_returns_result_id(self, engine):
        out = json.loads(engine.coding_reliability({
            "coder1": ["A", "B", "A", "C", "B", "A"],
            "coder2": ["A", "B", "B", "C", "B", "A"],
        }))
        assert "result_id" in out
        assert "kappa" in out

    def test_no_store_no_result_id(self):
        plain = QualitativeEngine(provider=None, store=None)
        out = json.loads(plain.content({"texts": _sample_texts()}))
        assert "result_id" not in out
        assert "word_frequencies" in out

    def test_thematic_no_texts_error_no_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.thematic({"texts": []}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_content_no_texts_error_no_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.content({"texts": []}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_grounded_code_unknown_stage_error_no_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.grounded_code({
            "texts": _sample_texts(),
            "stage": "made_up_stage",
        }))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_sentiment_no_texts_error_no_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.sentiment({"texts": []}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_coding_reliability_length_mismatch_error_no_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.coding_reliability({
            "coder1": ["A", "B", "C"],
            "coder2": ["A", "B"],
        }))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before


# ----------------------------------------------------------------------
# Persistence shape
# ----------------------------------------------------------------------
class TestPersistenceShape:
    def test_kind_is_result_for_thematic(self, engine, store):
        out = json.loads(engine.thematic({"texts": _sample_texts()}))
        meta = store.get_metadata(out["result_id"])
        assert meta["kind"] == "result"
        assert meta["tool"] == "research_thematic"

    def test_tool_name_per_method(self, engine, store):
        rid1 = json.loads(engine.content({"texts": _sample_texts()}))["result_id"]
        rid2 = json.loads(engine.grounded_code({
            "texts": _sample_texts(), "stage": "open",
        }))["result_id"]
        rid3 = json.loads(engine.sentiment({
            "texts": _interview_texts(),
        }))["result_id"]
        rid4 = json.loads(engine.coding_reliability({
            "coder1": ["A", "B"], "coder2": ["A", "B"],
        }))["result_id"]
        assert store.get_metadata(rid1)["tool"] == "research_content"
        assert store.get_metadata(rid2)["tool"] == "research_grounded_code"
        assert store.get_metadata(rid3)["tool"] == "research_sentiment"
        assert store.get_metadata(rid4)["tool"] == "research_coding_reliability"

    def test_params_sanitized_for_large_text_lists(self, engine, store):
        # 100 texts, each 100+ chars -> total_chars > 4000 -> summarized
        big_texts = [
            f"Text segment {i}: " + ("learning environment online interaction " * 8)
            for i in range(100)
        ]
        out = json.loads(engine.content({"texts": big_texts}))
        meta = store.get_metadata(out["result_id"])
        assert isinstance(meta["params"]["texts"], str)
        assert "list" in meta["params"]["texts"]

    def test_params_sanitized_for_long_coder_lists(self, engine, store):
        coder1 = ["A", "B", "A", "C"] * 30   # 120 entries
        coder2 = ["A", "B", "B", "C"] * 30
        out = json.loads(engine.coding_reliability({
            "coder1": coder1,
            "coder2": coder2,
        }))
        meta = store.get_metadata(out["result_id"])
        assert isinstance(meta["params"]["coder1"], str)
        assert "list" in meta["params"]["coder1"]
        assert isinstance(meta["params"]["coder2"], str)

    def test_scalar_params_kept_in_metadata(self, engine, store):
        out = json.loads(engine.content({
            "texts": _sample_texts(),
            "min_freq": 3,
            "window": 4,
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["params"]["min_freq"] == 3
        assert meta["params"]["window"] == 4

    def test_stage_param_kept(self, engine, store):
        out = json.loads(engine.grounded_code({
            "texts": _sample_texts(),
            "stage": "selective",
            "existing_codes": ["learning", "flexibility"],
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["params"]["stage"] == "selective"


# ----------------------------------------------------------------------
# Lineage
# ----------------------------------------------------------------------
class TestLineage:
    def test_no_parents_for_inline_texts(self, engine, store):
        out = json.loads(engine.content({"texts": _sample_texts()}))
        meta = store.get_metadata(out["result_id"])
        assert meta["parents"] == []

    def test_no_parents_for_inline_coder_lists(self, engine, store):
        out = json.loads(engine.coding_reliability({
            "coder1": ["A", "B"], "coder2": ["A", "B"],
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["parents"] == []


# ----------------------------------------------------------------------
# Legacy compatibility
# ----------------------------------------------------------------------
class TestLegacyCompat:
    def test_thematic_legacy_inductive(self, engine):
        out = json.loads(engine.thematic({
            "texts": _sample_texts(),
            "approach": "inductive",
            "n_themes": 3,
        }))
        assert "themes" in out
        assert "result_id" in out

    def test_thematic_legacy_deductive(self, engine):
        out = json.loads(engine.thematic({
            "texts": _sample_texts(),
            "approach": "deductive",
            "existing_themes": ["flexibility", "interaction", "technology"],
        }))
        assert "themes" in out
        assert "result_id" in out

    def test_content_legacy_with_keywords(self, engine):
        out = json.loads(engine.content({
            "texts": _sample_texts(),
            "keywords": ["learning", "flexibility", "online"],
            "min_freq": 1,
        }))
        assert "keyword_frequencies" in out
        assert "result_id" in out

    def test_grounded_code_legacy_open(self, engine):
        out = json.loads(engine.grounded_code({
            "texts": _sample_texts(),
            "stage": "open",
        }))
        assert "codes" in out
        assert "result_id" in out

    def test_sentiment_legacy(self, engine):
        out = json.loads(engine.sentiment({"texts": _interview_texts()}))
        assert "overall_distribution" in out
        assert "result_id" in out

    def test_coding_reliability_legacy(self, engine):
        out = json.loads(engine.coding_reliability({
            "coder1": [1, 2, 1, 3, 2, 1, 2, 3],
            "coder2": [1, 2, 2, 3, 2, 1, 2, 3],
        }))
        assert "kappa" in out
        assert "agreement_rate" in out
        assert "result_id" in out
