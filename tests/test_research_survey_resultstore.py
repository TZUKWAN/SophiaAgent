"""Tests for SurveyEngine ↔ ResultStore integration (P1.4c).

These tests exercise the new behaviours layered onto the engine in P1.4c:

- Every public method returns a ``result_id`` when a ``ResultStore`` is
  configured (cronbach, factor_analysis, item_analysis, sample_size,
  likert_analysis).
- Column-name selectors (``items_cols``, ``data_cols``, ``total_score_col``)
  resolve against an attached DataFrame.
- Lineage is recorded back to upstream ``res_*`` references in args.
- Legacy list-based inputs still work and still get a ``result_id``.
- Errors do NOT produce ``result_id`` and do NOT persist.
- ``params`` are sanitized (no huge arrays inside the SQLite blob).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sophia.research.result_store import ResultStore
from sophia.research.seed import GlobalSeed
from sophia.research.survey import SurveyEngine


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
    return SurveyEngine(store=store)


@pytest.fixture(autouse=True)
def _reset_seed():
    GlobalSeed.reset()
    yield
    GlobalSeed.reset()


def _reliable_items(n_items=5, n_resp=50, seed=42):
    """Highly-correlated items so Cronbach's alpha > 0.7."""
    rng = np.random.default_rng(seed)
    base = rng.normal(3, 0.5, n_resp)
    items = []
    for _ in range(n_items):
        items.append((base + rng.normal(0, 0.2, n_resp)).tolist())
    return items


def _likert_panel(n_resp=80, n_items=4, seed=77):
    """4 Likert items (1-5), latent + noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(3, 0.7, n_resp)
    cols = {}
    for j in range(n_items):
        item = latent + rng.normal(0, 0.5, n_resp)
        item = np.clip(np.round(item), 1, 5).astype(int)
        cols[f"q{j + 1}"] = item.tolist()
    return pd.DataFrame(cols)


def _factor_panel(n_resp=100, n_vars=6, n_factors=2, seed=11):
    """Synthetic factor structure: 6 items load on 2 factors."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n_resp)
    f2 = rng.normal(0, 1, n_resp)
    cols = {}
    # First 3 load on F1
    for j in range(3):
        cols[f"x{j + 1}"] = (0.7 * f1 + rng.normal(0, 0.5, n_resp)).tolist()
    # Last 3 load on F2
    for j in range(3, 6):
        cols[f"x{j + 1}"] = (0.7 * f2 + rng.normal(0, 0.5, n_resp)).tolist()
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------
# result_id round-trip
# ----------------------------------------------------------------------
class TestResultIdRoundTrip:
    def test_cronbach_returns_result_id(self, engine):
        out = json.loads(engine.cronbach({"items": _reliable_items()}))
        assert "result_id" in out
        assert out["result_id"].startswith("res_")

    def test_factor_analysis_returns_result_id(self, engine):
        df = _factor_panel()
        out = json.loads(engine.factor_analysis({
            "data": df.values.tolist(),
            "n_factors": 2,
        }))
        assert "result_id" in out

    def test_item_analysis_returns_result_id(self, engine):
        out = json.loads(engine.item_analysis({"items": _reliable_items()}))
        assert "result_id" in out

    def test_sample_size_returns_result_id(self, engine):
        out = json.loads(engine.sample_size({
            "population": 10000,
            "margin_error": 0.05,
            "confidence": 0.95,
        }))
        assert "result_id" in out

    def test_likert_analysis_returns_result_id(self, engine):
        df = _likert_panel()
        out = json.loads(engine.likert_analysis({
            "data": df.values.tolist(),
            "scale_min": 1,
            "scale_max": 5,
        }))
        assert "result_id" in out

    def test_no_store_no_result_id(self):
        plain = SurveyEngine(store=None)
        out = json.loads(plain.cronbach({"items": _reliable_items()}))
        assert "result_id" not in out
        assert "alpha" in out

    def test_cronbach_error_does_not_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.cronbach({"items": [[1, 2, 3]]}))  # 1 item
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_sample_size_invalid_error_no_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.sample_size({"margin_error": 1.5}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before


# ----------------------------------------------------------------------
# Column-name resolution
# ----------------------------------------------------------------------
class TestColumnResolution:
    def test_cronbach_with_items_cols_via_result_id(self, engine, store):
        df = _likert_panel()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.cronbach({
            "result_id": src,
            "items_cols": ["q1", "q2", "q3", "q4"],
        }))
        assert "alpha" in out
        assert out["n_items"] == 4
        assert out["n_responses"] == 80
        # item_names should default to column names
        assert "q1" in out["item_total_corr"]
        assert "result_id" in out

    def test_factor_analysis_with_data_cols(self, engine, store):
        df = _factor_panel()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.factor_analysis({
            "result_id": src,
            "data_cols": [f"x{i + 1}" for i in range(6)],
            "n_factors": 2,
            "rotation": "varimax",
        }))
        assert out["n_factors"] == 2
        assert out["n_vars"] == 6
        assert out["n_obs"] == 100
        assert "loadings" in out

    def test_item_analysis_with_items_cols(self, engine, store):
        df = _likert_panel()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.item_analysis({
            "result_id": src,
            "items_cols": ["q1", "q2", "q3", "q4"],
        }))
        assert "items" in out
        assert len(out["items"]) == 4
        # item_names propagated from items_cols
        names = [it["item"] for it in out["items"]]
        assert names == ["q1", "q2", "q3", "q4"]

    def test_item_analysis_with_total_score_col(self, engine, store):
        df = _likert_panel()
        df["total"] = df[["q1", "q2", "q3", "q4"]].sum(axis=1)
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.item_analysis({
            "result_id": src,
            "items_cols": ["q1", "q2", "q3", "q4"],
            "total_score_col": "total",
        }))
        assert "items" in out
        assert out["overall"]["n_responses"] == 80

    def test_likert_analysis_with_data_cols(self, engine, store):
        df = _likert_panel()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.likert_analysis({
            "result_id": src,
            "data_cols": ["q1", "q2", "q3", "q4"],
            "scale_min": 1,
            "scale_max": 5,
        }))
        assert out["n_respondents"] == 80
        assert out["n_items"] == 4
        names = [it["item"] for it in out["items"]]
        assert names == ["q1", "q2", "q3", "q4"]

    def test_cronbach_matches_direct_input(self, engine, store):
        """Column-resolved cronbach should match direct list input."""
        df = _likert_panel()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out_via_cols = json.loads(engine.cronbach({
            "result_id": src,
            "items_cols": ["q1", "q2", "q3", "q4"],
        }))
        items_direct = [df[c].tolist() for c in ["q1", "q2", "q3", "q4"]]
        out_direct = json.loads(engine.cronbach({"items": items_direct}))
        assert out_via_cols["alpha"] == pytest.approx(out_direct["alpha"], rel=1e-6)


# ----------------------------------------------------------------------
# Lineage
# ----------------------------------------------------------------------
class TestLineage:
    def test_parents_recorded_from_result_id(self, engine, store):
        df = _likert_panel()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.cronbach({
            "result_id": src,
            "items_cols": ["q1", "q2", "q3", "q4"],
        }))
        meta = store.get_metadata(out["result_id"])
        assert src in meta["parents"]

    def test_no_parents_for_inline_input(self, engine, store):
        out = json.loads(engine.cronbach({"items": _reliable_items()}))
        meta = store.get_metadata(out["result_id"])
        assert meta["parents"] == []

    def test_sample_size_no_parents(self, engine, store):
        out = json.loads(engine.sample_size({
            "population": 5000,
            "margin_error": 0.04,
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["parents"] == []


# ----------------------------------------------------------------------
# Persistence shape
# ----------------------------------------------------------------------
class TestPersistenceShape:
    def test_kind_is_result_for_cronbach(self, engine, store):
        out = json.loads(engine.cronbach({"items": _reliable_items()}))
        meta = store.get_metadata(out["result_id"])
        assert meta["kind"] == "result"
        assert meta["tool"] == "research_cronbach"

    def test_params_sanitized_for_large_items(self, engine, store):
        # 5 items × 50 responses = nested list -> should be summarized
        out = json.loads(engine.cronbach({"items": _reliable_items()}))
        meta = store.get_metadata(out["result_id"])
        assert isinstance(meta["params"]["items"], str)
        assert "nested" in meta["params"]["items"] or "list" in meta["params"]["items"]

    def test_params_sanitized_for_factor_data(self, engine, store):
        df = _factor_panel()
        out = json.loads(engine.factor_analysis({
            "data": df.values.tolist(),
            "n_factors": 2,
        }))
        meta = store.get_metadata(out["result_id"])
        # data has 100 rows × 6 cols = 600 entries
        assert isinstance(meta["params"]["data"], str)

    def test_scalar_params_kept_in_metadata(self, engine, store):
        out = json.loads(engine.sample_size({
            "population": 12000,
            "margin_error": 0.03,
            "confidence": 0.99,
            "proportion": 0.4,
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["params"]["population"] == 12000
        assert meta["params"]["margin_error"] == 0.03
        assert meta["params"]["confidence"] == 0.99
        assert meta["params"]["proportion"] == 0.4

    def test_tool_name_per_method(self, engine, store):
        rid1 = json.loads(engine.cronbach({"items": _reliable_items()}))["result_id"]
        rid2 = json.loads(engine.item_analysis({"items": _reliable_items()}))["result_id"]
        rid3 = json.loads(engine.sample_size({"population": 1000}))["result_id"]
        df = _likert_panel()
        rid4 = json.loads(engine.likert_analysis({
            "data": df.values.tolist(),
        }))["result_id"]
        rid5 = json.loads(engine.factor_analysis({
            "data": _factor_panel().values.tolist(),
            "n_factors": 2,
        }))["result_id"]
        assert store.get_metadata(rid1)["tool"] == "research_cronbach"
        assert store.get_metadata(rid2)["tool"] == "research_item_analysis"
        assert store.get_metadata(rid3)["tool"] == "research_sample_size"
        assert store.get_metadata(rid4)["tool"] == "research_likert_analysis"
        assert store.get_metadata(rid5)["tool"] == "research_factor_analysis"


# ----------------------------------------------------------------------
# Legacy compatibility
# ----------------------------------------------------------------------
class TestLegacyCompat:
    def test_cronbach_legacy_items(self, engine):
        out = json.loads(engine.cronbach({
            "items": _reliable_items(),
            "item_names": ["a", "b", "c", "d", "e"],
        }))
        assert "alpha" in out
        assert "result_id" in out
        assert "a" in out["item_total_corr"]

    def test_factor_analysis_legacy(self, engine):
        df = _factor_panel()
        out = json.loads(engine.factor_analysis({
            "data": df.values.tolist(),
            "n_factors": 2,
            "rotation": "varimax",
        }))
        assert "loadings" in out
        assert "result_id" in out

    def test_item_analysis_legacy(self, engine):
        out = json.loads(engine.item_analysis({
            "items": _reliable_items(),
            "item_names": ["q1", "q2", "q3", "q4", "q5"],
        }))
        assert "items" in out
        assert "result_id" in out

    def test_sample_size_legacy_minimal(self, engine):
        out = json.loads(engine.sample_size({"margin_error": 0.05}))
        assert "n_simple" in out
        assert "result_id" in out

    def test_likert_legacy(self, engine):
        df = _likert_panel()
        out = json.loads(engine.likert_analysis({
            "data": df.values.tolist(),
            "scale_min": 1,
            "scale_max": 5,
            "item_names": ["q1", "q2", "q3", "q4"],
        }))
        assert "items" in out
        assert "result_id" in out
        assert "inter_item_consistency" in out
