"""Tests for StatisticalEngine ↔ ResultStore integration (P1.4).

These tests exercise the new behaviours layered onto the engine in P1.4:

- Every method returns a ``result_id`` when a ``ResultStore`` is configured.
- Column-name selectors (``*_col`` args) resolve against an attached
  DataFrame (loaded via ``result_id`` or inline ``data`` dict).
- Lineage is recorded back to upstream ``res_*`` references in args.
- Recursive ``auto_test`` only stores the outer result, not the inner test.
- Legacy list-of-numbers inputs still work and still get a ``result_id``.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from sophia.research.pipeline import ExperimentPipeline
from sophia.research.result_store import ResultStore
from sophia.research.seed import GlobalSeed
from sophia.research.statistics import StatisticalEngine


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
    return StatisticalEngine(store=store)


@pytest.fixture
def pipeline(workspace, store):
    return ExperimentPipeline(workspace, store=store)


@pytest.fixture
def sample_csv(workspace):
    """Write a sample CSV into the workspace and return its relative path."""
    path = os.path.join(workspace, "sample.csv")
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, 60),
        "y": rng.normal(0.5, 1, 60),
        "group": ["A"] * 30 + ["B"] * 30,
        "score": rng.normal(75, 10, 60),
    })
    df.to_csv(path, index=False)
    return "sample.csv"


@pytest.fixture(autouse=True)
def _reset_seed():
    GlobalSeed.reset()
    yield
    GlobalSeed.reset()


# ----------------------------------------------------------------------
# result_id round-trip
# ----------------------------------------------------------------------
class TestResultIdRoundTrip:
    def test_describe_returns_result_id(self, engine):
        out = json.loads(engine.describe({"data": [1, 2, 3, 4, 5]}))
        assert "result_id" in out
        assert out["result_id"].startswith("res_")

    def test_ttest_returns_result_id(self, engine):
        out = json.loads(engine.ttest({
            "group1": [1, 2, 3, 4, 5],
            "group2": [3, 4, 5, 6, 7],
        }))
        assert "result_id" in out

    def test_correlation_returns_result_id(self, engine):
        out = json.loads(engine.correlation({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 3, 4, 5, 6],
        }))
        assert "result_id" in out

    def test_regression_returns_result_id(self, engine):
        out = json.loads(engine.regression({
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "X": [1.0, 2.1, 3.0, 3.9, 5.1],
        }))
        assert "result_id" in out

    def test_no_store_no_result_id(self):
        plain = StatisticalEngine(store=None)
        out = json.loads(plain.ttest({
            "group1": [1, 2, 3], "group2": [4, 5, 6],
        }))
        assert "result_id" not in out
        assert "t" in out

    def test_error_does_not_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.ttest({"group1": [], "group2": []}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before


# ----------------------------------------------------------------------
# Column-name resolution
# ----------------------------------------------------------------------
class TestColumnResolution:
    def test_ttest_with_group_cols_via_result_id(self, engine, store):
        df = pd.DataFrame({
            "treatment": [10, 12, 14, 16, 18, 20] * 5,
            "control":   [ 8,  9, 10, 11, 12, 13] * 5,
        })
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.ttest({
            "result_id": src,
            "group1_col": "treatment",
            "group2_col": "control",
        }))
        assert "t" in out
        assert "result_id" in out
        # Compare against direct call to confirm correctness
        direct = json.loads(engine.ttest({
            "group1": df["treatment"].tolist(),
            "group2": df["control"].tolist(),
        }))
        assert abs(out["t"] - direct["t"]) < 1e-9

    def test_correlation_with_xy_cols(self, engine, store):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.correlation({
            "result_id": src,
            "x_col": "a",
            "y_col": "b",
        }))
        assert out["r"] == pytest.approx(1.0)
        assert out["n"] == 5

    def test_regression_with_x_cols_and_y_col(self, engine, store):
        rng = np.random.default_rng(0)
        n = 60
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 2 * x1 - 1.5 * x2 + rng.normal(0, 0.1, n)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.regression({
            "result_id": src,
            "y_col": "y",
            "x_cols": ["x1", "x2"],
        }))
        assert "result_id" in out
        assert out["coefficients"]["x1"] == pytest.approx(2.0, abs=0.1)
        assert out["coefficients"]["x2"] == pytest.approx(-1.5, abs=0.1)

    def test_anova_via_value_group_long_format(self, engine, store):
        df = pd.DataFrame({
            "score": [10, 12, 11, 20, 22, 21, 30, 32, 31, 40, 42, 41],
            "group": ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3,
        })
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.anova({
            "result_id": src,
            "value_col": "score",
            "group_col": "group",
        }))
        assert "F" in out
        assert out["p"] < 0.001
        assert "result_id" in out

    def test_describe_with_data_col(self, engine, store):
        df = pd.DataFrame({"price": [10, 20, 30, 40, 50]})
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.describe({
            "result_id": src,
            "data_col": "price",
        }))
        assert out["n"] == 5
        assert out["mean"] == pytest.approx(30.0)

    def test_chi_square_with_row_col_keys(self, engine, store):
        df = pd.DataFrame({
            "sex":      ["M"] * 50 + ["F"] * 50,
            "outcome":  (["yes"] * 30 + ["no"] * 20 +
                         ["yes"] * 15 + ["no"] * 35),
        })
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.chi_square({
            "result_id": src,
            "row_col": "sex",
            "col_col": "outcome",
        }))
        assert "chi2" in out
        assert out["p"] < 0.05


# ----------------------------------------------------------------------
# Lineage
# ----------------------------------------------------------------------
class TestLineage:
    def test_parents_recorded_from_result_id(self, engine, store):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6]})
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.correlation({
            "result_id": src, "x_col": "x", "y_col": "y",
        }))
        meta = store.get_metadata(out["result_id"])
        assert src in meta["parents"]

    def test_lineage_chain_pipeline_to_stats(self, pipeline, store, engine, sample_csv):
        loaded = json.loads(pipeline.load_data({"path": sample_csv}))
        load_rid = loaded["result_id"]
        out = json.loads(engine.describe({
            "result_id": load_rid, "data_col": "score",
        }))
        lineage = store.lineage(out["result_id"])
        ids = [item["id"] for item in lineage]
        assert load_rid in ids
        assert out["result_id"] in ids

    def test_no_parents_when_inline_data(self, engine, store):
        out = json.loads(engine.ttest({
            "group1": [1, 2, 3], "group2": [4, 5, 6],
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["parents"] == []


# ----------------------------------------------------------------------
# Persistence shape
# ----------------------------------------------------------------------
class TestPersistenceShape:
    def test_kind_is_result(self, engine, store):
        out = json.loads(engine.ttest({
            "group1": [1, 2, 3], "group2": [4, 5, 6],
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["kind"] == "result"
        assert meta["tool"] == "research_ttest"

    def test_params_sanitized(self, engine, store):
        # Large arrays must NOT be stored verbatim in params
        big = list(range(500))
        out = json.loads(engine.describe({"data": big}))
        meta = store.get_metadata(out["result_id"])
        # Stored param should be a placeholder string
        assert isinstance(meta["params"]["data"], str)
        assert "list" in meta["params"]["data"]

    def test_correlation_params_recorded(self, engine, store):
        out = json.loads(engine.correlation({
            "x": [1, 2, 3], "y": [2, 4, 6], "method": "spearman",
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["params"]["method"] == "spearman"


# ----------------------------------------------------------------------
# auto_test does not double-store
# ----------------------------------------------------------------------
class TestAutoTestStorage:
    def test_auto_test_creates_single_result_id(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.auto_test({
            "groups": [[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]],
        }))
        after = store.get_stats()["total"]
        # Only the outer auto_test should add ONE record
        assert after == before + 1
        assert "result_id" in out
        # Child result should NOT have its own result_id
        assert "result_id" not in out["test_result"]

    def test_auto_test_with_value_group_cols(self, engine, store):
        df = pd.DataFrame({
            "score": [10, 12, 11, 20, 22, 21] * 3,
            "g": (["A"] * 3 + ["B"] * 3) * 3,
        })
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(engine.auto_test({
            "result_id": src,
            "value_col": "score",
            "group_col": "g",
        }))
        assert "recommended_test" in out
        assert "result_id" in out


# ----------------------------------------------------------------------
# Backwards compatibility — legacy callers must still work
# ----------------------------------------------------------------------
class TestLegacyCompat:
    def test_ttest_legacy_lists_still_work(self, engine):
        out = json.loads(engine.ttest({
            "group1": [1, 2, 3], "group2": [4, 5, 6],
        }))
        assert "t" in out
        assert "result_id" in out

    def test_anova_legacy_groups_list(self, engine):
        out = json.loads(engine.anova({
            "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        }))
        assert "F" in out
        assert "result_id" in out

    def test_chi_square_legacy_observed_table(self, engine):
        out = json.loads(engine.chi_square({
            "table": [[10, 20], [20, 10]],
        }))
        assert "chi2" in out
        assert "result_id" in out

    def test_normality_legacy_list(self, engine):
        out = json.loads(engine.normality({
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "test": "shapiro",
        }))
        assert "shapiro_wilk" in out
        assert "result_id" in out

    def test_effect_size_legacy(self, engine):
        out = json.loads(engine.effect_size({
            "group1": [1, 2, 3, 4, 5],
            "group2": [3, 4, 5, 6, 7],
            "metric": "cohens_d",
        }))
        assert "value" in out
        assert "result_id" in out
