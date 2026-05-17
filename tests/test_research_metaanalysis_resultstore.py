"""Tests for MetaAnalysisEngine ↔ ResultStore integration (P1.4e).

These tests exercise the new behaviours layered onto the engine in P1.4e:

- Every public method returns a ``result_id`` when a ``ResultStore`` is
  configured (fixed_effect, random_effect, heterogeneity, bias_test, subgroup).
- Errors do NOT produce ``result_id`` and do NOT persist.
- ``params`` are sanitized (no huge arrays inside the SQLite blob).
- Legacy inputs still work and still get a ``result_id``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sophia.research.meta_analysis import MetaAnalysisEngine
from sophia.research.result_store import ResultStore
from sophia.research.seed import GlobalSeed


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
    return MetaAnalysisEngine(store=store)


@pytest.fixture(autouse=True)
def _reset_seed():
    GlobalSeed.reset()
    yield
    GlobalSeed.reset()


def _example_effects(k=5):
    rng = np.random.default_rng(11)
    effects = [0.2 + rng.normal(0, 0.1) for _ in range(k)]
    variances = [0.04 + rng.exponential(0.02) for _ in range(k)]
    study_names = [f"Study {i + 1}" for i in range(k)]
    return effects, variances, study_names


def _example_subgroups(k=6):
    effects = [0.3, 0.4, 0.25, 0.15, 0.35, 0.2]
    variances = [0.05, 0.04, 0.06, 0.04, 0.05, 0.07]
    subgroups = ["A", "A", "A", "B", "B", "B"]
    return effects, variances, subgroups


# ----------------------------------------------------------------------
# result_id round-trip
# ----------------------------------------------------------------------
class TestResultIdRoundTrip:
    def test_fixed_effect_returns_result_id(self, engine):
        eff, var, names = _example_effects()
        out = json.loads(engine.fixed_effect({
            "effects": eff, "variances": var, "study_names": names,
        }))
        assert "result_id" in out
        assert out["result_id"].startswith("res_")
        assert "pooled_effect" in out

    def test_random_effect_returns_result_id(self, engine):
        eff, var, names = _example_effects()
        out = json.loads(engine.random_effect({
            "effects": eff, "variances": var, "study_names": names,
        }))
        assert "result_id" in out
        assert "tau2" in out

    def test_heterogeneity_returns_result_id(self, engine):
        eff, var, _ = _example_effects()
        out = json.loads(engine.heterogeneity({
            "effects": eff, "variances": var,
        }))
        assert "result_id" in out
        assert "I2" in out

    def test_bias_test_egger_returns_result_id(self, engine):
        eff, var, names = _example_effects(k=8)
        out = json.loads(engine.bias_test({
            "effects": eff, "variances": var, "test": "egger",
        }))
        assert "result_id" in out
        assert "test" in out

    def test_bias_test_fail_safe_returns_result_id(self, engine):
        eff, var, _ = _example_effects(k=8)
        out = json.loads(engine.bias_test({
            "effects": eff, "variances": var, "test": "fail_safe",
        }))
        assert "result_id" in out
        assert "N_fail_safe" in out

    def test_subgroup_returns_result_id(self, engine):
        eff, var, sg = _example_subgroups()
        out = json.loads(engine.subgroup({
            "effects": eff, "variances": var, "subgroups": sg,
        }))
        assert "result_id" in out
        assert "subgroups" in out

    def test_no_store_no_result_id(self):
        plain = MetaAnalysisEngine(store=None)
        eff, var, _ = _example_effects()
        out = json.loads(plain.fixed_effect({
            "effects": eff, "variances": var,
        }))
        assert "result_id" not in out
        assert "pooled_effect" in out

    def test_fixed_effect_mismatched_length_error_no_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.fixed_effect({
            "effects": [0.1, 0.2], "variances": [0.05],
        }))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_bias_test_unknown_type_error_no_store(self, engine, store):
        before = store.get_stats()["total"]
        eff, var, _ = _example_effects(k=4)
        out = json.loads(engine.bias_test({
            "effects": eff, "variances": var, "test": "unknown",
        }))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before


# ----------------------------------------------------------------------
# Persistence shape
# ----------------------------------------------------------------------
class TestPersistenceShape:
    def test_kind_is_result(self, engine, store):
        eff, var, _ = _example_effects()
        out = json.loads(engine.fixed_effect({
            "effects": eff, "variances": var,
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["kind"] == "result"
        assert meta["tool"] == "research_fixed_effect"

    def test_tool_name_per_method(self, engine, store):
        eff, var, names = _example_effects()
        rid1 = json.loads(engine.fixed_effect({
            "effects": eff, "variances": var,
        }))["result_id"]
        rid2 = json.loads(engine.random_effect({
            "effects": eff, "variances": var,
        }))["result_id"]
        rid3 = json.loads(engine.heterogeneity({
            "effects": eff, "variances": var,
        }))["result_id"]
        rid4 = json.loads(engine.bias_test({
            "effects": eff, "variances": var, "test": "egger",
        }))["result_id"]
        rid5 = json.loads(engine.subgroup({
            "effects": [0.3, 0.4, 0.25, 0.15, 0.35, 0.2],
            "variances": [0.05, 0.04, 0.06, 0.04, 0.05, 0.07],
            "subgroups": ["A", "A", "A", "B", "B", "B"],
        }))["result_id"]
        assert store.get_metadata(rid1)["tool"] == "research_fixed_effect"
        assert store.get_metadata(rid2)["tool"] == "research_random_effect"
        assert store.get_metadata(rid3)["tool"] == "research_heterogeneity"
        assert store.get_metadata(rid4)["tool"] == "research_bias_test"
        assert store.get_metadata(rid5)["tool"] == "research_subgroup"

    def test_params_sanitized_for_large_arrays(self, engine, store):
        eff = list(range(100))
        var = [1.0] * 100
        out = json.loads(engine.fixed_effect({
            "effects": eff, "variances": var,
        }))
        meta = store.get_metadata(out["result_id"])
        assert isinstance(meta["params"]["effects"], str)
        assert "list" in meta["params"]["effects"]
        assert isinstance(meta["params"]["variances"], str)

    def test_scalar_params_kept_in_metadata(self, engine, store):
        eff, var, names = _example_effects()
        out = json.loads(engine.fixed_effect({
            "effects": eff,
            "variances": var,
            "effect_label": "Odds Ratio",
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["params"]["effect_label"] == "Odds Ratio"


# ----------------------------------------------------------------------
# Lineage
# ----------------------------------------------------------------------
class TestLineage:
    def test_no_parents_for_inline_input(self, engine, store):
        eff, var, _ = _example_effects()
        out = json.loads(engine.fixed_effect({
            "effects": eff, "variances": var,
        }))
        meta = store.get_metadata(out["result_id"])
        assert meta["parents"] == []


# ----------------------------------------------------------------------
# Legacy compatibility
# ----------------------------------------------------------------------
class TestLegacyCompat:
    def test_fixed_effect_legacy(self, engine):
        out = json.loads(engine.fixed_effect({
            "effects": [0.1, 0.2, 0.15, 0.3],
            "variances": [0.05, 0.04, 0.06, 0.05],
            "study_names": ["A", "B", "C", "D"],
        }))
        assert "pooled_effect" in out
        assert "result_id" in out

    def test_random_effect_legacy(self, engine):
        out = json.loads(engine.random_effect({
            "effects": [0.1, 0.2, 0.15, 0.3],
            "variances": [0.05, 0.04, 0.06, 0.05],
        }))
        assert "tau2" in out
        assert "result_id" in out

    def test_heterogeneity_legacy(self, engine):
        out = json.loads(engine.heterogeneity({
            "effects": [0.1, 0.2, 0.15, 0.3],
            "variances": [0.05, 0.04, 0.06, 0.05],
        }))
        assert "I2" in out
        assert "result_id" in out

    def test_bias_test_egger_legacy(self, engine):
        out = json.loads(engine.bias_test({
            "effects": [0.1, 0.2, 0.15, 0.3, 0.25, 0.18],
            "variances": [0.05, 0.04, 0.06, 0.05, 0.04, 0.05],
            "test": "egger",
        }))
        assert "intercept" in out
        assert "result_id" in out

    def test_subgroup_legacy(self, engine):
        out = json.loads(engine.subgroup({
            "effects": [0.3, 0.4, 0.25, 0.15, 0.35, 0.2],
            "variances": [0.05, 0.04, 0.06, 0.04, 0.05, 0.07],
            "subgroups": ["A", "A", "A", "B", "B", "B"],
        }))
        assert "subgroups" in out
        assert "result_id" in out
