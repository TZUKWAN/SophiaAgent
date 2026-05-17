"""Tests for CausalEngine ↔ ResultStore integration (P1.4b).

These tests exercise the new behaviours layered onto the engine in P1.4b:

- Every public method returns a ``result_id`` when a ``ResultStore`` is
  configured (did, rdd, iv, psm, its, mediation, causal_effect, sensitivity).
- Column-name selectors (``*_col``, ``covariate_cols``) resolve against an
  attached DataFrame.
- Lineage is recorded back to upstream ``res_*`` references in args.
- Legacy list/dict-based inputs still work and still get a ``result_id``.
- Errors do NOT produce ``result_id`` and do NOT persist.
- ``params`` are sanitized (no huge arrays inside the SQLite blob).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from sophia.research.causal import CausalEngine
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
    return CausalEngine(store=store)


@pytest.fixture(autouse=True)
def _reset_seed():
    GlobalSeed.reset()
    yield
    GlobalSeed.reset()


def _did_dataset(n_units=20, n_times=4):
    """Build a balanced panel suitable for DiD with positive treatment effect."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(n_units):
        is_treated = 1 if u < n_units // 2 else 0
        for t in range(n_times):
            is_post = 1 if t >= n_times // 2 else 0
            base = 10 + 0.5 * u + 0.3 * t
            effect = 3.0 * is_treated * is_post
            y = base + effect + rng.normal(0, 0.5)
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "treat": is_treated,
                    "post": is_post,
                    "y": y,
                    "x1": rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows)


def _rdd_dataset(n=400, cutoff=0.0):
    rng = np.random.default_rng(11)
    running = rng.uniform(-2, 2, n)
    y = 1.0 + 0.5 * running + (2.0 if True else 0.0) * (running >= cutoff).astype(
        float
    ) + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, "x": running})


def _iv_dataset(n=300):
    rng = np.random.default_rng(13)
    z = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    endog = 0.8 * z + 0.5 * u + rng.normal(0, 0.3, n)
    y = 1.5 * endog + 0.7 * u + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "endog": endog, "z": z})


def _psm_dataset(n=200):
    rng = np.random.default_rng(17)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 0.5 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-logit))
    treat = (rng.uniform(0, 1, n) < p).astype(int)
    y = 2.0 * treat + 0.4 * x1 - 0.2 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"treat": treat, "y": y, "x1": x1, "x2": x2})


def _its_dataset(n=40):
    rng = np.random.default_rng(19)
    t = np.arange(n)
    intervention = (t >= n // 2).astype(int)
    y = 5 + 0.1 * t + 3.0 * intervention + rng.normal(0, 0.4, n)
    return pd.DataFrame({"time": t, "y": y, "intervention": intervention})


def _mediation_dataset(n=100):
    rng = np.random.default_rng(23)
    x = rng.normal(0, 1, n)
    m = 0.7 * x + rng.normal(0, 0.5, n)
    y = 0.3 * x + 0.5 * m + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "m": m, "y": y})


# ----------------------------------------------------------------------
# result_id round-trip
# ----------------------------------------------------------------------
class TestResultIdRoundTrip:
    def test_did_returns_result_id(self, engine):
        df = _did_dataset()
        out = json.loads(
            engine.did(
                {
                    "y": df["y"].tolist(),
                    "treat": df["treat"].tolist(),
                    "post": df["post"].tolist(),
                }
            )
        )
        assert "result_id" in out
        assert out["result_id"].startswith("res_")

    def test_rdd_returns_result_id(self, engine):
        df = _rdd_dataset()
        out = json.loads(
            engine.rdd(
                {
                    "y": df["y"].tolist(),
                    "running": df["x"].tolist(),
                    "cutoff": 0.0,
                }
            )
        )
        assert "result_id" in out

    def test_iv_returns_result_id(self, engine):
        df = _iv_dataset()
        out = json.loads(
            engine.iv(
                {
                    "y": df["y"].tolist(),
                    "endogenous": df["endog"].tolist(),
                    "instrument": df["z"].tolist(),
                }
            )
        )
        assert "result_id" in out

    def test_psm_returns_result_id(self, engine):
        df = _psm_dataset()
        out = json.loads(
            engine.psm(
                {
                    "treat": df["treat"].tolist(),
                    "outcomes": df["y"].tolist(),
                    "covariates": {
                        "x1": df["x1"].tolist(),
                        "x2": df["x2"].tolist(),
                    },
                    "method": "nearest",
                }
            )
        )
        assert "result_id" in out

    def test_its_returns_result_id(self, engine):
        df = _its_dataset()
        out = json.loads(
            engine.its(
                {
                    "y": df["y"].tolist(),
                    "time": df["time"].tolist(),
                    "intervention": 20.0,  # ITS expects a scalar time-point
                }
            )
        )
        assert "result_id" in out

    def test_mediation_returns_result_id(self, engine):
        df = _mediation_dataset()
        out = json.loads(
            engine.mediation(
                {
                    "y": df["y"].tolist(),
                    "x": df["x"].tolist(),
                    "mediator": df["m"].tolist(),
                    "bootstrap": 200,
                    "seed": 1,
                }
            )
        )
        assert "result_id" in out

    def test_causal_effect_returns_result_id(self, engine):
        df = _psm_dataset()
        out = json.loads(
            engine.causal_effect(
                {
                    "y": df["y"].tolist(),
                    "treat": df["treat"].tolist(),
                    "covariates": {
                        "x1": df["x1"].tolist(),
                        "x2": df["x2"].tolist(),
                    },
                    "method": "ols",
                }
            )
        )
        assert "result_id" in out

    def test_sensitivity_returns_result_id(self, engine):
        out = json.loads(
            engine.sensitivity(
                {
                    "method": "oster",
                    "beta_control": 0.4,
                    "r_control": 0.3,
                    "beta_uncontrolled": 0.6,
                    "r_uncontrolled": 0.1,
                }
            )
        )
        assert "result_id" in out

    def test_no_store_no_result_id(self):
        plain = CausalEngine(store=None)
        df = _did_dataset()
        out = json.loads(
            plain.did(
                {
                    "y": df["y"].tolist(),
                    "treat": df["treat"].tolist(),
                    "post": df["post"].tolist(),
                }
            )
        )
        assert "result_id" not in out
        assert "did_estimate" in out

    def test_error_does_not_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.did({"y": [], "treat": [], "post": []}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_sensitivity_missing_args_error_no_store(self, engine, store):
        before = store.get_stats()["total"]
        out = json.loads(engine.sensitivity({"method": "rosenbaum"}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before


# ----------------------------------------------------------------------
# Column-name resolution
# ----------------------------------------------------------------------
class TestColumnResolution:
    def test_did_with_column_selectors_via_result_id(self, engine, store):
        df = _did_dataset()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(
            engine.did(
                {
                    "result_id": src,
                    "y_col": "y",
                    "treat_col": "treat",
                    "post_col": "post",
                }
            )
        )
        assert "did_estimate" in out
        # Effect was 3.0 in synthetic generator
        assert out["did_estimate"] == pytest.approx(3.0, abs=0.4)
        assert "result_id" in out

    def test_did_covariate_cols(self, engine, store):
        df = _did_dataset()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(
            engine.did(
                {
                    "result_id": src,
                    "y_col": "y",
                    "treat_col": "treat",
                    "post_col": "post",
                    "covariate_cols": ["x1"],
                }
            )
        )
        assert "did_estimate" in out
        assert "coefficients" in out
        assert "x1" in out["coefficients"]

    def test_rdd_with_y_col_running_col(self, engine, store):
        df = _rdd_dataset()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(
            engine.rdd(
                {
                    "result_id": src,
                    "y_col": "y",
                    "running_col": "x",
                    "cutoff": 0.0,
                }
            )
        )
        assert "late_estimate" in out or "estimate" in out or "late" in out

    def test_iv_with_column_selectors(self, engine, store):
        df = _iv_dataset()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(
            engine.iv(
                {
                    "result_id": src,
                    "y_col": "y",
                    "endogenous_col": "endog",
                    "instrument_col": "z",
                }
            )
        )
        assert "iv_coefficient" in out or "iv_estimate" in out or "coefficient" in out or "beta" in out

    def test_psm_with_outcome_col_treat_col_covariates(self, engine, store):
        df = _psm_dataset()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(
            engine.psm(
                {
                    "result_id": src,
                    "treat_col": "treat",
                    "outcome_col": "y",
                    "covariate_cols": ["x1", "x2"],
                    "method": "nearest",
                }
            )
        )
        assert "att" in out or "ate" in out or "matched" in out

    def test_its_with_column_selectors(self, engine, store):
        df = _its_dataset()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(
            engine.its(
                {
                    "result_id": src,
                    "y_col": "y",
                    "time_col": "time",
                    "intervention": 20.0,
                }
            )
        )
        assert "level_change" in out or "level_shift" in out or "intercept_change" in out or "did_estimate" in out or "method" in out

    def test_mediation_with_column_selectors(self, engine, store):
        df = _mediation_dataset()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(
            engine.mediation(
                {
                    "result_id": src,
                    "y_col": "y",
                    "x_col": "x",
                    "mediator_col": "m",
                    "bootstrap": 200,
                    "seed": 1,
                }
            )
        )
        assert "indirect_effect" in out
        # We set ab = 0.7 * 0.5 = 0.35 in the generator
        assert out["indirect_effect"] == pytest.approx(0.35, abs=0.15)

    def test_causal_effect_with_column_selectors(self, engine, store):
        df = _psm_dataset()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(
            engine.causal_effect(
                {
                    "result_id": src,
                    "y_col": "y",
                    "treat_col": "treat",
                    "covariate_cols": ["x1", "x2"],
                    "method": "ols",
                }
            )
        )
        assert "ate" in out
        # True ATE = 2.0
        assert out["ate"] == pytest.approx(2.0, abs=0.5)


# ----------------------------------------------------------------------
# Lineage
# ----------------------------------------------------------------------
class TestLineage:
    def test_parents_recorded_from_result_id(self, engine, store):
        df = _did_dataset()
        src = store.store(df, kind="dataframe", tool="test_seed")
        out = json.loads(
            engine.did(
                {
                    "result_id": src,
                    "y_col": "y",
                    "treat_col": "treat",
                    "post_col": "post",
                }
            )
        )
        meta = store.get_metadata(out["result_id"])
        assert src in meta["parents"]

    def test_no_parents_for_inline_input(self, engine, store):
        df = _mediation_dataset()
        out = json.loads(
            engine.mediation(
                {
                    "y": df["y"].tolist(),
                    "x": df["x"].tolist(),
                    "mediator": df["m"].tolist(),
                    "bootstrap": 100,
                    "seed": 1,
                }
            )
        )
        meta = store.get_metadata(out["result_id"])
        assert meta["parents"] == []


# ----------------------------------------------------------------------
# Persistence shape
# ----------------------------------------------------------------------
class TestPersistenceShape:
    def test_kind_is_result_for_did(self, engine, store):
        df = _did_dataset()
        out = json.loads(
            engine.did(
                {
                    "y": df["y"].tolist(),
                    "treat": df["treat"].tolist(),
                    "post": df["post"].tolist(),
                }
            )
        )
        meta = store.get_metadata(out["result_id"])
        assert meta["kind"] == "result"
        assert meta["tool"] == "research_did"

    def test_params_sanitized_for_large_arrays(self, engine, store):
        df = _did_dataset(n_units=50, n_times=10)  # 500 rows
        out = json.loads(
            engine.did(
                {
                    "y": df["y"].tolist(),
                    "treat": df["treat"].tolist(),
                    "post": df["post"].tolist(),
                }
            )
        )
        meta = store.get_metadata(out["result_id"])
        # y/treat/post lists are 500 entries — should be summarized
        assert isinstance(meta["params"]["y"], str)
        assert "list" in meta["params"]["y"]

    def test_params_sanitized_for_covariates_dict(self, engine, store):
        df = _psm_dataset()
        out = json.loads(
            engine.psm(
                {
                    "treat": df["treat"].tolist(),
                    "outcomes": df["y"].tolist(),
                    "covariates": {
                        "x1": df["x1"].tolist(),
                        "x2": df["x2"].tolist(),
                    },
                    "method": "nearest",
                }
            )
        )
        meta = store.get_metadata(out["result_id"])
        # covariates has 2*200 = 400 elements total; should be summarized
        assert isinstance(meta["params"]["covariates"], str)
        assert "dict" in meta["params"]["covariates"]

    def test_method_param_kept_in_metadata(self, engine, store):
        df = _psm_dataset()
        out = json.loads(
            engine.causal_effect(
                {
                    "y": df["y"].tolist(),
                    "treat": df["treat"].tolist(),
                    "covariates": {
                        "x1": df["x1"].tolist(),
                        "x2": df["x2"].tolist(),
                    },
                    "method": "ipw",
                }
            )
        )
        meta = store.get_metadata(out["result_id"])
        assert meta["params"]["method"] == "ipw"


# ----------------------------------------------------------------------
# Legacy compatibility
# ----------------------------------------------------------------------
class TestLegacyCompat:
    def test_did_legacy_lists(self, engine):
        df = _did_dataset()
        out = json.loads(
            engine.did(
                {
                    "y": df["y"].tolist(),
                    "treat": df["treat"].tolist(),
                    "post": df["post"].tolist(),
                }
            )
        )
        assert "did_estimate" in out
        assert "result_id" in out

    def test_iv_legacy_lists(self, engine):
        df = _iv_dataset()
        out = json.loads(
            engine.iv(
                {
                    "y": df["y"].tolist(),
                    "endogenous": df["endog"].tolist(),
                    "instrument": df["z"].tolist(),
                }
            )
        )
        # IV must produce some kind of estimate field
        assert any(
            k in out
            for k in ("iv_coefficient", "iv_estimate", "beta", "coefficient", "ate")
        )
        assert "result_id" in out

    def test_psm_legacy_dict_covariates(self, engine):
        df = _psm_dataset()
        out = json.loads(
            engine.psm(
                {
                    "treat": df["treat"].tolist(),
                    "outcomes": df["y"].tolist(),
                    "covariates": {
                        "x1": df["x1"].tolist(),
                        "x2": df["x2"].tolist(),
                    },
                    "method": "weight",
                }
            )
        )
        assert "result_id" in out

    def test_sensitivity_rosenbaum_legacy(self, engine):
        out = json.loads(
            engine.sensitivity(
                {
                    "method": "rosenbaum",
                    "estimate": 0.5,
                    "se": 0.15,
                }
            )
        )
        assert "result_id" in out
        assert "bounds" in out

    def test_sensitivity_eittheim_legacy(self, engine):
        out = json.loads(
            engine.sensitivity(
                {
                    "method": "eittheim",
                    "estimate": 0.5,
                    "se": 0.15,
                }
            )
        )
        assert "result_id" in out
        assert "bias_to_insignificance" in out
