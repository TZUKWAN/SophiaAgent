"""Tests for the full DiD implementation (P2): TWFE, parallel trends,
event study, placebo, SE comparison, Goodman-Bacon, APA output.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from sophia.research.causal import CausalEngine


@pytest.fixture
def engine(tmp_path):
    return CausalEngine()


@pytest.fixture
def panel_did_data():
    """Panel data with known DID effect = 5.0.

    100 units, 10 periods. Units 50-99 treated from period 5 onward.
    """
    np.random.seed(42)
    n_units = 100
    n_periods = 10
    n = n_units * n_periods
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    treat = (units >= 50).astype(float)
    post = (times >= 5).astype(float)
    treat_post = treat * post
    unit_fe = np.random.normal(0, 1, n_units)
    y = (10.0 + unit_fe[units] + 0.5 * times + 5.0 * treat_post
         + np.random.normal(0, 1, n))
    return {
        "y": y.tolist(),
        "treat": treat.tolist(),
        "post": post.tolist(),
        "unit": units.tolist(),
        "time": times.tolist(),
    }


@pytest.fixture
def staggered_did_data():
    """Staggered adoption: units treated at different times."""
    np.random.seed(42)
    n_units = 80
    n_periods = 12
    n = n_units * n_periods
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    # Staggered treatment: first 20 units at t=3, next 20 at t=5, next 20 at t=7, last 20 never
    treat_time = np.repeat([3, 5, 7, np.inf], 20)
    treat = (times >= treat_time[units]).astype(float)
    post = treat.copy()
    treat_post = treat * post
    unit_fe = np.random.normal(0, 1, n_units)
    y = (10.0 + unit_fe[units] + 0.3 * times + 4.0 * treat_post
         + np.random.normal(0, 1, n))
    return {
        "y": y.tolist(),
        "treat": treat.tolist(),
        "post": post.tolist(),
        "unit": units.tolist(),
        "time": times.tolist(),
    }


# =====================================================================
# TWFE + clustered SE
# =====================================================================
class TestDiDTWFE:
    def test_twfe_estimate_close_to_true(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        assert "did_estimate" in result
        assert abs(result["did_estimate"] - 5.0) < 0.5

    def test_twfe_clustered_se(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        assert result.get("method", "").startswith("Difference-in-Differences (TWFE")
        assert "se" in result
        assert result["se"] > 0

    def test_twfe_panel_info(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        assert result.get("n_units") == 100
        assert result.get("n_periods") == 10

    def test_twfe_coefficients_table(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        coeffs = result.get("coefficients", {})
        assert "interaction" in coeffs
        assert "estimate" in coeffs["interaction"]
        assert "se" in coeffs["interaction"]


# =====================================================================
# Parallel trends test
# =====================================================================
class TestDiDParallelTrends:
    def test_parallel_trends_present(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        assert "parallel_trends_test" in result
        pt = result["parallel_trends_test"]
        assert "F" in pt
        assert "p_value" in pt
        assert "passes" in pt

    def test_parallel_trends_passes_for_valid_data(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        pt = result["parallel_trends_test"]
        assert pt["passes"] is True  # valid synthetic data should pass


# =====================================================================
# Event study
# =====================================================================
class TestDiDEventStudy:
    def test_event_study_when_requested(self, engine, panel_did_data):
        args = {**panel_did_data, "event_study": True}
        result = json.loads(engine.did(args))
        assert "event_study" in result
        es = result["event_study"]
        assert "base_period" in es
        assert "coefficients" in es

    def test_event_study_base_period_is_negative_one(self, engine, panel_did_data):
        args = {**panel_did_data, "event_study": True}
        result = json.loads(engine.did(args))
        assert result["event_study"]["base_period"] == -1.0

    def test_event_study_not_present_by_default(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        assert "event_study" not in result


# =====================================================================
# Placebo tests
# =====================================================================
class TestDiDPlacebo:
    def test_placebo_when_requested(self, engine, panel_did_data):
        args = {**panel_did_data, "placebo": True, "n_placebo": 50}
        result = json.loads(engine.did(args))
        assert "placebo" in result
        placebo = result["placebo"]
        assert "in_time" in placebo
        assert "in_space" in placebo

    def test_in_space_p_value_present(self, engine, panel_did_data):
        args = {**panel_did_data, "placebo": True, "n_placebo": 50}
        result = json.loads(engine.did(args))
        in_space = result["placebo"]["in_space"]
        assert "p_value" in in_space
        assert "n_iters" in in_space

    def test_placebo_not_present_by_default(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        assert "placebo" not in result


# =====================================================================
# SE comparison
# =====================================================================
class TestDiDSEComparison:
    def test_se_comparison_present(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        assert "se_comparison" in result
        se = result["se_comparison"]
        # At least clustered should be present
        assert "clustered" in se

    def test_se_comparison_has_multiple_estimators(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        se = result["se_comparison"]
        # Should contain at least clustered and one other
        assert len(se) >= 2


# =====================================================================
# Goodman-Bacon decomposition
# =====================================================================
class TestDiDBacon:
    def test_bacon_for_staggered(self, engine, staggered_did_data):
        args = {**staggered_did_data, "bacon": True}
        result = json.loads(engine.did(args))
        assert "bacon_decomposition" in result
        bacon = result["bacon_decomposition"]
        assert "comparisons" in bacon
        assert len(bacon["comparisons"]) > 0

    def test_bacon_not_present_for_classic(self, engine, panel_did_data):
        args = {**panel_did_data, "bacon": True}
        result = json.loads(engine.did(args))
        # Classic DID has uniform timing, so bacon may still run but comparisons
        # should be minimal or absent
        assert "bacon_decomposition" not in result or result.get("bacon_decomposition") is None

    def test_bacon_not_present_by_default(self, engine, staggered_did_data):
        result = json.loads(engine.did(staggered_did_data))
        assert "bacon_decomposition" not in result


# =====================================================================
# APA report
# =====================================================================
class TestDiDAPA:
    def test_apa_report_present(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        assert "apa_report" in result
        assert isinstance(result["apa_report"], str)
        assert len(result["apa_report"]) > 20

    def test_apa_contains_estimate(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        apa = result["apa_report"]
        assert "β =" in apa or "beta =" in apa

    def test_apa_contains_parallel_trends(self, engine, panel_did_data):
        result = json.loads(engine.did(panel_did_data))
        apa = result["apa_report"]
        assert "parallel-trends" in apa.lower() or "parallel trends" in apa.lower()


# =====================================================================
# Backward compatibility (classic 2x2 without unit/time)
# =====================================================================
class TestDiDBackwardCompat:
    def test_classic_ols_without_unit_time(self, engine):
        np.random.seed(42)
        n = 200
        treat = np.random.binomial(1, 0.5, n)
        post = np.random.binomial(1, 0.5, n)
        y = 10.0 + 3.0 * treat * post + np.random.normal(0, 1, n)
        args = {"y": y.tolist(), "treat": treat.tolist(), "post": post.tolist()}
        result = json.loads(engine.did(args))
        assert "did_estimate" in result
        assert abs(result["did_estimate"] - 3.0) < 1.0

    def test_classic_method_label(self, engine):
        np.random.seed(42)
        n = 200
        treat = np.random.binomial(1, 0.5, n)
        post = np.random.binomial(1, 0.5, n)
        y = 10.0 + 3.0 * treat * post + np.random.normal(0, 1, n)
        args = {"y": y.tolist(), "treat": treat.tolist(), "post": post.tolist()}
        result = json.loads(engine.did(args))
        assert "OLS" in result.get("method", "")
