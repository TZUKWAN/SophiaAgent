"""Tests for Synthetic Control Method (P3)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from sophia.research.causal import CausalEngine


@pytest.fixture
def engine(tmp_path):
    return CausalEngine()


@pytest.fixture
def scm_data():
    """Panel data for SCM: unit 0 treated at t=5, donors 1-10."""
    np.random.seed(42)
    n_units = 11
    n_periods = 10
    n = n_units * n_periods
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)
    # Unit 0 is treated from period 5 onward
    treat = (units == 0).astype(float) * (times >= 5).astype(float)
    # Base outcome + unit FE + time trend + treatment effect
    unit_fe = np.random.normal(0, 1, n_units)
    y = (10.0 + unit_fe[units] + 0.3 * times + 4.0 * treat
         + np.random.normal(0, 0.5, n))
    return {
        "y": y.tolist(),
        "unit": units.tolist(),
        "time": times.tolist(),
        "treated_unit": 0,
        "treatment_time": 5,
    }


class TestSCMBasic:
    def test_scm_returns_required_fields(self, engine, scm_data):
        result = json.loads(engine.synthetic_control(scm_data))
        assert "weights" in result
        assert "average_treatment_effect" in result
        assert "rmspe_pre" in result
        assert "rmspe_post" in result

    def test_scm_weights_sum_to_one(self, engine, scm_data):
        result = json.loads(engine.synthetic_control(scm_data))
        weights = result["weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_scm_weights_non_negative(self, engine, scm_data):
        result = json.loads(engine.synthetic_control(scm_data))
        weights = result["weights"]
        assert all(w >= 0 for w in weights.values())

    def test_scm_estimate_reasonable(self, engine, scm_data):
        result = json.loads(engine.synthetic_control(scm_data))
        ate = result["average_treatment_effect"]
        # True effect is 4.0; allow some tolerance
        assert abs(ate - 4.0) < 2.0

    def test_scm_rmspe_ratio_present(self, engine, scm_data):
        result = json.loads(engine.synthetic_control(scm_data))
        assert result["rmspe_ratio"] is not None
        assert result["rmspe_ratio"] > 0

    def test_scm_predictor_balance(self, engine, scm_data):
        result = json.loads(engine.synthetic_control(scm_data))
        assert "predictor_balance" in result
        pb = result["predictor_balance"]
        assert len(pb) > 0
        for key, vals in pb.items():
            assert "treated" in vals
            assert "synthetic" in vals
            assert "gap" in vals

    def test_scm_apa_report(self, engine, scm_data):
        result = json.loads(engine.synthetic_control(scm_data))
        assert "apa_report" in result
        assert isinstance(result["apa_report"], str)
        assert len(result["apa_report"]) > 10


class TestSCMPlacebo:
    def test_scm_placebo_when_requested(self, engine, scm_data):
        args = {**scm_data, "placebo": True}
        result = json.loads(engine.synthetic_control(args))
        assert "placebo" in result
        placebo = result["placebo"]
        assert "donor_placebos" in placebo
        assert "permutation_p_value" in placebo

    def test_scm_placebo_not_present_by_default(self, engine, scm_data):
        result = json.loads(engine.synthetic_control(scm_data))
        assert "placebo" not in result


class TestSCMErrors:
    def test_scm_missing_treated_unit(self, engine, scm_data):
        args = {k: v for k, v in scm_data.items() if k != "treated_unit"}
        result = json.loads(engine.synthetic_control(args))
        assert "error" in result

    def test_scm_missing_treatment_time(self, engine, scm_data):
        args = {k: v for k, v in scm_data.items() if k != "treatment_time"}
        result = json.loads(engine.synthetic_control(args))
        assert "error" in result

    def test_scm_too_few_donors(self, engine):
        args = {
            "y": [1, 2, 3, 4],
            "unit": [0, 0, 1, 1],
            "time": [0, 1, 0, 1],
            "treated_unit": 0,
            "treatment_time": 1,
        }
        result = json.loads(engine.synthetic_control(args))
        assert "error" in result
