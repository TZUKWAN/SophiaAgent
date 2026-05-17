"""Tests for MetaAnalysisEngine -- comprehensive pytest suite.

Uses real data (classic meta-analysis examples).  Covers every public method
and common edge cases.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sophia.research.meta_analysis import MetaAnalysisEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> MetaAnalysisEngine:
    return MetaAnalysisEngine()


@pytest.fixture
def bcg_vaccine_data():
    """Classic BCG vaccine meta-analysis data (13 studies).

    Effects are log risk ratios, variances are their variances.
    """
    effects = [
        -0.889, -1.585, -1.348, -1.444, -0.218,
        -0.786, -1.621, 0.012, -0.470, -1.401,
        -0.935, -0.415, -0.179,
    ]
    variances = [
        0.350, 0.198, 0.416, 0.040, 0.048,
        0.073, 0.074, 0.539, 0.733, 0.074,
        0.084, 0.084, 0.089,
    ]
    study_names = [
        "Aronson", "Ferguson & Simes", "Rosenthal et al",
        "Hart & Sutherland", "Frimodt-Moller et al",
        "Stein & Aronson", "Vandiviere et al",
        "TPT Madras", "Coetzee & Berjak",
        "Rosenthal et al", "Comstock et al",
        "Comstock & Webster", "Comstock et al",
    ]
    return effects, variances, study_names


@pytest.fixture
def homogeneous_effects():
    """5 studies with similar effects (low heterogeneity)."""
    effects = [0.50, 0.55, 0.48, 0.52, 0.49]
    variances = [0.02, 0.03, 0.025, 0.02, 0.03]
    return effects, variances


@pytest.fixture
def heterogeneous_effects():
    """5 studies with widely varying effects (high heterogeneity)."""
    effects = [-0.5, 0.2, 1.5, -1.0, 0.8]
    variances = [0.05, 0.04, 0.06, 0.03, 0.05]
    return effects, variances


# ===========================================================================
# fixed_effect
# ===========================================================================

class TestFixedEffect:

    def test_basic_fixed_effect(self, engine, bcg_vaccine_data):
        effects, variances, names = bcg_vaccine_data
        result = json.loads(engine.fixed_effect({
            "effects": effects, "variances": variances,
            "study_names": names,
        }))
        assert "pooled_effect" in result
        assert "se" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert "z" in result
        assert "p" in result
        assert "Q" in result
        assert result["k"] == 13
        assert result["model"] == "fixed-effect"

    def test_pooled_effect_direction(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        result = json.loads(engine.fixed_effect({
            "effects": effects, "variances": variances,
        }))
        # BCG vaccine effects are mostly negative (protective)
        assert result["pooled_effect"] < 0

    def test_ci_contains_pooled(self, engine, homogeneous_effects):
        effects, variances = homogeneous_effects
        result = json.loads(engine.fixed_effect({
            "effects": effects, "variances": variances,
        }))
        assert result["ci_low"] < result["pooled_effect"] < result["ci_high"]

    def test_weights_sum_correctly(self, engine, homogeneous_effects):
        effects, variances = homogeneous_effects
        result = json.loads(engine.fixed_effect({
            "effects": effects, "variances": variances,
        }))
        studies = result["studies"]
        total_weight_pct = sum(s["weight_pct"] for s in studies)
        assert abs(total_weight_pct - 100.0) < 0.1

    def test_empty_effects_error(self, engine):
        result = json.loads(engine.fixed_effect({
            "effects": [], "variances": [],
        }))
        assert "error" in result

    def test_mismatched_lengths_error(self, engine):
        result = json.loads(engine.fixed_effect({
            "effects": [0.5, 0.6], "variances": [0.1],
        }))
        assert "error" in result

    def test_negative_variance_error(self, engine):
        result = json.loads(engine.fixed_effect({
            "effects": [0.5, 0.6], "variances": [0.1, -0.2],
        }))
        assert "error" in result


# ===========================================================================
# random_effect
# ===========================================================================

class TestRandomEffect:

    def test_basic_random_effect(self, engine, bcg_vaccine_data):
        effects, variances, names = bcg_vaccine_data
        result = json.loads(engine.random_effect({
            "effects": effects, "variances": variances,
            "study_names": names,
        }))
        assert "pooled_effect" in result
        assert "tau2" in result
        assert "Q" in result
        assert "I2" in result
        assert "H" in result
        assert result["model"] == "random-effects (DerSimonian-Laird)"

    def test_random_effect_ci_wider_than_fixed(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        fe_result = json.loads(engine.fixed_effect({
            "effects": effects, "variances": variances,
        }))
        re_result = json.loads(engine.random_effect({
            "effects": effects, "variances": variances,
        }))
        fe_ci_width = fe_result["ci_high"] - fe_result["ci_low"]
        re_ci_width = re_result["ci_high"] - re_result["ci_low"]
        # Random-effects CI should be wider or equal
        assert re_ci_width >= fe_ci_width - 0.01

    def test_tau2_non_negative(self, engine, heterogeneous_effects):
        effects, variances = heterogeneous_effects
        result = json.loads(engine.random_effect({
            "effects": effects, "variances": variances,
        }))
        assert result["tau2"] >= 0

    def test_I2_range(self, engine, heterogeneous_effects):
        effects, variances = heterogeneous_effects
        result = json.loads(engine.random_effect({
            "effects": effects, "variances": variances,
        }))
        assert 0 <= result["I2"] <= 100

    def test_H_statistic(self, engine, heterogeneous_effects):
        effects, variances = heterogeneous_effects
        result = json.loads(engine.random_effect({
            "effects": effects, "variances": variances,
        }))
        assert result["H"] >= 1.0

    def test_study_names_mismatch_error(self, engine, homogeneous_effects):
        effects, variances = homogeneous_effects
        result = json.loads(engine.random_effect({
            "effects": effects, "variances": variances,
            "study_names": ["a", "b"],
        }))
        assert "error" in result


# ===========================================================================
# heterogeneity
# ===========================================================================

class TestHeterogeneity:

    def test_basic_heterogeneity(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        result = json.loads(engine.heterogeneity({
            "effects": effects, "variances": variances,
        }))
        assert "Q" in result
        assert "df" in result
        assert "p_Q" in result
        assert "I2" in result
        assert "tau2" in result
        assert "H" in result
        assert "interpretation" in result

    def test_homogeneous_low_heterogeneity(self, engine, homogeneous_effects):
        effects, variances = homogeneous_effects
        result = json.loads(engine.heterogeneity({
            "effects": effects, "variances": variances,
        }))
        # Very similar effects should have low I2
        assert result["I2"] < 50
        assert result["Q"] < 20  # Q should be small

    def test_heterogeneous_high_heterogeneity(self, engine, heterogeneous_effects):
        effects, variances = heterogeneous_effects
        result = json.loads(engine.heterogeneity({
            "effects": effects, "variances": variances,
        }))
        assert result["I2"] > 50

    def test_df_equals_k_minus_1(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        result = json.loads(engine.heterogeneity({
            "effects": effects, "variances": variances,
        }))
        assert result["df"] == len(effects) - 1

    def test_too_few_studies_error(self, engine):
        result = json.loads(engine.heterogeneity({
            "effects": [0.5], "variances": [0.1],
        }))
        assert "error" in result

    def test_interpretation_levels(self, engine):
        # Test with moderate heterogeneity data
        effects = [0.3, 0.5, 0.7, 0.4, 0.6]
        variances = [0.02, 0.03, 0.025, 0.02, 0.03]
        result = json.loads(engine.heterogeneity({
            "effects": effects, "variances": variances,
        }))
        assert result["interpretation"] in (
            "Low heterogeneity", "Moderate heterogeneity", "High heterogeneity"
        )


# ===========================================================================
# bias_test
# ===========================================================================

class TestBiasTest:

    def test_egger_test(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        result = json.loads(engine.bias_test({
            "effects": effects, "variances": variances, "test": "egger",
        }))
        assert result["test"] == "Egger's test"
        assert "intercept" in result
        assert "p_value" in result
        assert "interpretation" in result

    def test_begg_test(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        result = json.loads(engine.bias_test({
            "effects": effects, "variances": variances, "test": "begg",
        }))
        assert result["test"] == "Begg's rank correlation test"
        assert "tau" in result
        assert "p_value" in result
        assert -1 <= result["tau"] <= 1

    def test_fail_safe_n(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        result = json.loads(engine.bias_test({
            "effects": effects, "variances": variances, "test": "fail_safe",
        }))
        assert result["test"] == "Fail-safe N (Rosenthal)"
        assert "N_fail_safe" in result
        assert result["N_fail_safe"] >= 0

    def test_too_few_studies_for_bias_error(self, engine):
        result = json.loads(engine.bias_test({
            "effects": [0.5, 0.6], "variances": [0.1, 0.2], "test": "egger",
        }))
        assert "error" in result

    def test_unknown_test_error(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        result = json.loads(engine.bias_test({
            "effects": effects, "variances": variances, "test": "unknown",
        }))
        assert "error" in result

    def test_egger_interpretation_exists(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        result = json.loads(engine.bias_test({
            "effects": effects, "variances": variances, "test": "egger",
        }))
        assert len(result["interpretation"]) > 10

    def test_fail_safe_tolerance(self, engine, bcg_vaccine_data):
        effects, variances, _ = bcg_vaccine_data
        result = json.loads(engine.bias_test({
            "effects": effects, "variances": variances, "test": "fail_safe",
        }))
        assert "tolerance_threshold" in result
        assert result["tolerance_threshold"] == 5 * len(effects) + 10


# ===========================================================================
# subgroup
# ===========================================================================

class TestSubgroup:

    def test_basic_subgroup_analysis(self, engine, bcg_vaccine_data):
        effects, variances, names = bcg_vaccine_data
        # Split into 2 subgroups: first 7 vs last 6
        subgroups = ["group_A"] * 7 + ["group_B"] * 6
        result = json.loads(engine.subgroup({
            "effects": effects, "variances": variances,
            "subgroups": subgroups, "study_names": names,
        }))
        assert "subgroups" in result
        assert "Q_between" in result
        assert "Q_within" in result
        assert "Q_total" in result
        assert result["n_subgroups"] == 2

    def test_subgroup_pooled_effects(self, engine):
        effects = [0.2, 0.3, 0.4, 1.5, 1.6, 1.7]
        variances = [0.05, 0.04, 0.06, 0.05, 0.04, 0.06]
        subgroups = ["low", "low", "low", "high", "high", "high"]
        result = json.loads(engine.subgroup({
            "effects": effects, "variances": variances,
            "subgroups": subgroups,
        }))
        sg_results = result["subgroups"]
        assert len(sg_results) == 2
        # Group effects should differ
        for sg in sg_results:
            assert "pooled_effect" in sg
            assert "k" in sg
            assert "ci_low" in sg
            assert "ci_high" in sg

    def test_Q_decomposition(self, engine):
        effects = [0.2, 0.3, 0.4, 1.5, 1.6, 1.7]
        variances = [0.05, 0.04, 0.06, 0.05, 0.04, 0.06]
        subgroups = ["A", "A", "A", "B", "B", "B"]
        result = json.loads(engine.subgroup({
            "effects": effects, "variances": variances,
            "subgroups": subgroups,
        }))
        # Q_total = Q_within + Q_between (approximately)
        total = result["Q_total"]
        within = result["Q_within"]
        between = result["Q_between"]
        assert abs(total - (within + between)) < 0.01

    def test_single_subgroup_error(self, engine):
        effects = [0.2, 0.3, 0.4]
        variances = [0.05, 0.04, 0.06]
        subgroups = ["A", "A", "A"]
        result = json.loads(engine.subgroup({
            "effects": effects, "variances": variances,
            "subgroups": subgroups,
        }))
        assert "error" in result

    def test_subgroup_length_mismatch_error(self, engine):
        result = json.loads(engine.subgroup({
            "effects": [0.2, 0.3], "variances": [0.1, 0.2],
            "subgroups": ["A"],
        }))
        assert "error" in result

    def test_three_subgroups(self, engine):
        effects = [0.2, 0.3, 1.0, 1.1, 2.0, 2.1]
        variances = [0.04] * 6
        subgroups = ["A", "A", "B", "B", "C", "C"]
        result = json.loads(engine.subgroup({
            "effects": effects, "variances": variances,
            "subgroups": subgroups,
        }))
        assert result["n_subgroups"] == 3
        assert len(result["subgroups"]) == 3
        assert result["df_between"] == 2
