"""Tests for SurveyEngine -- comprehensive pytest suite.

Uses real data generated with numpy.  Covers every public method and
common edge cases.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sophia.research.survey import SurveyEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> SurveyEngine:
    return SurveyEngine()


@pytest.fixture
def reliable_items():
    """5 items with high internal consistency (50 respondents)."""
    np.random.seed(42)
    n = 50
    base = np.random.normal(3, 0.5, n)
    items = []
    for _ in range(5):
        item = base + np.random.normal(0, 0.2, n)
        items.append(item.tolist())
    return items


@pytest.fixture
def likert_data():
    """Simulated Likert data (100 respondents, 5 items, scale 1-5)."""
    np.random.seed(123)
    n = 100
    data = []
    for _ in range(n):
        base = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.3, 0.35, 0.2])
        row = [max(1, min(5, base + np.random.randint(-1, 2))) for _ in range(5)]
        data.append(row)
    return data


@pytest.fixture
def survey_items():
    """10 items with moderate reliability (100 respondents)."""
    np.random.seed(77)
    n = 100
    latent = np.random.normal(0, 1, n)
    items = []
    for i in range(10):
        item = 0.5 * latent + np.random.normal(0, 0.5, n)
        items.append(item.tolist())
    return items


# ===========================================================================
# cronbach
# ===========================================================================

class TestCronbach:

    def test_basic_alpha_computation(self, engine, reliable_items):
        result = json.loads(engine.cronbach({"items": reliable_items}))
        assert "alpha" in result
        assert "n_items" in result
        assert "n_responses" in result
        assert result["n_items"] == 5
        assert result["n_responses"] == 50

    def test_high_reliability_items(self, engine, reliable_items):
        result = json.loads(engine.cronbach({"items": reliable_items}))
        # These items are highly correlated, alpha should be high
        assert result["alpha"] > 0.7

    def test_item_total_correlations(self, engine, reliable_items):
        result = json.loads(engine.cronbach({
            "items": reliable_items,
            "item_names": ["q1", "q2", "q3", "q4", "q5"],
        }))
        assert "item_total_corr" in result
        itc = result["item_total_corr"]
        assert len(itc) == 5
        for name in ["q1", "q2", "q3", "q4", "q5"]:
            assert name in itc

    def test_alpha_if_deleted(self, engine, reliable_items):
        result = json.loads(engine.cronbach({"items": reliable_items}))
        assert "alpha_if_deleted" in result
        aid = result["alpha_if_deleted"]
        assert len(aid) == 5
        # All alpha-if-deleted should be reasonable
        for key, val in aid.items():
            if val is not None:
                assert -1.0 <= val <= 1.0

    def test_too_few_items_error(self, engine):
        result = json.loads(engine.cronbach({"items": [[1, 2, 3]]}))
        assert "error" in result

    def test_item_names_mismatch_error(self, engine, reliable_items):
        result = json.loads(engine.cronbach({
            "items": reliable_items,
            "item_names": ["a", "b"],
        }))
        assert "error" in result


# ===========================================================================
# factor_analysis
# ===========================================================================

class TestFactorAnalysis:

    def test_basic_factor_extraction(self, engine):
        np.random.seed(42)
        data = np.random.multivariate_normal(
            [0, 0, 0, 0],
            [[1, 0.8, 0.3, 0.1],
             [0.8, 1, 0.2, 0.1],
             [0.3, 0.2, 1, 0.7],
             [0.1, 0.1, 0.7, 1]],
            200
        ).tolist()
        result = json.loads(engine.factor_analysis({
            "data": data, "n_factors": 2
        }))
        assert "loadings" in result
        assert "variance_explained" in result
        assert "communalities" in result
        assert result["n_factors"] == 2
        assert len(result["loadings"]) == 4  # 4 variables
        assert len(result["communalities"]) == 4

    def test_varimax_rotation(self, engine):
        np.random.seed(42)
        data = np.random.multivariate_normal(
            [0, 0, 0, 0],
            [[1, 0.8, 0.2, 0.1],
             [0.8, 1, 0.2, 0.1],
             [0.2, 0.2, 1, 0.8],
             [0.1, 0.1, 0.8, 1]],
            150
        ).tolist()
        result = json.loads(engine.factor_analysis({
            "data": data, "n_factors": 2, "rotation": "varimax"
        }))
        assert result["rotation"] == "varimax"
        loadings = np.array(result["loadings"])
        # After varimax, loadings should tend toward 0 or 1
        assert loadings.shape == (4, 2)

    def test_no_rotation(self, engine):
        np.random.seed(42)
        data = np.random.randn(100, 3).tolist()
        result = json.loads(engine.factor_analysis({
            "data": data, "n_factors": 2, "rotation": "none"
        }))
        assert result["rotation"] == "none"

    def test_too_few_observations_error(self, engine):
        result = json.loads(engine.factor_analysis({
            "data": [[1, 2], [3, 4]], "n_factors": 1
        }))
        assert "error" in result

    def test_n_factors_exceeds_vars_error(self, engine):
        data = np.random.randn(50, 3).tolist()
        result = json.loads(engine.factor_analysis({
            "data": data, "n_factors": 5
        }))
        assert "error" in result

    def test_eigenvalues_returned(self, engine):
        np.random.seed(42)
        data = np.random.randn(100, 4).tolist()
        result = json.loads(engine.factor_analysis({
            "data": data, "n_factors": 2
        }))
        assert "eigenvalues_all" in result
        assert len(result["eigenvalues_all"]) == 4


# ===========================================================================
# item_analysis
# ===========================================================================

class TestItemAnalysis:

    def test_basic_item_analysis(self, engine, survey_items):
        result = json.loads(engine.item_analysis({"items": survey_items}))
        assert "items" in result
        assert "overall" in result
        assert len(result["items"]) == 10
        assert result["overall"]["n_items"] == 10
        assert result["overall"]["n_responses"] == 100

    def test_difficulty_and_discrimination(self, engine, survey_items):
        result = json.loads(engine.item_analysis({
            "items": survey_items,
            "item_names": [f"q{i}" for i in range(10)],
        }))
        for item_result in result["items"]:
            assert "difficulty" in item_result
            assert "discrimination" in item_result
            assert "mean" in item_result
            assert "std" in item_result
            assert "alpha_if_deleted" in item_result

    def test_with_provided_total_score(self, engine):
        items = [[4, 3, 5, 2, 4], [3, 4, 4, 3, 5]]
        total = [7, 7, 9, 5, 9]
        result = json.loads(engine.item_analysis({
            "items": items, "total_score": total,
        }))
        assert len(result["items"]) == 2

    def test_total_score_length_mismatch_error(self, engine):
        items = [[1, 2, 3], [4, 5, 6]]
        result = json.loads(engine.item_analysis({
            "items": items, "total_score": [5, 7],
        }))
        assert "error" in result

    def test_empty_items_error(self, engine):
        result = json.loads(engine.item_analysis({"items": []}))
        assert "error" in result


# ===========================================================================
# sample_size
# ===========================================================================

class TestSampleSize:

    def test_basic_sample_size(self, engine):
        result = json.loads(engine.sample_size({
            "population": 10000, "margin_error": 0.05,
            "confidence": 0.95,
        }))
        assert "n_simple" in result
        assert "n_adjusted_design_effect" in result
        assert result["n_simple"] > 0

    def test_cochrains_formula_standard(self, engine):
        """Standard case: 95% CI, 5% margin, p=0.5, infinite pop -> n=385."""
        result = json.loads(engine.sample_size({
            "margin_error": 0.05, "confidence": 0.95,
        }))
        # n0 = (1.96^2 * 0.5 * 0.5) / 0.05^2 = 384.16 -> 385
        assert result["n_simple"] == 385

    def test_finite_population_adjustment(self, engine):
        result = json.loads(engine.sample_size({
            "population": 500, "margin_error": 0.05, "confidence": 0.95,
        }))
        # Finite population correction should reduce the sample
        assert result["n_adjusted_finite_population"] is not None
        assert result["n_adjusted_finite_population"] < result["n_simple"]

    def test_design_effect(self, engine):
        result_no_de = json.loads(engine.sample_size({
            "margin_error": 0.05, "confidence": 0.95, "design_effect": 1.0,
        }))
        result_de = json.loads(engine.sample_size({
            "margin_error": 0.05, "confidence": 0.95, "design_effect": 2.0,
        }))
        # Design effect of 2 should double the sample
        assert result_de["n_adjusted_design_effect"] == 2 * result_no_de["n_simple"]

    def test_higher_confidence_larger_sample(self, engine):
        result_90 = json.loads(engine.sample_size({"confidence": 0.90, "margin_error": 0.05}))
        result_99 = json.loads(engine.sample_size({"confidence": 0.99, "margin_error": 0.05}))
        assert result_99["n_simple"] > result_90["n_simple"]

    def test_stratum_estimates(self, engine):
        result = json.loads(engine.sample_size({
            "population": 10000, "margin_error": 0.05,
        }))
        assert "n_per_stratum" in result
        assert "2_strata_equal" in result["n_per_stratum"]

    def test_invalid_margin_error(self, engine):
        result = json.loads(engine.sample_size({"margin_error": 0.0}))
        assert "error" in result

    def test_invalid_confidence(self, engine):
        result = json.loads(engine.sample_size({"confidence": 1.0}))
        assert "error" in result


# ===========================================================================
# likert_analysis
# ===========================================================================

class TestLikertAnalysis:

    def test_basic_likert_output(self, engine, likert_data):
        result = json.loads(engine.likert_analysis({"data": likert_data}))
        assert "items" in result
        assert "overall" in result
        assert "inter_item_consistency" in result
        assert len(result["items"]) == 5
        assert result["n_respondents"] == 100
        assert result["n_items"] == 5

    def test_frequency_distribution(self, engine, likert_data):
        result = json.loads(engine.likert_analysis({"data": likert_data}))
        for item_result in result["items"]:
            assert "frequency" in item_result
            freq = item_result["frequency"]
            # Should have entries for all scale points
            for val in ["1", "2", "3", "4", "5"]:
                assert val in freq
                assert "count" in freq[val]
                assert "percentage" in freq[val]

    def test_median_and_iqr(self, engine, likert_data):
        result = json.loads(engine.likert_analysis({"data": likert_data}))
        for item_result in result["items"]:
            assert "median" in item_result
            assert "iqr" in item_result
            assert "q1" in item_result
            assert "q3" in item_result
            assert item_result["iqr"] == pytest.approx(
                item_result["q3"] - item_result["q1"], abs=1e-10
            )

    def test_top_box_bottom_box(self, engine, likert_data):
        result = json.loads(engine.likert_analysis({"data": likert_data}))
        for item_result in result["items"]:
            assert "top_box_pct" in item_result
            assert "bottom_box_pct" in item_result
            assert 0 <= item_result["top_box_pct"] <= 100
            assert 0 <= item_result["bottom_box_pct"] <= 100

    def test_custom_scale_range(self, engine):
        np.random.seed(42)
        data = np.random.randint(1, 8, (100, 3)).tolist()  # 7-point scale
        result = json.loads(engine.likert_analysis({
            "data": data, "scale_min": 1, "scale_max": 7,
        }))
        assert result["scale_range"]["min"] == 1
        assert result["scale_range"]["max"] == 7

    def test_inter_item_consistency(self, engine, likert_data):
        result = json.loads(engine.likert_analysis({"data": likert_data}))
        consistency = result["inter_item_consistency"]
        assert "cronbach_alpha" in consistency
        assert "mean_inter_item_r" in consistency
        if consistency["cronbach_alpha"] is not None:
            assert -1.0 <= consistency["cronbach_alpha"] <= 1.0

    def test_invalid_data_error(self, engine):
        result = json.loads(engine.likert_analysis({"data": [1, 2, 3]}))
        assert "error" in result

    def test_item_names_with_likert(self, engine, likert_data):
        names = ["satisfaction", "quality", "value", "service", "recommend"]
        result = json.loads(engine.likert_analysis({
            "data": likert_data, "item_names": names,
        }))
        returned_names = [it["item"] for it in result["items"]]
        assert returned_names == names
