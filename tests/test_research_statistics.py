"""Tests for StatisticalEngine – comprehensive pytest suite.

Uses real data generated with numpy.  Covers every public method and
common edge cases.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sophia.research.statistics import HAS_PINGOUIN, HAS_SCIPY, StatisticalEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> StatisticalEngine:
    return StatisticalEngine()


@pytest.fixture
def normal_groups():
    """Two normal samples with different means."""
    np.random.seed(42)
    g1 = np.random.normal(0, 1, 100).tolist()
    g2 = np.random.normal(0.5, 1, 100).tolist()
    return g1, g2


@pytest.fixture
def paired_groups():
    """Two paired samples (pre/post)."""
    np.random.seed(99)
    pre = np.random.normal(10, 2, 50).tolist()
    post = [x + np.random.normal(1.5, 0.5) for x in pre]
    return pre, post


@pytest.fixture
def three_groups():
    """Three groups for ANOVA."""
    np.random.seed(7)
    g1 = np.random.normal(5, 1, 30).tolist()
    g2 = np.random.normal(6, 1, 30).tolist()
    g3 = np.random.normal(7, 1, 30).tolist()
    return [g1, g2, g3]


@pytest.fixture
def non_normal_groups():
    """Exponentially distributed (non-normal) groups."""
    np.random.seed(123)
    g1 = np.random.exponential(2, 80).tolist()
    g2 = np.random.exponential(3, 80).tolist()
    return g1, g2


@pytest.fixture
def contingency_2x2():
    return [[10, 20], [30, 40]]


@pytest.fixture
def contingency_3x3():
    return [[10, 20, 30], [15, 25, 35], [20, 30, 40]]


# ===========================================================================
# describe
# ===========================================================================

class TestDescribe:

    def test_basic_output_fields(self, engine, normal_groups):
        g1, _ = normal_groups
        result = json.loads(engine.describe({"data": g1}))
        for key in ("n", "mean", "std", "min", "q1", "median", "q3", "max",
                     "skew", "kurtosis", "se", "ci_95"):
            assert key in result, f"Missing key: {key}"
        assert result["n"] == 100

    def test_mean_and_std_reasonable(self, engine):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = json.loads(engine.describe({"data": data}))
        assert abs(result["mean"] - 3.0) < 1e-10
        assert abs(result["std"] - math.sqrt(2.5)) < 1e-10

    def test_median_percentiles(self, engine):
        data = list(range(1, 101))  # 1..100
        result = json.loads(engine.describe({"data": data}))
        assert result["median"] == 50.5
        assert result["q1"] == pytest.approx(25.75, abs=0.5)
        assert result["q3"] == pytest.approx(75.25, abs=0.5)

    def test_empty_data_error(self, engine):
        result = json.loads(engine.describe({"data": []}))
        assert "error" in result

    def test_none_data_error(self, engine):
        result = json.loads(engine.describe({"data": None}))
        assert "error" in result

    def test_ci_95_present(self, engine, normal_groups):
        g1, _ = normal_groups
        result = json.loads(engine.describe({"data": g1}))
        assert result["ci_95"] is not None
        assert len(result["ci_95"]) == 2
        assert result["ci_95"][0] < result["mean"]
        assert result["ci_95"][1] > result["mean"]

    def test_single_value(self, engine):
        result = json.loads(engine.describe({"data": [42.0]}))
        assert result["n"] == 1
        assert result["mean"] == 42.0
        assert result["std"] == 0.0


# ===========================================================================
# ttest
# ===========================================================================

class TestTTest:

    def test_independent_ttest_returns_fields(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.ttest({"group1": g1, "group2": g2}))
        assert result["test"] == "independent t-test"
        assert "t" in result
        assert "p" in result
        assert "df" in result
        assert "cohen_d" in result

    def test_paired_ttest(self, engine, paired_groups):
        pre, post = paired_groups
        result = json.loads(engine.ttest({
            "group1": pre, "group2": post, "paired": True
        }))
        assert "paired" in result["test"]
        assert result["df"] == 49

    def test_welch_ttest(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.ttest({
            "group1": g1, "group2": g2, "welch": True
        }))
        assert "Welch" in result["test"]

    def test_one_sample_ttest(self, engine):
        np.random.seed(10)
        data = np.random.normal(5, 1, 50).tolist()
        result = json.loads(engine.ttest({"group1": data, "popmean": 5.0}))
        assert "one-sample" in result["test"]
        assert result["p"] is not None
        assert result["p"] > 0.01  # should not reject H0

    def test_different_means_rejects_null(self, engine):
        np.random.seed(55)
        g1 = np.random.normal(0, 1, 200).tolist()
        g2 = np.random.normal(2, 1, 200).tolist()
        result = json.loads(engine.ttest({"group1": g1, "group2": g2}))
        assert result["p"] is not None
        assert result["p"] < 0.001

    def test_empty_group_error(self, engine):
        result = json.loads(engine.ttest({"group1": [], "group2": [1, 2]}))
        assert "error" in result

    def test_paired_length_mismatch_error(self, engine):
        result = json.loads(engine.ttest({
            "group1": [1, 2, 3], "group2": [1, 2], "paired": True
        }))
        assert "error" in result

    def test_cohen_d_magnitude(self, engine):
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 100).tolist()
        g2 = np.random.normal(5, 1, 100).tolist()
        result = json.loads(engine.ttest({"group1": g1, "group2": g2}))
        # With means differing by 5 SD, cohen_d should be very large
        assert result["cohen_d"] is not None
        assert abs(result["cohen_d"]) > 2.0


# ===========================================================================
# anova
# ===========================================================================

class TestAnova:

    def test_one_way_anova_fields(self, engine, three_groups):
        result = json.loads(engine.anova({"data": three_groups}))
        assert result["test"] == "one-way ANOVA"
        for key in ("F", "p", "np2", "df1", "df2"):
            assert key in result

    def test_anova_rejects_with_different_means(self, engine):
        np.random.seed(10)
        g1 = np.random.normal(0, 1, 100).tolist()
        g2 = np.random.normal(3, 1, 100).tolist()
        g3 = np.random.normal(6, 1, 100).tolist()
        result = json.loads(engine.anova({"data": [g1, g2, g3]}))
        assert result["p"] is not None
        assert result["p"] < 0.001

    def test_anova_with_group_labels(self, engine, three_groups):
        result = json.loads(engine.anova({
            "data": three_groups,
            "groups": ["control", "treatment_A", "treatment_B"]
        }))
        assert result["source"] == "group"

    def test_repeated_measures_anova(self, engine):
        np.random.seed(77)
        g1 = np.random.normal(5, 1, 30).tolist()
        g2 = [x + np.random.normal(0.5, 0.3) for x in g1]
        g3 = [x + np.random.normal(1.0, 0.3) for x in g1]
        result = json.loads(engine.anova({
            "data": [g1, g2, g3], "repeated": True
        }))
        assert "repeated-measures" in result["test"]
        assert "F" in result

    def test_welch_anova(self, engine, three_groups):
        result = json.loads(engine.anova({
            "data": three_groups, "type": "welch"
        }))
        assert "Welch" in result["test"]

    def test_single_group_error(self, engine):
        result = json.loads(engine.anova({"data": [[1, 2, 3]]}))
        assert "error" in result

    def test_too_few_values_per_group_error(self, engine):
        result = json.loads(engine.anova({"data": [[1], [2, 3]]}))
        assert "error" in result

    def test_mismatched_labels_error(self, engine, three_groups):
        result = json.loads(engine.anova({
            "data": three_groups, "groups": ["a", "b"]
        }))
        assert "error" in result


# ===========================================================================
# chi_square
# ===========================================================================

class TestChiSquare:

    def test_independence_test(self, engine, contingency_2x2):
        result = json.loads(engine.chi_square({"table": contingency_2x2}))
        assert "chi2" in result
        assert "p" in result
        assert "dof" in result
        assert "cramers_v" in result
        assert result["dof"] == 1

    def test_goodness_of_fit(self, engine):
        observed = [10, 20, 30, 40]
        result = json.loads(engine.chi_square({
            "table": observed, "test": "goodness"
        }))
        assert "goodness-of-fit" in result["test"]

    def test_fisher_exact(self, engine, contingency_2x2):
        result = json.loads(engine.chi_square({
            "table": contingency_2x2, "test": "fisher"
        }))
        assert "odds_ratio" in result
        assert "p" in result

    def test_fisher_requires_2x2(self, engine, contingency_3x3):
        result = json.loads(engine.chi_square({
            "table": contingency_3x3, "test": "fisher"
        }))
        assert "error" in result

    def test_cramers_v_range(self, engine, contingency_3x3):
        result = json.loads(engine.chi_square({"table": contingency_3x3}))
        cv = result["cramers_v"]
        assert 0.0 <= cv <= 1.0

    def test_empty_table_error(self, engine):
        result = json.loads(engine.chi_square({"table": []}))
        assert "error" in result


# ===========================================================================
# nonparametric
# ===========================================================================

class TestNonparametric:

    def test_mann_whitney(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.nonparametric({
            "groups": [g1, g2], "test": "mann-whitney"
        }))
        assert result["test"] == "Mann-Whitney U test"
        assert "U" in result
        assert "p" in result

    def test_wilcoxon(self, engine, paired_groups):
        pre, post = paired_groups
        result = json.loads(engine.nonparametric({
            "groups": [pre, post], "test": "wilcoxon"
        }))
        assert "Wilcoxon" in result["test"]

    def test_kruskal(self, engine, three_groups):
        result = json.loads(engine.nonparametric({
            "groups": three_groups, "test": "kruskal"
        }))
        assert "Kruskal-Wallis" in result["test"]
        assert "H" in result

    def test_friedman(self, engine):
        np.random.seed(88)
        g1 = np.random.normal(5, 1, 20).tolist()
        g2 = [x + 0.5 for x in g1]
        g3 = [x + 1.0 for x in g1]
        result = json.loads(engine.nonparametric({
            "groups": [g1, g2, g3], "test": "friedman"
        }))
        assert "Friedman" in result["test"]

    def test_mann_whitney_different_distributions(self, engine, non_normal_groups):
        g1, g2 = non_normal_groups
        result = json.loads(engine.nonparametric({
            "groups": [g1, g2], "test": "mann-whitney"
        }))
        assert result["p"] is not None

    def test_friedman_unequal_sizes_error(self, engine):
        result = json.loads(engine.nonparametric({
            "groups": [[1, 2, 3], [4, 5], [7, 8, 9]], "test": "friedman"
        }))
        assert "error" in result

    def test_unknown_test_error(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.nonparametric({
            "groups": [g1, g2], "test": "unknown_test"
        }))
        assert "error" in result


# ===========================================================================
# correlation
# ===========================================================================

class TestCorrelation:

    def test_pearson(self, engine):
        np.random.seed(42)
        x = np.random.normal(0, 1, 100).tolist()
        y = [xi + np.random.normal(0, 0.3) for xi in x]
        result = json.loads(engine.correlation({"x": x, "y": y}))
        assert result["method"] == "pearson"
        assert result["r"] > 0.8
        assert result["p"] < 0.001
        assert result["r_squared"] == pytest.approx(result["r"] ** 2, rel=1e-6)

    def test_spearman(self, engine):
        np.random.seed(11)
        x = list(range(50))
        y = list(range(50))
        y.reverse()
        result = json.loads(engine.correlation({
            "x": x, "y": y, "method": "spearman"
        }))
        assert result["r"] == pytest.approx(-1.0, abs=1e-10)

    def test_kendall(self, engine):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        result = json.loads(engine.correlation({
            "x": x, "y": y, "method": "kendall"
        }))
        assert result["r"] == pytest.approx(-1.0, abs=1e-10)

    def test_length_mismatch_error(self, engine):
        result = json.loads(engine.correlation({"x": [1, 2], "y": [1]}))
        assert "error" in result

    def test_too_few_observations_error(self, engine):
        result = json.loads(engine.correlation({"x": [1, 2], "y": [3, 4]}))
        assert "error" in result

    def test_pearson_ci(self, engine):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200).tolist()
        y = [xi * 0.8 + np.random.normal(0, 0.5) for xi in x]
        result = json.loads(engine.correlation({"x": x, "y": y}))
        assert result["ci_95"] is not None
        assert result["ci_95"][0] < result["r"] < result["ci_95"][1]

    def test_zero_correlation(self, engine):
        np.random.seed(77)
        x = np.random.normal(0, 1, 500).tolist()
        y = np.random.normal(0, 1, 500).tolist()
        result = json.loads(engine.correlation({"x": x, "y": y}))
        assert abs(result["r"]) < 0.15


# ===========================================================================
# regression
# ===========================================================================

class TestRegression:

    def test_simple_regression(self, engine):
        np.random.seed(42)
        x = np.random.normal(0, 1, 100).tolist()
        y = [2.0 * xi + 3.0 + np.random.normal(0, 0.5) for xi in x]
        result = json.loads(engine.regression({
            "y": y, "X": x, "x_names": ["x1"], "y_name": "y"
        }))
        assert result["r_squared"] > 0.8
        coeffs = result["coefficients"]
        assert abs(coeffs["intercept"] - 3.0) < 0.5
        assert abs(coeffs["x1"] - 2.0) < 0.5

    def test_multiple_regression(self, engine):
        np.random.seed(123)
        x1 = np.random.normal(0, 1, 100).tolist()
        x2 = np.random.normal(0, 1, 100).tolist()
        y = [1.5 * a + 2.5 * b + 5.0 + np.random.normal(0, 0.5) for a, b in zip(x1, x2)]
        result = json.loads(engine.regression({
            "y": y,
            "X": [x1, x2],
            "x_names": ["x1", "x2"],
            "y_name": "y"
        }))
        assert result["r_squared"] > 0.85
        assert result["predictors"] == 2

    def test_regression_output_fields(self, engine):
        np.random.seed(10)
        x = np.random.normal(0, 1, 50).tolist()
        y = [xi + np.random.normal(0, 0.5) for xi in x]
        result = json.loads(engine.regression({"y": y, "X": x}))
        for key in ("coefficients", "r_squared", "adj_r_squared", "F", "F_pvalue",
                     "std_errors", "t_stats", "p_values"):
            assert key in result, f"Missing key: {key}"

    def test_regression_length_mismatch(self, engine):
        result = json.loads(engine.regression({
            "y": [1, 2, 3], "X": [1, 2]
        }))
        assert "error" in result

    def test_regression_missing_y(self, engine):
        result = json.loads(engine.regression({"X": [[1, 2], [3, 4]]}))
        assert "error" in result

    def test_regression_missing_X(self, engine):
        result = json.loads(engine.regression({"y": [1, 2, 3]}))
        assert "error" in result


# ===========================================================================
# normality
# ===========================================================================

class TestNormality:

    def test_shapiro_on_normal_data(self, engine):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100).tolist()
        result = json.loads(engine.normality({"data": data, "test": "shapiro"}))
        assert "shapiro_wilk" in result
        assert result["shapiro_wilk"]["p"] > 0.05

    def test_shapiro_on_non_normal(self, engine):
        np.random.seed(42)
        data = np.random.exponential(2, 200).tolist()
        result = json.loads(engine.normality({"data": data, "test": "shapiro"}))
        assert result["shapiro_wilk"]["p"] < 0.05

    def test_ks_test(self, engine):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100).tolist()
        result = json.loads(engine.normality({"data": data, "test": "ks"}))
        assert "kolmogorov_smirnov" in result

    def test_anderson_test(self, engine):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100).tolist()
        result = json.loads(engine.normality({"data": data, "test": "anderson"}))
        assert "anderson_darling" in result
        assert "critical_values" in result["anderson_darling"]

    def test_all_normality_tests(self, engine):
        np.random.seed(42)
        data = np.random.normal(5, 2, 150).tolist()
        result = json.loads(engine.normality({"data": data, "test": "all"}))
        assert "shapiro_wilk" in result
        assert "kolmogorov_smirnov" in result
        assert "anderson_darling" in result

    def test_too_few_data_points_error(self, engine):
        result = json.loads(engine.normality({"data": [1, 2], "test": "shapiro"}))
        assert "error" in result

    def test_unknown_test_error(self, engine):
        result = json.loads(engine.normality({"data": [1, 2, 3, 4, 5], "test": "unknown"}))
        assert "error" in result


# ===========================================================================
# effect_size
# ===========================================================================

class TestEffectSize:

    def test_cohens_d(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.effect_size({
            "group1": g1, "group2": g2, "metric": "cohens_d"
        }))
        assert result["metric"] == "Cohen's d"
        assert "value" in result
        assert "magnitude" in result

    def test_hedges_g(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.effect_size({
            "group1": g1, "group2": g2, "metric": "hedges_g"
        }))
        assert result["metric"] == "Hedges' g"
        # Hedges' g should be slightly smaller in absolute value than Cohen's d
        d_result = json.loads(engine.effect_size({
            "group1": g1, "group2": g2, "metric": "cohens_d"
        }))
        assert abs(result["value"]) <= abs(d_result["value"]) + 0.01

    def test_eta_squared(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.effect_size({
            "group1": g1, "group2": g2, "metric": "eta_squared"
        }))
        assert result["metric"] == "Eta-squared"
        assert 0 <= result["value"] <= 1

    def test_odds_ratio(self, engine, contingency_2x2):
        result = json.loads(engine.effect_size({
            "metric": "odds_ratio", "table": contingency_2x2
        }))
        assert result["metric"] == "Odds ratio"
        assert result["value"] is not None

    def test_odds_ratio_with_ci(self, engine):
        table = [[50, 30], [20, 60]]
        result = json.loads(engine.effect_size({
            "metric": "odds_ratio", "table": table
        }))
        assert result["ci_95"] is not None
        assert len(result["ci_95"]) == 2

    def test_effect_size_large(self, engine):
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 100).tolist()
        g2 = np.random.normal(5, 1, 100).tolist()
        result = json.loads(engine.effect_size({
            "group1": g1, "group2": g2, "metric": "cohens_d"
        }))
        assert result["value"] is not None
        assert result["magnitude"] == "large"

    def test_missing_groups_error(self, engine):
        result = json.loads(engine.effect_size({"metric": "cohens_d"}))
        assert "error" in result


# ===========================================================================
# bayesian
# ===========================================================================

class TestBayesian:

    @pytest.mark.skipif(not HAS_PINGOUIN, reason="pingouin not installed")
    def test_bayesian_ttest_returns_bf10(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.bayesian({"group1": g1, "group2": g2}))
        assert "BF10" in result
        assert result["BF10"] is not None
        assert "interpretation" in result

    @pytest.mark.skipif(not HAS_PINGOUIN, reason="pingouin not installed")
    def test_bayesian_large_effect(self, engine):
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 200).tolist()
        g2 = np.random.normal(3, 1, 200).tolist()
        result = json.loads(engine.bayesian({"group1": g1, "group2": g2}))
        assert result["BF10"] > 100  # decisive evidence for H1

    def test_bayesian_empty_groups_error(self, engine):
        result = json.loads(engine.bayesian({"group1": [], "group2": [1]}))
        assert "error" in result


# ===========================================================================
# auto_test
# ===========================================================================

class TestAutoTest:

    def test_auto_two_normal_groups(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.auto_test({
            "groups": [g1, g2], "paired": False
        }))
        assert "recommended_test" in result
        assert "test_result" in result
        assert "reasoning" in result
        assert len(result["reasoning"]) >= 3

    def test_auto_two_non_normal_groups(self, engine, non_normal_groups):
        g1, g2 = non_normal_groups
        result = json.loads(engine.auto_test({
            "groups": [g1, g2], "paired": False
        }))
        # With exponential data, Shapiro-Wilk should detect non-normality
        assert result["recommended_test"] in ("Mann-Whitney U test", "independent t-test")

    def test_auto_three_groups(self, engine, three_groups):
        result = json.loads(engine.auto_test({
            "groups": three_groups, "paired": False
        }))
        assert result["n_groups"] == 3
        assert result["recommended_test"] in (
            "one-way ANOVA", "Kruskal-Wallis test"
        )

    def test_auto_paired_groups(self, engine, paired_groups):
        pre, post = paired_groups
        result = json.loads(engine.auto_test({
            "groups": [pre, post], "paired": True
        }))
        assert result["paired"] is True
        assert "paired" in result["recommended_test"] or "Wilcoxon" in result["recommended_test"]

    def test_auto_with_data_dict(self, engine, normal_groups):
        g1, g2 = normal_groups
        result = json.loads(engine.auto_test({
            "data": {"treatment": g1, "control": g2}
        }))
        assert result["n_groups"] == 2

    def test_auto_no_data_error(self, engine):
        result = json.loads(engine.auto_test({}))
        assert "error" in result

    def test_auto_single_group_error(self, engine, normal_groups):
        g1, _ = normal_groups
        result = json.loads(engine.auto_test({"groups": [g1]}))
        assert "error" in result

    def test_auto_normality_checks_included(self, engine, three_groups):
        result = json.loads(engine.auto_test({"groups": three_groups}))
        assert "normality_checks" in result
        assert len(result["normality_checks"]) == 3


# ===========================================================================
# Edge cases and integration
# ===========================================================================

class TestEdgeCases:

    def test_nan_values_filtered(self, engine):
        data = [1.0, 2.0, float("nan"), 3.0, 4.0, float("nan"), 5.0]
        result = json.loads(engine.describe({"data": data}))
        assert result["n"] == 5

    def test_large_dataset(self, engine):
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000).tolist()
        result = json.loads(engine.describe({"data": data}))
        assert result["n"] == 10000
        assert abs(result["mean"]) < 0.1

    def test_all_same_values(self, engine):
        data = [5.0] * 50
        result = json.loads(engine.describe({"data": data}))
        assert result["std"] == 0.0
        assert result["mean"] == 5.0

    def test_json_output_is_valid(self, engine, normal_groups):
        g1, g2 = normal_groups
        for method_name in ("describe", "ttest", "correlation", "normality"):
            if method_name == "describe":
                raw = engine.describe({"data": g1})
            elif method_name == "ttest":
                raw = engine.ttest({"group1": g1, "group2": g2})
            elif method_name == "correlation":
                raw = engine.correlation({"x": g1, "y": g2})
            elif method_name == "normality":
                raw = engine.normality({"data": g1})
            # Must be valid JSON
            parsed = json.loads(raw)
            assert isinstance(parsed, dict)

    def test_negative_data(self, engine):
        data = [-5.0, -3.0, -1.0, -2.0, -4.0]
        result = json.loads(engine.describe({"data": data}))
        assert result["mean"] < 0
        assert result["min"] == -5.0
        assert result["max"] == -1.0

    def test_chi_square_expected_frequencies_shape(self, engine, contingency_3x3):
        result = json.loads(engine.chi_square({"table": contingency_3x3}))
        expected = result["expected"]
        assert len(expected) == 3
        assert len(expected[0]) == 3

    def test_nonparametric_single_group_error(self, engine, normal_groups):
        g1, _ = normal_groups
        result = json.loads(engine.nonparametric({
            "groups": [g1], "test": "mann-whitney"
        }))
        assert "error" in result
