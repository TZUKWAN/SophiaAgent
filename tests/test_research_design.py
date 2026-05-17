"""Tests for ResearchDesignEngine: DOE, power analysis, random assignment."""
import json
import math

import numpy as np
import pytest

from sophia.research.design import ResearchDesignEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(result: str) -> dict:
    """Parse JSON result string into a dict."""
    return json.loads(result)


@pytest.fixture
def engine():
    return ResearchDesignEngine()


# ===================================================================
# Factorial design tests
# ===================================================================

class TestFactorialDesignFull:
    """Full factorial designs."""

    def test_full_2level_3factors(self, engine):
        res = _parse(engine.factorial_design({
            "factors": 3, "levels": 2, "type": "full",
        }))
        assert res["design_type"] == "full_factorial"
        assert res["runs"] == 8  # 2^3
        assert res["factors"] == 3
        assert len(res["design"]) == 8
        for row in res["design"]:
            assert len(row) == 3
            for val in row:
                assert val in (-1, 1)

    def test_full_3level_2factors(self, engine):
        res = _parse(engine.factorial_design({
            "factors": 2, "levels": 3, "type": "full",
        }))
        assert res["runs"] == 9  # 3^2
        for row in res["design"]:
            assert len(row) == 2

    def test_full_mixed_levels(self, engine):
        res = _parse(engine.factorial_design({
            "factors": 2, "levels": [2, 3], "type": "full",
        }))
        assert res["runs"] == 6  # 2 * 3
        # fullfact returns 0-indexed levels (0, 1) and (0, 1, 2)
        col0_vals = {row[0] for row in res["design"]}
        col1_vals = {row[1] for row in res["design"]}
        assert col0_vals == {0.0, 1.0}
        assert col1_vals == {0.0, 1.0, 2.0}

    def test_full_default_args(self, engine):
        res = _parse(engine.factorial_design({}))
        assert "design" in res
        assert res["factors"] == 3
        assert res["runs"] == 8

    def test_full_factor_names(self, engine):
        res = _parse(engine.factorial_design({"factors": 4, "levels": 2}))
        assert res["factor_names"] == ["X1", "X2", "X3", "X4"]


class TestFactorialDesignFractional:
    """Fractional factorial designs."""

    def test_fractional_explicit_generators(self, engine):
        res = _parse(engine.factorial_design({
            "factors": 3, "type": "fractional", "generators": "a b ab",
        }))
        assert res["design_type"] == "fractional_factorial"
        assert res["runs"] == 4  # 2^(3-1) = 4
        assert res["factors"] == 3
        assert res["generators"] == "a b ab"

    def test_fractional_default_generators(self, engine):
        res = _parse(engine.factorial_design({
            "factors": 4, "type": "fractional",
        }))
        assert res["runs"] == 8  # 2^(4-1)
        assert res["factors"] == 4
        assert "generators" in res

    def test_fractional_resolution_info(self, engine):
        res = _parse(engine.factorial_design({
            "factors": 5, "type": "fractional",
        }))
        assert "resolution" in res


class TestFactorialDesignPlackettBurman:
    """Plackett-Burman screening designs."""

    def test_pb_3factors(self, engine):
        res = _parse(engine.factorial_design({
            "factors": 3, "type": "plackett-burman",
        }))
        assert res["design_type"] == "plackett_burman"
        assert res["factors"] == 3
        assert res["runs"] >= 4  # minimum PB run size

    def test_pb_7factors(self, engine):
        res = _parse(engine.factorial_design({
            "factors": 7, "type": "plackett-burman",
        }))
        assert res["runs"] >= 8
        for row in res["design"]:
            assert len(row) == 7

    def test_invalid_design_type(self, engine):
        res = _parse(engine.factorial_design({"type": "nonexistent"}))
        assert "error" in res


# ===================================================================
# Response-surface design tests
# ===================================================================

class TestResponseSurfaceBoxBehnken:
    """Box-Behnken designs."""

    def test_bb_3factors(self, engine):
        res = _parse(engine.response_surface({
            "factors": 3, "type": "box-behnken",
        }))
        assert res["design_type"] == "box_behnken"
        # Default center=1 gives 13 runs with 1 center point
        assert res["factors"] == 3
        assert res["center_points"] >= 1

    def test_bb_4factors(self, engine):
        res = _parse(engine.response_surface({
            "factors": 4, "type": "box-behnken",
        }))
        # With default center=1: 24 factorial + 1 center = 25
        assert res["runs"] == 25
        for row in res["design"]:
            assert len(row) == 4


class TestResponseSurfaceCCD:
    """Central composite designs."""

    def test_ccf_3factors(self, engine):
        res = _parse(engine.response_surface({
            "factors": 3, "type": "ccf",
        }))
        assert "central_composite_ccf" == res["design_type"]
        assert res["runs"] > 0
        assert res["center_points"] >= 1

    def test_ccc_3factors(self, engine):
        res = _parse(engine.response_surface({
            "factors": 3, "type": "ccc",
        }))
        assert res["design_type"] == "central_composite_ccc"
        assert res["runs"] > 0

    def test_cci_3factors(self, engine):
        res = _parse(engine.response_surface({
            "factors": 3, "type": "cci",
        }))
        assert res["design_type"] == "central_composite_cci"

    def test_rsm_invalid_type(self, engine):
        res = _parse(engine.response_surface({"type": "unknown"}))
        assert "error" in res


# ===================================================================
# Latin Hypercube sampling tests
# ===================================================================

class TestLatinHypercube:
    """Latin Hypercube sampling."""

    def test_lhs_basic(self, engine):
        res = _parse(engine.latin_hypercube({
            "dimensions": 3, "samples": 20,
        }))
        mat = res["samples_matrix"]
        assert len(mat) == 20
        for row in mat:
            assert len(row) == 3
        assert res["criterion"] == "maximin"

    def test_lhs_center_criterion(self, engine):
        res = _parse(engine.latin_hypercube({
            "dimensions": 2, "samples": 10, "criterion": "center",
        }))
        assert res["criterion"] == "center"
        # Values in [0, 1]
        assert res["range"]["min"] >= 0.0
        assert res["range"]["max"] <= 1.0

    def test_lhs_correlation_criterion(self, engine):
        res = _parse(engine.latin_hypercube({
            "dimensions": 4, "samples": 15, "criterion": "correlation",
        }))
        mat = np.array(res["samples_matrix"])
        assert mat.shape == (15, 4)

    def test_lhs_default_args(self, engine):
        res = _parse(engine.latin_hypercube({}))
        assert res["dimensions"] == 2
        assert res["samples"] == 10


# ===================================================================
# Power analysis tests
# ===================================================================

class TestPowerAnalysisTTest:
    """T-test power / sample-size."""

    def test_ttest_compute_n(self, engine):
        res = _parse(engine.power_analysis({
            "test": "ttest", "effect_size": 0.5, "alpha": 0.05,
            "power": 0.80,
        }))
        assert res["result_type"] == "sample_size"
        # Cohen's handbook: n ~ 64 for d=0.5, alpha=0.05, power=0.80
        assert 60 <= res["result_value"] <= 70

    def test_ttest_compute_power(self, engine):
        res = _parse(engine.power_analysis({
            "test": "ttest", "effect_size": 0.5, "alpha": 0.05,
            "n": 64,
        }))
        assert res["result_type"] == "power"
        assert 0.75 <= res["result_value"] <= 0.90

    def test_ttest_large_effect(self, engine):
        res = _parse(engine.power_analysis({
            "test": "ttest", "effect_size": 1.0, "power": 0.80,
        }))
        # Large effect -> small sample needed
        assert res["result_value"] <= 20


class TestPowerAnalysisAnova:
    """ANOVA power / sample-size."""

    def test_anova_compute_n(self, engine):
        res = _parse(engine.power_analysis({
            "test": "anova", "effect_size": 0.25, "k_groups": 3,
            "power": 0.80,
        }))
        assert res["result_type"] == "sample_size"
        # f=0.25, k=3 -> n per group ~ 50-55
        assert 40 <= res["result_value"] <= 70

    def test_anova_compute_power(self, engine):
        res = _parse(engine.power_analysis({
            "test": "anova", "effect_size": 0.25, "k_groups": 3,
            "n": 30,
        }))
        assert res["result_type"] == "power"
        assert 0.0 <= res["result_value"] <= 1.0

    def test_anova_eta_squared_reported(self, engine):
        res = _parse(engine.power_analysis({
            "test": "anova", "effect_size": 0.5, "k_groups": 4,
            "power": 0.80,
        }))
        assert "eta_squared" in res
        expected_eta2 = 0.5 ** 2 / (1 + 0.5 ** 2)
        assert abs(res["eta_squared"] - expected_eta2) < 1e-6


class TestPowerAnalysisCorrelation:
    """Correlation power / sample-size."""

    def test_corr_compute_n(self, engine):
        res = _parse(engine.power_analysis({
            "test": "correlation", "effect_size": 0.3, "power": 0.80,
        }))
        assert res["result_type"] == "sample_size"
        # r=0.3, power=0.80 -> n ~ 85
        assert 80 <= res["result_value"] <= 95

    def test_corr_compute_power(self, engine):
        res = _parse(engine.power_analysis({
            "test": "correlation", "effect_size": 0.3, "n": 100,
        }))
        assert res["result_type"] == "power"
        assert res["result_value"] > 0.80


class TestPowerAnalysisChi2:
    """Chi-square power / sample-size."""

    def test_chi2_compute_n(self, engine):
        res = _parse(engine.power_analysis({
            "test": "chi2", "effect_size": 0.3, "k_groups": 3,
            "power": 0.80,
        }))
        assert res["result_type"] == "sample_size"
        # Should need a moderate total N
        assert res["result_value"] > 20

    def test_chi2_compute_power(self, engine):
        res = _parse(engine.power_analysis({
            "test": "chi2", "effect_size": 0.3, "k_groups": 3,
            "n": 100,
        }))
        assert res["result_type"] == "power"
        assert 0.0 <= res["result_value"] <= 1.0


class TestPowerAnalysisProportion:
    """Proportion test power / sample-size."""

    def test_proportion_compute_n(self, engine):
        res = _parse(engine.power_analysis({
            "test": "proportion", "effect_size": 0.3, "power": 0.80,
        }))
        assert res["result_type"] == "sample_size"
        assert res["result_value"] > 50

    def test_proportion_compute_power(self, engine):
        res = _parse(engine.power_analysis({
            "test": "proportion", "effect_size": 0.3, "n": 100,
        }))
        assert res["result_type"] == "power"
        assert 0.0 <= res["result_value"] <= 1.0

    def test_unknown_test(self, engine):
        res = _parse(engine.power_analysis({"test": "nonexistent"}))
        assert "error" in res


# ===================================================================
# Random assignment tests
# ===================================================================

class TestRandomAssignmentSimple:
    """Simple random assignment."""

    def test_simple_basic(self, engine):
        res = _parse(engine.random_assignment({
            "n": 20, "n_groups": 2, "method": "simple", "seed": 42,
        }))
        assert len(res["assignments"]) == 20
        assert res["n"] == 20
        assert res["n_groups"] == 2
        assert set(res["assignments"]).issubset({0, 1})

    def test_simple_balanced_groups(self, engine):
        res = _parse(engine.random_assignment({
            "n": 100, "n_groups": 4, "method": "simple", "seed": 123,
        }))
        counts = res["group_sizes"]
        # With 100 units and 4 groups, simple random should distribute
        # roughly evenly (each group ~ 25, but not guaranteed exact).
        for g in range(4):
            assert counts.get(str(g), 0) > 10

    def test_simple_reproducible_with_seed(self, engine):
        r1 = _parse(engine.random_assignment({
            "n": 30, "n_groups": 3, "method": "simple", "seed": 99,
        }))
        r2 = _parse(engine.random_assignment({
            "n": 30, "n_groups": 3, "method": "simple", "seed": 99,
        }))
        assert r1["assignments"] == r2["assignments"]

    def test_simple_different_seeds(self, engine):
        r1 = _parse(engine.random_assignment({
            "n": 50, "n_groups": 2, "method": "simple", "seed": 1,
        }))
        r2 = _parse(engine.random_assignment({
            "n": 50, "n_groups": 2, "method": "simple", "seed": 2,
        }))
        assert r1["assignments"] != r2["assignments"]


class TestRandomAssignmentBlock:
    """Block random assignment."""

    def test_block_balanced(self, engine):
        res = _parse(engine.random_assignment({
            "n": 24, "n_groups": 3, "method": "block",
            "block_size": 6, "seed": 42,
        }))
        assert len(res["assignments"]) == 24
        counts = res["group_sizes"]
        # Block randomisation should give perfectly balanced groups
        assert counts.get("0", 0) == 8
        assert counts.get("1", 0) == 8
        assert counts.get("2", 0) == 8

    def test_block_odd_n(self, engine):
        res = _parse(engine.random_assignment({
            "n": 25, "n_groups": 2, "method": "block",
            "block_size": 4, "seed": 7,
        }))
        assert len(res["assignments"]) == 25
        # One group may have +1 due to trimming
        total = sum(res["group_sizes"].values())
        assert total == 25

    def test_block_default_size(self, engine):
        res = _parse(engine.random_assignment({
            "n": 20, "n_groups": 4, "method": "block", "seed": 1,
        }))
        # Default block_size = n_groups = 4, so 5 blocks of 4
        counts = res["group_sizes"]
        for g in range(4):
            assert counts.get(str(g), 0) == 5


class TestRandomAssignmentStratified:
    """Stratified random assignment."""

    def test_stratified_basic(self, engine):
        res = _parse(engine.random_assignment({
            "n": 20, "n_groups": 2, "method": "stratified",
            "strata": [
                list(range(0, 10)),   # stratum 1: units 0-9
                list(range(10, 20)),  # stratum 2: units 10-19
            ],
            "seed": 42,
        }))
        assert len(res["assignments"]) == 20

        # Check each stratum is internally balanced
        s1_assigns = [res["assignments"][i] for i in range(10)]
        s2_assigns = [res["assignments"][i] for i in range(10, 20)]
        assert s1_assigns.count(0) == 5
        assert s1_assigns.count(1) == 5
        assert s2_assigns.count(0) == 5
        assert s2_assigns.count(1) == 5

    def test_stratified_three_groups(self, engine):
        res = _parse(engine.random_assignment({
            "n": 30, "n_groups": 3, "method": "stratified",
            "strata": [
                list(range(0, 12)),
                list(range(12, 30)),
            ],
            "seed": 7,
        }))
        assert len(res["assignments"]) == 30
        assert set(res["assignments"]).issubset({0, 1, 2})

    def test_stratified_missing_strata(self, engine):
        res = _parse(engine.random_assignment({
            "n": 10, "method": "stratified",
        }))
        assert "error" in res


class TestRandomAssignmentEdgeCases:
    """Edge cases for random assignment."""

    def test_one_group(self, engine):
        res = _parse(engine.random_assignment({
            "n": 10, "n_groups": 1, "method": "simple", "seed": 1,
        }))
        assert all(a == 0 for a in res["assignments"])

    def test_n_equals_n_groups(self, engine):
        res = _parse(engine.random_assignment({
            "n": 5, "n_groups": 5, "method": "simple", "seed": 42,
        }))
        assert sorted(res["assignments"]) == [0, 1, 2, 3, 4]

    def test_invalid_method(self, engine):
        res = _parse(engine.random_assignment({"method": "invalid"}))
        assert "error" in res
