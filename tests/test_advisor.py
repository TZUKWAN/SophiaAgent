"""Tests for MethodologyAdvisor."""
import json

import pytest

from sophia.research.advisor import MethodologyAdvisor


@pytest.fixture
def advisor():
    return MethodologyAdvisor()


def _parse(result: str) -> dict:
    return json.loads(result)


class TestMethodologyAdvisor:

    def test_did_recommended_for_panel_quasi(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "What is the effect of a new education policy on test scores?",
            "data_description": {"N": 5000, "type": "panel", "units": 100, "periods": 5, "variables": 10},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
        }))
        assert "recommended_methods" in res
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "did" in methods
        # did should be highly ranked for panel quasi-experimental
        top3 = methods[:3]
        assert "did" in top3

    def test_psm_recommended_for_cross_sectional(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Does job training improve earnings?",
            "data_description": {"N": 2000, "type": "cross-sectional", "variables": 15},
            "design": "observational",
            "outcome_type": "continuous",
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "psm" in methods

    def test_iv_recommended_when_instrument_mentioned(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Effect of schooling on wages using quarter of birth as instrument",
            "data_description": {"N": 10000, "variables": 8},
            "design": "observational",
            "outcome_type": "continuous",
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "iv" in methods

    def test_rdd_recommended_when_cutoff_mentioned(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Effect of scholarship on GPA using test score cutoff",
            "data_description": {"N": 3000, "variables": 5},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "rdd" in methods

    def test_nonparametric_recommended_for_small_sample(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Difference in anxiety scores between two therapy groups",
            "data_description": {"N": 20, "variables": 3},
            "design": "randomized",
            "outcome_type": "continuous",
            "constraints": ["small sample"],
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "nonparametric" in methods

    def test_meta_recommended_for_multiple_studies(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Meta-analysis of mindfulness interventions on stress reduction",
            "data_description": {"N": 0, "studies": 12},
            "design": "observational",
            "outcome_type": "continuous",
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "meta_random" in methods or "meta_fixed" in methods

    def test_thematic_recommended_for_text_data(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "What themes emerge from student interviews about online learning?",
            "data_description": {"N": 50, "type": "text"},
            "design": "observational",
            "outcome_type": "text",
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "thematic" in methods

    def test_cronbach_recommended_for_survey_scale(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Reliability of a 10-item job satisfaction scale",
            "data_description": {"N": 300, "variables": 10},
            "design": "survey",
            "outcome_type": "continuous",
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "cronbach" in methods

    def test_preflight_checks_present(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Effect of policy on employment",
            "data_description": {"N": 5000, "type": "panel", "units": 100, "periods": 5},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
            "constraints": ["no pre-treatment data"],
        }))
        assert "preflight_checks" in res
        checks = res["preflight_checks"]
        assert any(c["check"] == "panel_without_pretest" for c in checks)

    def test_decision_tree_trace_present(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Does X cause Y?",
            "data_description": {"N": 1000, "variables": 5},
            "design": "observational",
            "outcome_type": "continuous",
        }))
        assert "decision_tree_trace" in res
        assert len(res["decision_tree_trace"]) > 0
        assert "score" in res["decision_tree_trace"][0]

    def test_each_recommendation_has_required_fields(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Test scores after tutoring",
            "data_description": {"N": 200, "variables": 4},
            "design": "randomized",
            "outcome_type": "continuous",
        }))
        for rec in res["recommended_methods"]:
            assert "method_id" in rec
            assert "tool_name" in rec
            assert "rationale" in rec
            assert "preconditions" in rec
            assert "alternatives" in rec
            assert "rank" in rec
            assert "confidence" in rec
            assert isinstance(rec["confidence"], float)
            assert 0.0 <= rec["confidence"] <= 1.0

    def test_explicit_exclusion_by_constraint(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Effect of minimum wage on employment using panel data",
            "data_description": {"N": 5000, "type": "panel", "units": 100, "periods": 5},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
            "constraints": ["no did"],
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        # did should be excluded or have very low confidence
        if "did" in methods:
            did_rec = next(m for m in res["recommended_methods"] if m["method_id"] == "did")
            assert did_rec["confidence"] < 0.3

    def test_scm_recommended_for_single_treated_unit(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "Effect of a tax reform on GDP in one country",
            "data_description": {"N": 500, "type": "panel", "units": 20, "periods": 25},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "scm" in methods

    def test_sample_size_recommended_for_planning(self, advisor):
        res = _parse(advisor.advise({
            "research_question": "How many participants do I need for a survey on job satisfaction?",
            "data_description": {"N": 0},
            "design": "survey",
            "outcome_type": "continuous",
        }))
        methods = [m["method_id"] for m in res["recommended_methods"]]
        assert "sample_size" in methods
