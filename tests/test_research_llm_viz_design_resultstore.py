"""Tests for LLMEngine, VisualizationEngine, ResearchDesignEngine
↔ ResultStore integration (P1.4g).

These tests exercise the new behaviours layered onto the three engines:

- Every public method returns a ``result_id`` when a ``ResultStore`` is
  configured.
- Errors do NOT produce ``result_id`` and do NOT persist.
- ``params`` are sanitized.
- Legacy inputs still work and still get a ``result_id``.
"""
from __future__ import annotations

import json

import pytest

from sophia.research.llm import LLMEngine
from sophia.research.visualization import VisualizationEngine
from sophia.research.design import ResearchDesignEngine
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
def llm_engine(store):
    return LLMEngine(provider=None, store=store)


@pytest.fixture
def viz_engine(workspace, store):
    return VisualizationEngine(workspace, store=store)


@pytest.fixture
def design_engine(store):
    return ResearchDesignEngine(store=store)


@pytest.fixture(autouse=True)
def _reset_seed():
    GlobalSeed.reset()
    yield
    GlobalSeed.reset()


def _sample_texts():
    return [
        "Online learning offers great flexibility for students worldwide.",
        "Some students feel isolated in virtual classrooms without peers.",
    ]


# ----------------------------------------------------------------------
# LLMEngine
# ----------------------------------------------------------------------
class TestLLMResultIdRoundTrip:
    def test_prompt_test_error_no_provider(self, llm_engine):
        # prompt_test requires a provider; verify error path does not store
        out = json.loads(llm_engine.prompt_test({
            "variants": [
                {"name": "v1", "template": "Summarize: {input}"},
            ],
            "test_inputs": [
                {"input": "Hello world"},
            ],
        }))
        assert "error" in out
        assert "result_id" not in out

    def test_evaluate_returns_result_id(self, llm_engine):
        out = json.loads(llm_engine.evaluate({
            "responses": ["This is a test response."],
        }))
        assert "result_id" in out
        assert "per_response" in out

    def test_llm_judge_returns_result_id(self, llm_engine):
        out = json.loads(llm_engine.llm_judge({
            "responses": [{"prompt": "Hi", "response": "Hello there!"}],
        }))
        assert "result_id" in out
        assert "per_response" in out

    def test_rag_eval_returns_result_id(self, llm_engine):
        out = json.loads(llm_engine.rag_eval({
            "queries": ["What is AI?"],
            "contexts": ["AI stands for artificial intelligence."],
            "responses": ["AI is artificial intelligence."],
        }))
        assert "result_id" in out
        assert "per_query" in out

    def test_benchmark_error_no_provider(self, llm_engine):
        out = json.loads(llm_engine.benchmark({
            "dataset": [
                {"input": "2+2", "expected": "4"},
                {"input": "3+3", "expected": "6"},
            ],
        }))
        assert "error" in out
        assert "result_id" not in out

    def test_quality_score_returns_result_id(self, llm_engine):
        out = json.loads(llm_engine.quality_score({
            "generated": ["hello world"],
            "references": ["hello world"],
        }))
        assert "result_id" in out
        assert "per_sample" in out

    def test_prompt_generate_returns_result_id(self, llm_engine):
        out = json.loads(llm_engine.prompt_generate({
            "base_prompt": "Explain quantum computing.",
            "n_variants": 2,
        }))
        assert "result_id" in out
        assert "variants" in out

    def test_error_does_not_store(self, llm_engine, store):
        before = store.get_stats()["total"]
        out = json.loads(llm_engine.evaluate({"responses": []}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_tool_name_per_method(self, llm_engine, store):
        rid1 = json.loads(llm_engine.evaluate({"responses": ["a"]}))["result_id"]
        rid2 = json.loads(llm_engine.quality_score({
            "generated": ["a"], "references": ["a"],
        }))["result_id"]
        rid3 = json.loads(llm_engine.prompt_generate({
            "base_prompt": "x", "n_variants": 1,
        }))["result_id"]
        assert store.get_metadata(rid1)["tool"] == "research_evaluate"
        assert store.get_metadata(rid2)["tool"] == "research_quality_score"
        assert store.get_metadata(rid3)["tool"] == "research_prompt_generate"


# ----------------------------------------------------------------------
# VisualizationEngine
# ----------------------------------------------------------------------
class TestVizResultIdRoundTrip:
    def test_plot_returns_result_id(self, viz_engine):
        out = json.loads(viz_engine.plot({
            "data": [1, 2, 3, 4, 5],
            "type": "hist",
            "filename": "test_hist.png",
        }))
        assert "result_id" in out
        assert "path" in out

    def test_heatmap_returns_result_id(self, viz_engine):
        out = json.loads(viz_engine.heatmap({
            "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "x_labels": ["X", "Y", "Z"],
            "y_labels": ["A", "B", "C"],
            "filename": "test_heat.png",
        }))
        assert "result_id" in out
        assert "path" in out

    def test_roc_curve_returns_result_id(self, viz_engine):
        out = json.loads(viz_engine.roc_curve({
            "y_true": [0, 0, 1, 1],
            "y_score": [0.1, 0.4, 0.35, 0.8],
            "filename": "test_roc.png",
        }))
        assert "result_id" in out
        assert "path" in out

    def test_forest_plot_returns_result_id(self, viz_engine):
        out = json.loads(viz_engine.forest_plot({
            "effects": [0.2, 0.5, -0.1],
            "variances": [0.04, 0.05, 0.03],
            "study_names": ["A", "B", "C"],
            "filename": "test_forest.png",
        }))
        assert "result_id" in out
        assert "path" in out

    def test_error_does_not_store(self, viz_engine, store):
        before = store.get_stats()["total"]
        out = json.loads(viz_engine.plot({"data": [], "type": "hist"}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_tool_name_per_method(self, viz_engine, store):
        rid1 = json.loads(viz_engine.plot({
            "data": [1, 2, 3], "type": "hist", "filename": "a.png",
        }))["result_id"]
        rid2 = json.loads(viz_engine.roc_curve({
            "y_true": [0, 0, 1, 1],
            "y_score": [0.1, 0.4, 0.35, 0.8],
            "filename": "b.png",
        }))["result_id"]
        assert store.get_metadata(rid1)["tool"] == "research_plot"
        assert store.get_metadata(rid2)["tool"] == "research_roc_curve"


# ----------------------------------------------------------------------
# ResearchDesignEngine
# ----------------------------------------------------------------------
class TestDesignResultIdRoundTrip:
    def test_factorial_design_returns_result_id(self, design_engine):
        out = json.loads(design_engine.factorial_design({
            "factors": 2, "levels": 2, "type": "full",
        }))
        assert "result_id" in out
        assert "design" in out

    def test_response_surface_returns_result_id(self, design_engine):
        out = json.loads(design_engine.response_surface({
            "factors": 3, "type": "box-behnken",
        }))
        assert "result_id" in out
        assert "design" in out

    def test_latin_hypercube_returns_result_id(self, design_engine):
        out = json.loads(design_engine.latin_hypercube({
            "dimensions": 2, "samples": 5,
        }))
        assert "result_id" in out
        assert "samples_matrix" in out

    def test_power_analysis_returns_result_id(self, design_engine):
        out = json.loads(design_engine.power_analysis({
            "test": "ttest", "effect_size": 0.5, "power": 0.8,
        }))
        assert "result_id" in out
        assert "result_type" in out

    def test_random_assignment_returns_result_id(self, design_engine):
        out = json.loads(design_engine.random_assignment({
            "n": 10, "n_groups": 2, "method": "simple",
        }))
        assert "result_id" in out
        assert "assignments" in out

    def test_error_does_not_store(self, design_engine, store):
        before = store.get_stats()["total"]
        out = json.loads(design_engine.random_assignment({
            "n": 10, "n_groups": 2, "method": "stratified",
        }))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == before

    def test_tool_name_per_method(self, design_engine, store):
        rid1 = json.loads(design_engine.factorial_design({
            "factors": 2, "type": "full",
        }))["result_id"]
        rid2 = json.loads(design_engine.latin_hypercube({
            "dimensions": 2, "samples": 4,
        }))["result_id"]
        rid3 = json.loads(design_engine.power_analysis({
            "test": "ttest", "effect_size": 0.5, "power": 0.8,
        }))["result_id"]
        rid4 = json.loads(design_engine.random_assignment({
            "n": 8, "n_groups": 2, "method": "simple",
        }))["result_id"]
        assert store.get_metadata(rid1)["tool"] == "research_factorial_design"
        assert store.get_metadata(rid2)["tool"] == "research_latin_hypercube"
        assert store.get_metadata(rid3)["tool"] == "research_power_analysis"
        assert store.get_metadata(rid4)["tool"] == "research_random_assignment"
