"""End-to-end integration tests for SophiaAgent research workflows.

These tests exercise the full data flow: load data -> engine analysis ->
ResultStore persistence -> LaTeX export, verifying that components integrate
correctly without LLM involvement.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from sophia.config import Config
from sophia.research.causal import CausalEngine
from sophia.research.qualitative import QualitativeEngine
from sophia.research.advisor import MethodologyAdvisor
from sophia.research.latex_exporter import LaTeXReporter
from sophia.research.result_store import ResultStore
from sophia.research.seed import GlobalSeed


@pytest.fixture
def tmp_workspace():
    with tempfile.TemporaryDirectory() as td:
        yield td


@pytest.fixture
def store(tmp_workspace):
    s = ResultStore(tmp_workspace)
    yield s
    s.close()


@pytest.fixture
def config(tmp_workspace):
    cfg = Config()
    cfg.session.workspace = tmp_workspace
    cfg.session.db_path = os.path.join(tmp_workspace, "sessions.db")
    return cfg


class TestEndToEndDiDWorkflow:
    """E2E: DiD analysis -> ResultStore -> LaTeX export."""

    def test_did_full_pipeline(self, store, tmp_workspace):
        engine = CausalEngine(store=store)

        # Generate panel data with enough periods for full diagnostics
        np.random.seed(42)
        n_units = 40
        n_periods = 8
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(n_periods), n_units)
        treat = (units >= 20).astype(float)
        post = (times >= 4).astype(float)
        y = (20.0
             + np.random.normal(0, 2, n_units)[units]
             + 0.3 * times
             + 2.5 * treat * post
             + np.random.normal(0, 1, len(units)))
        df = pd.DataFrame({
            "entity": units,
            "time": times,
            "treated": treat,
            "post": post,
            "employment": y,
        })
        rid_df = store.store(df, kind="dataframe", tool="research_load_data", params={"path": "synthetic_panel.csv"})

        # Run DiD
        result = json.loads(engine.did({
            "y": df["employment"].tolist(),
            "treat": df["treated"].tolist(),
            "post": df["post"].tolist(),
            "unit": df["entity"].tolist(),
            "time": df["time"].tolist(),
            "event_study": True,
            "placebo": True,
            "n_placebo": 100,
            "se_comparison": True,
        }))

        assert "error" not in result, result.get("error")
        assert "did_estimate" in result
        assert "apa_report" in result
        assert "parallel_trends_test" in result
        assert "event_study" in result
        assert "placebo" in result
        assert "se_comparison" in result

        # Store result
        rid_did = store.store(result, kind="result", tool="research_did",
                              params={"event_study": True, "placebo": True},
                              parents=[rid_df])

        # Verify lineage
        lineage = store.lineage(rid_did)
        assert len(lineage) >= 2
        tools = [meta["tool"] for meta in lineage]
        assert "research_did" in tools
        assert "research_load_data" in tools

        # Export LaTeX
        reporter = LaTeXReporter(store)
        export = json.loads(reporter.export({
            "result_ids": [rid_did],
            "title": "Minimum Wage and Employment: A DiD Analysis",
            "authors": ["Researcher A"],
            "template": "apa7",
            "include_tables": True,
            "output_name": "did_report",
        }))

        assert os.path.exists(export["path"])
        assert export["template"] == "apa7"
        with open(export["path"], "r", encoding="utf-8") as f:
            tex = f.read()
        assert "\\documentclass" in tex
        assert "Minimum Wage" in tex
        assert "DiD" in tex or "did" in tex.lower()

    def test_did_classic_without_panel(self, store):
        engine = CausalEngine(store=store)
        np.random.seed(42)
        n = 200
        treat = np.repeat([0, 1], n // 2)
        post = np.tile([0, 1], n // 2)
        y = 10 + 2 * treat + 1 * post + 3 * (treat * post) + np.random.normal(0, 1, n)

        result = json.loads(engine.did({
            "y": y.tolist(),
            "treat": treat.tolist(),
            "post": post.tolist(),
        }))

        assert "error" not in result
        assert "did_estimate" in result
        # True effect is 3.0; should be close
        assert 1.5 < result["did_estimate"] < 4.5


class TestEndToEndSCMWorkflow:
    """E2E: Synthetic Control Method workflow."""

    def test_scm_basic(self, store):
        engine = CausalEngine(store=store)

        df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "fixtures", "basque_country.csv")
        )

        result = json.loads(engine.synthetic_control({
            "y": df["gdp_per_capita"].tolist(),
            "unit": df["region"].tolist(),
            "time": df["year"].tolist(),
            "treated_unit": "Basque",
            "treatment_time": 1975,
            "placebo": True,
        }))

        assert "error" not in result, result.get("error")
        assert "weights" in result
        assert "predictor_balance" in result
        assert "average_treatment_effect" in result
        assert "rmspe_pre" in result
        assert "rmspe_post" in result
        assert "placebo" in result
        assert "apa_report" in result

        # Weights should sum to ~1 and be non-negative
        weights = result["weights"]
        assert len(weights) >= 1
        assert abs(sum(weights.values()) - 1.0) < 0.05
        assert all(w >= 0 for w in weights.values())

        # Basque should show negative effect post-1975 (terrorism impact)
        assert result["average_treatment_effect"] < 0

    def test_scm_store_lineage(self, store):
        engine = CausalEngine(store=store)
        df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "fixtures", "basque_country.csv")
        )
        rid_df = store.store(df, kind="dataframe", tool="research_load_data", params={})

        result = json.loads(engine.synthetic_control({
            "y": df["gdp_per_capita"].tolist(),
            "unit": df["region"].tolist(),
            "time": df["year"].tolist(),
            "treated_unit": "Basque",
            "treatment_time": 1975,
        }))
        rid_scm = store.store(result, kind="result", tool="research_scm",
                              params={}, parents=[rid_df])

        lineage = store.lineage(rid_scm)
        assert any(meta["tool"] == "research_load_data" for meta in lineage)
        assert any(meta["tool"] == "research_scm" for meta in lineage)


class TestEndToEndQualitativeWorkflow:
    """E2E: Qualitative analysis with keyword fallback (no LLM)."""

    def test_thematic_keyword_fallback(self, store):
        """Keyword-based thematic analysis works end-to-end."""
        with open(
            os.path.join(os.path.dirname(__file__), "fixtures", "interview_transcripts.json"),
            "r", encoding="utf-8"
        ) as f:
            texts = json.load(f)

        engine = QualitativeEngine(store=store)
        result = json.loads(engine.thematic({
            "texts": texts,
            "approach": "inductive",
            "language": "zh",
        }))

        assert "error" not in result, result.get("error")
        assert "themes" in result
        assert len(result["themes"]) > 0
        assert "apa" in result

    def test_thematic_store_roundtrip(self, store):
        with open(
            os.path.join(os.path.dirname(__file__), "fixtures", "interview_transcripts.json"),
            "r", encoding="utf-8"
        ) as f:
            texts = json.load(f)

        engine = QualitativeEngine(store=store)
        result = json.loads(engine.thematic({
            "texts": texts,
            "approach": "inductive",
            "language": "zh",
        }))
        rid = store.store(result, kind="result", tool="research_thematic", params={})

        loaded = store.get(rid)
        assert loaded["themes"] == result["themes"]


class TestEndToEndAdvisorWorkflow:
    """E2E: MethodologyAdvisor recommendations."""

    def test_advisor_recommends_did_for_panel(self):
        advisor = MethodologyAdvisor()
        result = json.loads(advisor.advise({
            "research_question": "Does minimum wage policy affect employment rates?",
            "data_description": {"N": 5000, "type": "panel", "units": 100, "periods": 5},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
        }))

        assert "error" not in result, result.get("error")
        methods = result["recommended_methods"]
        assert len(methods) > 0
        tool_names = [m["tool_name"] for m in methods]
        assert "research_did" in tool_names

        # DiD should be highly ranked
        did_rank = next(m["rank"] for m in methods if m["tool_name"] == "research_did")
        assert did_rank <= 3

    def test_advisor_preflight_checks(self):
        advisor = MethodologyAdvisor()
        result = json.loads(advisor.advise({
            "research_question": "What is the effect of a training program on test scores?",
            "data_description": {"N": 5000, "type": "panel", "units": 100, "periods": 5},
            "design": "quasi-experimental",
            "outcome_type": "continuous",
            "constraints": ["no pre-treatment data"],
        }))

        checks = result.get("preflight_checks", [])
        assert len(checks) > 0
        # With no pre-treatment data constraint, DiD should have a warning
        did_method = next((m for m in result["recommended_methods"] if m["tool_name"] == "research_did"), None)
        if did_method:
            assert any("parallel" in c.lower() or "pre" in c.lower() for c in did_method.get("preflight_checks", []))


class TestEndToEndReproducibility:
    """E2E: Same seed -> same results."""

    def test_seed_reproducibility_did(self, store):
        GlobalSeed.set(42)
        engine1 = CausalEngine(store=store)

        np.random.seed(42)
        n = 200
        treat = np.repeat([0, 1], n // 2)
        post = np.tile([0, 1], n // 2)
        y = 10 + 2 * treat + 1 * post + 3 * (treat * post) + np.random.normal(0, 1, n)

        result1 = json.loads(engine1.did({
            "y": y.tolist(),
            "treat": treat.tolist(),
            "post": post.tolist(),
        }))

        # Reset seed and re-run
        GlobalSeed.set(42)
        np.random.seed(42)
        y2 = 10 + 2 * treat + 1 * post + 3 * (treat * post) + np.random.normal(0, 1, n)

        engine2 = CausalEngine(store=store)
        result2 = json.loads(engine2.did({
            "y": y2.tolist(),
            "treat": treat.tolist(),
            "post": post.tolist(),
        }))

        # Data is identical, so estimates should be identical
        assert result1["did_estimate"] == result2["did_estimate"]
        assert result1["p_value"] == result2["p_value"]

    def test_seed_reproducibility_statistical(self, store):
        from sophia.research.statistics import StatisticalEngine
        GlobalSeed.set(123)
        engine1 = StatisticalEngine(store=store)
        data = np.random.normal(0, 1, 100).tolist()
        result1 = json.loads(engine1.ttest({
            "group1": data[:50],
            "group2": data[50:],
        }))

        GlobalSeed.set(123)
        np.random.seed(123)
        data2 = np.random.normal(0, 1, 100).tolist()
        engine2 = StatisticalEngine(store=store)
        result2 = json.loads(engine2.ttest({
            "group1": data2[:50],
            "group2": data2[50:],
        }))

        assert result1["t"] == result2["t"]
        assert result1["p"] == result2["p"]


class TestEndToEndResultStore:
    """E2E: ResultStore data flow integrity."""

    def test_dataframe_roundtrip(self, store):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        rid = store.store(df, kind="dataframe", tool="research_test", params={"x": 1})
        loaded = store.get_dataframe(rid)
        pd.testing.assert_frame_equal(loaded, df)

    def test_large_dataframe_pickle(self, store):
        df = pd.DataFrame(np.random.randn(10000, 50))
        rid = store.store(df, kind="dataframe", tool="research_test", params={})
        loaded = store.get(rid)
        assert loaded.shape == (10000, 50)

    def test_dict_roundtrip(self, store):
        payload = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        rid = store.store(payload, kind="result", tool="research_test", params={})
        loaded = store.get(rid)
        assert loaded == payload

    def test_lineage_three_levels(self, store):
        rid1 = store.store({"step": 1}, kind="result", tool="step1", params={})
        rid2 = store.store({"step": 2}, kind="result", tool="step2", params={}, parents=[rid1])
        rid3 = store.store({"step": 3}, kind="result", tool="step3", params={}, parents=[rid2])

        lineage = store.lineage(rid3)
        assert len(lineage) == 3
        depths = [m["depth"] for m in lineage]
        assert 0 in depths
        assert 1 in depths
        assert 2 in depths

    def test_metadata_preserved(self, store):
        rid = store.store({"x": 1}, kind="result", tool="research_demo",
                          params={"alpha": 0.05, "method": "ols"})
        meta = store.get_metadata(rid)
        assert meta["tool"] == "research_demo"
        assert meta["params"]["alpha"] == 0.05
        assert meta["params"]["method"] == "ols"
