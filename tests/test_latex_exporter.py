"""Tests for LaTeXReporter."""
import json
import os
import tempfile

import pytest

from sophia.research.latex_exporter import LaTeXReporter
from sophia.research.result_store import ResultStore


def _parse(result: str) -> dict:
    return json.loads(result)


@pytest.fixture
def tmp_store():
    with tempfile.TemporaryDirectory() as td:
        store = ResultStore(td)
        yield store
        store.close()


@pytest.fixture
def reporter(tmp_store):
    return LaTeXReporter(tmp_store)


class TestLaTeXExporter:

    def test_export_basic_apa7(self, reporter, tmp_store):
        # Store a dummy result
        rid = tmp_store.store({
            "test": "independent t-test",
            "t": 2.5,
            "p": 0.012,
            "df": 48,
            "cohen_d": 0.71,
            "apa": "An independent-samples t-test revealed a significant difference, t(48) = 2.50, p = .012, d = 0.71 (medium effect size).",
            "n": 50,
        }, kind="result", tool="research_ttest", params={"group1": [1, 2, 3]})

        res = _parse(reporter.export({
            "result_ids": [rid],
            "title": "Test Report",
            "authors": ["Alice Smith", "Bob Jones"],
            "template": "apa7",
            "output_name": "test_report",
        }))

        assert os.path.exists(res["path"])
        assert res["template"] == "apa7"
        assert res["n_results"] == 1
        with open(res["path"], "r", encoding="utf-8") as f:
            tex = f.read()
        assert "\\documentclass" in tex
        assert "Test Report" in tex
        assert "Alice Smith" in tex
        assert "t-test" in tex or "ttest" in tex

    def test_export_elsevier(self, reporter, tmp_store):
        rid = tmp_store.store({
            "alpha": 0.85,
            "n_items": 10,
            "n_responses": 200,
            "apa": "Cronbach's alpha for the 10-item scale was 0.850 (N = 200).",
        }, kind="result", tool="research_cronbach", params={})

        res = _parse(reporter.export({
            "result_ids": [rid],
            "title": "Reliability Study",
            "authors": ["Dr. X"],
            "template": "elsevier",
            "output_name": "rel_report",
        }))

        assert os.path.exists(res["path"])
        with open(res["path"], "r", encoding="utf-8") as f:
            tex = f.read()
        assert "elsarticle" in tex
        assert "Reliability Study" in tex

    def test_export_with_regression_table(self, reporter, tmp_store):
        rid = tmp_store.store({
            "test": "OLS regression",
            "coefficients": {"intercept": 1.2, "x1": 0.5, "x2": -0.3},
            "std_errors": {"intercept": 0.1, "x1": 0.08, "x2": 0.12},
            "t_stats": {"intercept": 12.0, "x1": 6.25, "x2": -2.5},
            "p_values": {"intercept": 0.001, "x1": 0.001, "x2": 0.015},
            "r_squared": 0.65,
            "apa": "Regression results: x1 was significant.",
        }, kind="result", tool="research_regression", params={})

        res = _parse(reporter.export({
            "result_ids": [rid],
            "title": "Regression Analysis",
            "authors": ["Author A"],
            "include_tables": True,
            "output_name": "reg_report",
        }))

        with open(res["path"], "r", encoding="utf-8") as f:
            tex = f.read()
        assert "tabular" in tex
        assert "x1" in tex
        assert "x2" in tex

    def test_export_missing_result_id_warns(self, reporter):
        res = _parse(reporter.export({
            "result_ids": ["res_nonexistent"],
            "title": "Bad Report",
            "authors": ["A"],
            "output_name": "bad",
        }))
        assert len(res["warnings"]) > 0
        assert "nonexistent" in res["warnings"][0]

    def test_export_multiple_results(self, reporter, tmp_store):
        r1 = tmp_store.store({"test": "t-test", "p": 0.02, "apa": "T-test significant.", "n": 30},
                             kind="result", tool="research_ttest", params={})
        r2 = tmp_store.store({"test": "anova", "p": 0.04, "apa": "ANOVA significant.", "n": 90},
                             kind="result", tool="research_anova", params={})

        res = _parse(reporter.export({
            "result_ids": [r1, r2],
            "title": "Multi-Result Report",
            "authors": ["A", "B"],
            "output_name": "multi",
        }))

        assert res["n_results"] == 2
        with open(res["path"], "r", encoding="utf-8") as f:
            tex = f.read()
        assert "T-test" in tex or "t-test" in tex
        assert "ANOVA" in tex or "anova" in tex
