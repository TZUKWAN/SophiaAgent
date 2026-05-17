"""Tests for VisualizationEngine: all plot types and error handling."""
import json
import os

import numpy as np
import pytest

from sophia.research.visualization import VisualizationEngine, HAS_MPL


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def engine(workspace):
    return VisualizationEngine(workspace)


# Skip all tests if matplotlib is not installed
pytestmark = pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")


class TestPlotGeneric:
    def test_hist_plot(self, engine):
        data = list(np.random.randn(100))
        result = json.loads(engine.plot({
            "data": data,
            "type": "hist",
            "title": "Histogram Test",
            "filename": "test_hist.png",
        }))
        assert "path" in result
        assert result["type"] == "hist"
        assert os.path.exists(result["path"])

    def test_box_plot(self, engine):
        data = [list(np.random.randn(50)), list(np.random.randn(50))]
        result = json.loads(engine.plot({
            "data": data,
            "type": "box",
            "labels": ["Group A", "Group B"],
            "filename": "test_box.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])

    def test_violin_plot(self, engine):
        data = [list(np.random.randn(30))]
        result = json.loads(engine.plot({
            "data": data,
            "type": "violin",
            "filename": "test_violin.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])

    def test_bar_plot(self, engine):
        result = json.loads(engine.plot({
            "data": [10, 20, 30],
            "type": "bar",
            "labels": ["A", "B", "C"],
            "filename": "test_bar.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])

    def test_scatter_plot(self, engine):
        result = json.loads(engine.plot({
            "data": [[1, 2, 3, 4], [10, 20, 25, 30]],
            "type": "scatter",
            "x_label": "X",
            "y_label": "Y",
            "filename": "test_scatter.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])

    def test_line_plot(self, engine):
        data = list(range(10))
        result = json.loads(engine.plot({
            "data": data,
            "type": "line",
            "filename": "test_line.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])

    def test_qq_plot(self, engine):
        data = list(np.random.randn(50))
        result = json.loads(engine.plot({
            "data": data,
            "type": "qq",
            "filename": "test_qq.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])

    def test_unknown_plot_type(self, engine):
        result = json.loads(engine.plot({
            "data": [1, 2, 3],
            "type": "unknown_type",
            "filename": "test_unknown.png",
        }))
        assert "error" in result


class TestForestPlot:
    def test_basic_forest(self, engine):
        result = json.loads(engine.forest_plot({
            "studies": ["Study A", "Study B", "Study C"],
            "effects": [0.5, 0.8, 1.2],
            "cis_low": [0.2, 0.5, 0.8],
            "cis_high": [0.8, 1.1, 1.6],
            "filename": "test_forest.png",
        }))
        assert "path" in result
        assert result["studies"] == 3
        assert os.path.exists(result["path"])

    def test_forest_with_pooled(self, engine):
        result = json.loads(engine.forest_plot({
            "studies": ["A", "B"],
            "effects": [0.5, 0.7],
            "cis_low": [0.3, 0.5],
            "cis_high": [0.7, 0.9],
            "pooled_effect": 0.6,
            "pooled_ci_low": 0.45,
            "pooled_ci_high": 0.75,
            "filename": "test_forest_pooled.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])


class TestFunnelPlot:
    def test_basic_funnel(self, engine):
        effects = [0.5, 0.8, 1.0, 0.6, 1.2, 0.9]
        se = [0.1, 0.15, 0.2, 0.12, 0.25, 0.18]
        result = json.loads(engine.funnel_plot({
            "effects": effects,
            "se": se,
            "filename": "test_funnel.png",
        }))
        assert "path" in result
        assert result["points"] == 6
        assert os.path.exists(result["path"])


class TestNetworkPlot:
    def test_basic_network(self, engine):
        result = json.loads(engine.network_plot({
            "nodes": ["A", "B", "C", "D"],
            "edges": [
                {"source": "A", "target": "B", "weight": 1.5},
                {"source": "B", "target": "C", "weight": 2.0},
                {"source": "C", "target": "D", "weight": 1.0},
                {"source": "A", "target": "D", "weight": 0.5},
            ],
            "layout": "spring",
            "filename": "test_network.png",
        }))
        assert "path" in result
        assert result["nodes"] == 4
        assert result["edges"] == 4
        assert os.path.exists(result["path"])

    def test_circular_layout(self, engine):
        result = json.loads(engine.network_plot({
            "nodes": ["X", "Y", "Z"],
            "edges": [
                {"source": "X", "target": "Y", "weight": 1},
                {"source": "Y", "target": "Z", "weight": 2},
            ],
            "layout": "circular",
            "filename": "test_network_circ.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])

    def test_empty_nodes(self, engine):
        result = json.loads(engine.network_plot({
            "nodes": [],
            "edges": [],
            "filename": "test_network_empty.png",
        }))
        assert "error" in result


class TestHeatmap:
    def test_basic_heatmap(self, engine):
        matrix = [[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]]
        result = json.loads(engine.heatmap({
            "matrix": matrix,
            "x_labels": ["A", "B", "C"],
            "y_labels": ["A", "B", "C"],
            "filename": "test_heatmap.png",
        }))
        assert "path" in result
        assert result["shape"] == [3, 3]
        assert os.path.exists(result["path"])


class TestRocCurve:
    def test_basic_roc(self, engine):
        rng = np.random.RandomState(42)
        y_true = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
        y_score = [0.1, 0.4, 0.8, 0.9, 0.3, 0.7, 0.85, 0.2, 0.95, 0.35]
        result = json.loads(engine.roc_curve({
            "y_true": y_true,
            "y_score": y_score,
            "auc": 0.95,
            "filename": "test_roc.png",
        }))
        assert "path" in result
        assert result["auc"] == 0.95
        assert os.path.exists(result["path"])


class TestConfusionMatrix:
    def test_basic_confusion(self, engine):
        matrix = [[50, 10], [5, 35]]
        result = json.loads(engine.confusion_matrix({
            "matrix": matrix,
            "labels": ["Pred 0", "Pred 1"],
            "filename": "test_confusion.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])


class TestDidPlot:
    def test_basic_did(self, engine):
        result = json.loads(engine.did_plot({
            "time": [1, 2, 3, 4, 5, 6],
            "treatment": [10, 12, 14, 25, 28, 30],
            "control": [10, 11, 13, 14, 15, 16],
            "intervention_time": 3.5,
            "filename": "test_did.png",
        }))
        assert "path" in result
        assert os.path.exists(result["path"])


class TestExperimentDashboard:
    def test_basic_dashboard(self, engine):
        result = json.loads(engine.experiment_dashboard({
            "runs": [
                {"name": "Run 1", "metrics": {"accuracy": 0.85, "loss": 0.35}},
                {"name": "Run 2", "metrics": {"accuracy": 0.92, "loss": 0.20}},
                {"name": "Run 3", "metrics": {"accuracy": 0.88, "loss": 0.28}},
            ],
            "filename": "test_dashboard.png",
        }))
        assert "path" in result
        assert result["runs"] == 3
        assert result["metrics"] == 2
        assert os.path.exists(result["path"])

    def test_empty_runs(self, engine):
        result = json.loads(engine.experiment_dashboard({
            "runs": [],
            "filename": "test_empty_dashboard.png",
        }))
        assert "error" in result


class TestEffectSizePlot:
    def test_basic_effect_size(self, engine):
        result = json.loads(engine.effect_size_plot({
            "effects": [
                {"name": "Treatment A", "value": 0.5, "ci_low": 0.2, "ci_high": 0.8},
                {"name": "Treatment B", "value": 0.8, "ci_low": 0.4, "ci_high": 1.2},
                {"name": "Treatment C", "value": -0.1, "ci_low": -0.4, "ci_high": 0.2},
            ],
            "filename": "test_effect_size.png",
        }))
        assert "path" in result
        assert result["effects"] == 3
        assert os.path.exists(result["path"])

    def test_empty_effects(self, engine):
        result = json.loads(engine.effect_size_plot({
            "effects": [],
            "filename": "test_empty_effects.png",
        }))
        assert "error" in result


class TestOutputPath:
    def test_all_plots_save_to_figures_dir(self, engine, workspace):
        figures_dir = os.path.join(workspace, ".research", "figures")
        engine.plot({"data": [1, 2, 3], "type": "hist", "filename": "path_test.png"})
        assert os.path.exists(os.path.join(figures_dir, "path_test.png"))

    def test_figure_is_valid_image(self, engine, workspace):
        engine.plot({"data": list(range(20)), "type": "hist", "filename": "valid_img.png"})
        path = os.path.join(workspace, ".research", "figures", "valid_img.png")
        assert os.path.getsize(path) > 0
        # PNG files start with the 8-byte signature
        with open(path, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"
