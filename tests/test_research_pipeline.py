"""Tests for ExperimentPipeline: load, validate, transform, save, snapshot, seed."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from sophia.research.pipeline import ExperimentPipeline
from sophia.research.workspace_guard import WorkspaceGuard


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def pipeline(workspace):
    return ExperimentPipeline(workspace)


@pytest.fixture
def sample_csv(workspace):
    """Write a sample CSV into the workspace."""
    path = os.path.join(workspace, "sample.csv")
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 999],
        "score": [88.5, 92.3, 76.1, 95.0, 89.7],
    })
    df.to_csv(path, index=False)
    return "sample.csv"


@pytest.fixture
def sample_json(workspace):
    """Write a sample JSON into the workspace."""
    path = os.path.join(workspace, "sample.json")
    data = [
        {"x": 1, "y": 2},
        {"x": 3, "y": 4},
        {"x": 5, "y": 6},
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    return "sample.json"


class TestLoadData:
    def test_load_csv_auto_format(self, pipeline, sample_csv):
        result = json.loads(pipeline.load_data({"path": "sample.csv"}))
        assert result["shape"] == [5, 3]
        assert "name" in result["columns"]
        assert "age" in result["columns"]
        assert len(result["head"]) == 5

    def test_load_csv_explicit_format(self, pipeline, sample_csv):
        result = json.loads(pipeline.load_data({"path": "sample.csv", "format": "csv"}))
        assert result["shape"] == [5, 3]

    def test_load_json(self, pipeline, sample_json):
        result = json.loads(pipeline.load_data({"path": "sample.json", "format": "json"}))
        assert result["shape"] == [3, 2]
        assert result["head"][0]["x"] == 1

    def test_load_with_column_selection(self, pipeline, sample_csv):
        result = json.loads(pipeline.load_data({"path": "sample.csv", "columns": ["name", "age"]}))
        assert result["shape"] == [5, 2]
        assert result["columns"] == ["name", "age"]

    def test_load_nonexistent_file(self, pipeline):
        result = json.loads(pipeline.load_data({"path": "nope.csv"}))
        assert "error" in result

    def test_load_missing_columns(self, pipeline, sample_csv):
        result = json.loads(pipeline.load_data({"path": "sample.csv", "columns": ["name", "nonexistent"]}))
        assert "error" in result
        assert "nonexistent" in result["error"]


class TestValidateData:
    def test_validate_missing_values(self, pipeline):
        data = [
            {"a": 1, "b": 10},
            {"a": 2, "b": None},
            {"a": None, "b": 30},
        ]
        result = json.loads(pipeline.validate_data({"data": data, "checks": ["missing"]}))
        assert "missing" in result
        assert result["missing"]["a"]["count"] == 1
        assert result["missing"]["b"]["count"] == 1

    def test_validate_outliers(self, pipeline):
        data = [{"val": float(v)} for v in [1, 2, 3, 4, 5, 6, 7, 8, 100]]
        result = json.loads(pipeline.validate_data({"data": data, "checks": ["outliers"]}))
        assert "outliers" in result
        assert "val" in result["outliers"]
        assert result["outliers"]["val"]["count"] >= 1

    def test_validate_distribution(self, pipeline):
        rng = np.random.RandomState(42)
        data = [{"x": float(v)} for v in rng.normal(0, 1, 100)]
        result = json.loads(pipeline.validate_data({"data": data, "checks": ["distribution"]}))
        assert "distribution" in result
        assert "x" in result["distribution"]
        assert abs(result["distribution"]["x"]["mean"]) < 1.0

    def test_validate_types(self, pipeline):
        data = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
        result = json.loads(pipeline.validate_data({"data": data, "checks": ["types"]}))
        assert "types" in result
        assert result["types"]["name"]["inferred"] == "string"
        assert result["types"]["age"]["inferred"] == "integer"

    def test_validate_all_checks(self, pipeline):
        data = [{"a": float(v)} for v in [1, 2, 3, 4, 5]]
        result = json.loads(pipeline.validate_data({"data": data}))
        assert "missing" in result
        assert "outliers" in result
        assert "types" in result
        assert "distribution" in result
        assert "issues_found" in result

    def test_validate_no_data(self, pipeline):
        result = json.loads(pipeline.validate_data({"checks": ["missing"]}))
        assert "error" in result

    def test_validate_from_path(self, pipeline, sample_csv):
        result = json.loads(pipeline.validate_data({"path": "sample.csv", "checks": ["types"]}))
        assert "types" in result


class TestTransform:
    def test_standardize(self, pipeline):
        data = [{"x": 10, "y": 20}, {"x": 20, "y": 30}, {"x": 30, "y": 40}]
        result = json.loads(pipeline.transform({
            "data": data,
            "operations": [{"type": "standardize", "params": {"columns": ["x"]}}],
        }))
        assert result["operations_applied"] == 1
        # Mean of standardized x should be ~0
        vals = [row["x"] for row in result["head"]]
        assert abs(sum(vals) / len(vals)) < 0.01

    def test_normalize_minmax(self, pipeline):
        data = [{"x": 10}, {"x": 20}, {"x": 30}]
        result = json.loads(pipeline.transform({
            "data": data,
            "operations": [{"type": "normalize", "params": {"columns": ["x"], "method": "minmax"}}],
        }))
        assert result["operations_applied"] == 1
        # Min should be 0, max should be 1
        vals = [row["x"] for row in result["head"]]
        assert min(vals) == 0.0
        assert max(vals) == 1.0

    def test_encode_onehot(self, pipeline):
        data = [{"color": "red"}, {"color": "blue"}, {"color": "green"}]
        result = json.loads(pipeline.transform({
            "data": data,
            "operations": [{"type": "encode", "params": {"columns": ["color"], "method": "onehot"}}],
        }))
        assert result["operations_applied"] == 1
        # Original 'color' column should be gone, replaced by dummies
        assert "color" not in result["columns"]
        assert any("color_" in c for c in result["columns"])

    def test_impute_mean(self, pipeline):
        data = [{"x": 10}, {"x": None}, {"x": 30}]
        result = json.loads(pipeline.transform({
            "data": data,
            "operations": [{"type": "impute", "params": {"columns": ["x"], "strategy": "mean"}}],
        }))
        assert result["operations_applied"] == 1
        # The None should be filled with mean of 10 and 30 = 20
        vals = [row["x"] for row in result["head"]]
        assert 20.0 in vals

    def test_filter_greater(self, pipeline):
        data = [{"x": 5}, {"x": 15}, {"x": 25}]
        result = json.loads(pipeline.transform({
            "data": data,
            "operations": [{"type": "filter", "params": {"column": "x", "condition": ">", "value": 10}}],
        }))
        assert result["shape"] == [2, 1]

    def test_select_columns(self, pipeline):
        data = [{"a": 1, "b": 2, "c": 3}]
        result = json.loads(pipeline.transform({
            "data": data,
            "operations": [{"type": "select", "params": {"columns": ["a", "b"]}}],
        }))
        assert result["columns"] == ["a", "b"]

    def test_rename_columns(self, pipeline):
        data = [{"old_name": 42}]
        result = json.loads(pipeline.transform({
            "data": data,
            "operations": [{"type": "rename", "params": {"mapping": {"old_name": "new_name"}}}],
        }))
        assert "new_name" in result["columns"]
        assert "old_name" not in result["columns"]

    def test_chained_operations(self, pipeline):
        data = [{"x": 10, "y": "a"}, {"x": 20, "y": "b"}, {"x": 30, "y": "a"}]
        result = json.loads(pipeline.transform({
            "data": data,
            "operations": [
                {"type": "filter", "params": {"column": "x", "condition": ">", "value": 10}},
                {"type": "select", "params": {"columns": ["x"]}},
            ],
        }))
        assert result["shape"] == [2, 1]
        assert result["operations_applied"] == 2


class TestSaveResults:
    def test_save_json(self, pipeline):
        result = json.loads(pipeline.save_results({
            "data": {"key": "value", "num": 42},
            "path": "test_result.json",
            "format": "json",
        }))
        assert "path" in result
        assert result["format"] == "json"
        assert result["size_bytes"] > 0

    def test_save_csv(self, pipeline):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = json.loads(pipeline.save_results({
            "data": data,
            "path": "test_result.csv",
            "format": "csv",
        }))
        assert result["format"] == "csv"
        assert result["size_bytes"] > 0

    def test_save_creates_file(self, pipeline, workspace):
        pipeline.save_results({
            "data": {"hello": "world"},
            "path": "nested/deep/result.json",
            "format": "json",
        })
        cache_dir = os.path.join(workspace, ".research", "cache")
        assert os.path.exists(os.path.join(cache_dir, "nested", "deep", "result.json"))


class TestExportReport:
    def test_export_markdown(self, pipeline):
        result = json.loads(pipeline.export_report({
            "title": "Test Report",
            "sections": [
                {"heading": "Intro", "content": "This is the intro."},
                {"heading": "Results", "content": "Results here."},
            ],
            "path": "test_report.md",
            "format": "markdown",
        }))
        assert result["sections"] == 2
        assert result["size_bytes"] > 0

    def test_export_html(self, pipeline):
        result = json.loads(pipeline.export_report({
            "title": "HTML Report",
            "sections": [{"heading": "Data", "content": "Some data."}],
            "path": "test_report.html",
            "format": "html",
        }))
        assert result["format"] == "html"
        assert result["size_bytes"] > 0

    def test_export_markdown_content(self, pipeline, workspace):
        pipeline.export_report({
            "title": "Content Check",
            "sections": [{"heading": "Sec1", "content": "Hello world"}],
            "path": "content_check.md",
            "format": "markdown",
        })
        report_path = os.path.join(workspace, ".research", "reports", "content_check.md")
        with open(report_path) as f:
            content = f.read()
        assert "# Content Check" in content
        assert "## Sec1" in content
        assert "Hello world" in content


class TestSnapshot:
    def test_create_snapshot(self, pipeline, workspace, sample_csv):
        result = json.loads(pipeline.snapshot({
            "label": "test_snap",
            "data_paths": ["sample.csv"],
            "code": "print('hello')",
        }))
        assert "snapshot_id" in result
        assert result["label"] == "test_snap"
        assert "sample.csv" in result["data_hashes"]
        assert len(result["data_hashes"]["sample.csv"]) == 32  # MD5 hex

    def test_snapshot_missing_file(self, pipeline):
        result = json.loads(pipeline.snapshot({
            "label": "missing",
            "data_paths": ["nonexistent.csv"],
        }))
        assert "ERROR" in result["data_hashes"]["nonexistent.csv"]

    def test_snapshot_saves_metadata(self, pipeline, workspace, sample_csv):
        result = json.loads(pipeline.snapshot({
            "label": "meta_test",
            "data_paths": ["sample.csv"],
            "code": "x = 1",
        }))
        # Verify the metadata file was saved in cache
        snap_id = result["snapshot_id"]
        meta_path = os.path.join(workspace, ".research", "cache", f"{snap_id}.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["label"] == "meta_test"


class TestSeedManager:
    def test_set_seed(self, pipeline):
        result = json.loads(pipeline.seed_manager({"action": "set", "seed": 42}))
        assert result["seed"] == 42
        assert result["action"] == "set"

    def test_get_seed(self, pipeline):
        pipeline.seed_manager({"action": "set", "seed": 99})
        result = json.loads(pipeline.seed_manager({"action": "get"}))
        assert result["seed"] == 99

    def test_reset_seed(self, pipeline):
        pipeline.seed_manager({"action": "set", "seed": 42})
        result = json.loads(pipeline.seed_manager({"action": "reset"}))
        assert result["seed"] is None

    def test_set_seed_reproducibility(self, pipeline):
        pipeline.seed_manager({"action": "set", "seed": 123})
        a = np.random.rand(5).tolist()
        pipeline.seed_manager({"action": "set", "seed": 123})
        b = np.random.rand(5).tolist()
        assert a == b

    def test_set_seed_missing_value(self, pipeline):
        result = json.loads(pipeline.seed_manager({"action": "set"}))
        assert "error" in result
