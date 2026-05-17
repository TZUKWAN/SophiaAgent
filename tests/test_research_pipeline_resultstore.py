"""Tests for ExperimentPipeline ↔ ResultStore integration (P1.3).

These tests exercise the new behaviours layered onto the pipeline in P1.3:

- ``load_data`` returns a ``result_id`` and persists the DataFrame.
- ``validate_data`` accepts a ``result_id`` and stores the report with
  lineage back to the source frame.
- ``transform`` accepts a ``result_id`` and stores the transformed
  DataFrame with the source as parent.
- ``save_results`` accepts a ``result_id`` and flushes the stored content.
- ``snapshot`` captures full lineage for any provided ``result_ids``.
- ``seed_manager`` delegates to :class:`GlobalSeed`.

The legacy ``test_research_pipeline.py`` tests still cover the path where no
``ResultStore`` is configured.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import pytest

from sophia.research.pipeline import ExperimentPipeline
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
def pipeline(workspace, store):
    return ExperimentPipeline(workspace, store=store)


@pytest.fixture
def sample_csv(workspace):
    """Write a sample CSV into the workspace and return its relative path."""
    path = os.path.join(workspace, "sample.csv")
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 40],
        "score": [88.5, 92.3, 76.1, 95.0, 89.7],
    })
    df.to_csv(path, index=False)
    return "sample.csv"


@pytest.fixture(autouse=True)
def _reset_seed():
    GlobalSeed.reset()
    yield
    GlobalSeed.reset()


# ----------------------------------------------------------------------
# load_data ↔ ResultStore
# ----------------------------------------------------------------------
class TestLoadDataResultStore:
    def test_returns_result_id(self, pipeline, sample_csv):
        out = json.loads(pipeline.load_data({"path": sample_csv}))
        assert "result_id" in out
        assert out["result_id"].startswith("res_")

    def test_result_id_round_trip(self, pipeline, store, sample_csv):
        out = json.loads(pipeline.load_data({"path": sample_csv}))
        df = store.get_dataframe(out["result_id"])
        assert list(df.columns) == ["name", "age", "score"]
        assert len(df) == 5

    def test_stored_kind_is_dataframe(self, pipeline, store, sample_csv):
        out = json.loads(pipeline.load_data({"path": sample_csv}))
        meta = store.get_metadata(out["result_id"])
        assert meta["kind"] == "dataframe"
        assert meta["tool"] == "research_load_data"

    def test_params_recorded(self, pipeline, store, sample_csv):
        out = json.loads(pipeline.load_data(
            {"path": sample_csv, "columns": ["name", "age"]}
        ))
        meta = store.get_metadata(out["result_id"])
        assert meta["params"]["path"] == sample_csv
        assert meta["params"]["columns"] == ["name", "age"]

    def test_no_result_id_when_no_store(self, workspace, sample_csv):
        plain = ExperimentPipeline(workspace, store=None)
        out = json.loads(plain.load_data({"path": sample_csv}))
        assert "result_id" not in out
        # Original shape data still present
        assert out["shape"] == [5, 3]

    def test_load_error_does_not_store(self, pipeline, store):
        out = json.loads(pipeline.load_data({"path": "nope.csv"}))
        assert "error" in out
        assert "result_id" not in out
        assert store.get_stats()["total"] == 0


# ----------------------------------------------------------------------
# validate_data ↔ ResultStore
# ----------------------------------------------------------------------
class TestValidateDataResultStore:
    def test_accepts_result_id(self, pipeline, store, sample_csv):
        loaded = json.loads(pipeline.load_data({"path": sample_csv}))
        rid = loaded["result_id"]
        out = json.loads(pipeline.validate_data(
            {"result_id": rid, "checks": ["types"]}
        ))
        assert "types" in out
        assert "name" in out["types"]

    def test_records_lineage_to_source(self, pipeline, store, sample_csv):
        loaded = json.loads(pipeline.load_data({"path": sample_csv}))
        src_rid = loaded["result_id"]
        report = json.loads(pipeline.validate_data(
            {"result_id": src_rid, "checks": ["missing"]}
        ))
        report_rid = report["result_id"]
        meta = store.get_metadata(report_rid)
        assert src_rid in meta["parents"]
        assert meta["kind"] == "result"
        assert meta["tool"] == "research_validate_data"

    def test_legacy_data_path_still_works(self, pipeline):
        data = [{"a": 1, "b": None}, {"a": 2, "b": 4}]
        out = json.loads(pipeline.validate_data(
            {"data": data, "checks": ["missing"]}
        ))
        assert out["missing"]["b"]["count"] == 1


# ----------------------------------------------------------------------
# transform ↔ ResultStore
# ----------------------------------------------------------------------
class TestTransformResultStore:
    def test_accepts_result_id(self, pipeline, store, sample_csv):
        loaded = json.loads(pipeline.load_data({"path": sample_csv}))
        rid = loaded["result_id"]
        out = json.loads(pipeline.transform({
            "result_id": rid,
            "operations": [{"type": "select", "params": {"columns": ["age", "score"]}}],
        }))
        assert out["columns"] == ["age", "score"]
        assert "result_id" in out

    def test_transformed_dataframe_stored(self, pipeline, store, sample_csv):
        loaded = json.loads(pipeline.load_data({"path": sample_csv}))
        src = loaded["result_id"]
        out = json.loads(pipeline.transform({
            "result_id": src,
            "operations": [{"type": "select", "params": {"columns": ["age"]}}],
        }))
        new_rid = out["result_id"]
        df = store.get_dataframe(new_rid)
        assert list(df.columns) == ["age"]
        # Source is recorded as parent
        meta = store.get_metadata(new_rid)
        assert src in meta["parents"]

    def test_transform_chain_lineage(self, pipeline, store, sample_csv):
        loaded = json.loads(pipeline.load_data({"path": sample_csv}))
        r1 = loaded["result_id"]
        step1 = json.loads(pipeline.transform({
            "result_id": r1,
            "operations": [{"type": "select", "params": {"columns": ["age", "score"]}}],
        }))
        r2 = step1["result_id"]
        step2 = json.loads(pipeline.transform({
            "result_id": r2,
            "operations": [{"type": "standardize", "params": {"columns": ["age"]}}],
        }))
        r3 = step2["result_id"]
        lineage = store.lineage(r3)
        ids = [item["id"] for item in lineage]
        assert ids[0] == r3
        assert r2 in ids
        assert r1 in ids


# ----------------------------------------------------------------------
# save_results ↔ ResultStore
# ----------------------------------------------------------------------
class TestSaveResultsResultStore:
    def test_save_from_result_id_json(self, pipeline, store, sample_csv, workspace):
        loaded = json.loads(pipeline.load_data({"path": sample_csv}))
        rid = loaded["result_id"]
        out = json.loads(pipeline.save_results({
            "result_id": rid,
            "path": "stored.json",
            "format": "json",
        }))
        assert "path" in out
        assert os.path.exists(out["path"])
        with open(out["path"], "r", encoding="utf-8") as f:
            data = json.load(f)
        # Loaded DataFrame should round-trip as list-of-dicts
        assert isinstance(data, list)
        assert data[0]["name"] == "Alice"

    def test_save_from_result_id_csv(self, pipeline, store, sample_csv):
        loaded = json.loads(pipeline.load_data({"path": sample_csv}))
        rid = loaded["result_id"]
        out = json.loads(pipeline.save_results({
            "result_id": rid,
            "path": "stored.csv",
            "format": "csv",
        }))
        assert os.path.exists(out["path"])
        df = pd.read_csv(out["path"])
        assert len(df) == 5

    def test_missing_result_id_returns_error(self, pipeline):
        out = json.loads(pipeline.save_results({
            "result_id": "res_nope",
            "path": "x.json",
        }))
        assert "error" in out

    def test_no_data_no_result_id_returns_error(self, pipeline):
        out = json.loads(pipeline.save_results({"path": "x.json"}))
        assert "error" in out

    def test_explicit_data_still_works(self, pipeline):
        out = json.loads(pipeline.save_results({
            "data": {"foo": "bar"},
            "path": "plain.json",
            "format": "json",
        }))
        assert "path" in out
        assert "error" not in out


# ----------------------------------------------------------------------
# snapshot ↔ ResultStore
# ----------------------------------------------------------------------
class TestSnapshotResultStore:
    def test_captures_result_lineage(self, pipeline, store, sample_csv):
        loaded = json.loads(pipeline.load_data({"path": sample_csv}))
        rid = loaded["result_id"]
        step = json.loads(pipeline.transform({
            "result_id": rid,
            "operations": [{"type": "select", "params": {"columns": ["age"]}}],
        }))
        new_rid = step["result_id"]
        snap = json.loads(pipeline.snapshot({
            "label": "lineage_snap",
            "result_ids": [new_rid],
            "code": "test",
        }))
        assert "result_lineage" in snap
        assert new_rid in snap["result_lineage"]
        ids = [item["id"] for item in snap["result_lineage"][new_rid]]
        assert rid in ids

    def test_snapshot_records_global_seed(self, pipeline, sample_csv):
        GlobalSeed.set(123)
        snap = json.loads(pipeline.snapshot({"label": "seed_snap"}))
        assert snap["seed"] == 123

    def test_unknown_result_id_recorded_as_error(self, pipeline):
        snap = json.loads(pipeline.snapshot({
            "label": "x", "result_ids": ["res_nope"],
        }))
        assert "error" in snap["result_lineage"]["res_nope"]


# ----------------------------------------------------------------------
# seed_manager ↔ GlobalSeed
# ----------------------------------------------------------------------
class TestSeedManagerGlobal:
    def test_set_propagates_to_global_seed(self, pipeline):
        out = json.loads(pipeline.seed_manager({"action": "set", "seed": 77}))
        assert out["seed"] == 77
        assert GlobalSeed.get() == 77

    def test_get_reflects_global_seed(self, pipeline):
        GlobalSeed.set(11)
        out = json.loads(pipeline.seed_manager({"action": "get"}))
        assert out["seed"] == 11

    def test_reset_clears_global_seed(self, pipeline):
        GlobalSeed.set(55)
        out = json.loads(pipeline.seed_manager({"action": "reset"}))
        assert out["seed"] is None
        assert GlobalSeed.get() is None

    def test_set_propagates_to_python_random(self, pipeline):
        pipeline.seed_manager({"action": "set", "seed": 42})
        a = [random.random() for _ in range(3)]
        pipeline.seed_manager({"action": "set", "seed": 42})
        b = [random.random() for _ in range(3)]
        assert a == b

    def test_set_propagates_to_numpy(self, pipeline):
        pipeline.seed_manager({"action": "set", "seed": 42})
        a = np.random.rand(3).tolist()
        pipeline.seed_manager({"action": "set", "seed": 42})
        b = np.random.rand(3).tolist()
        assert a == b

    def test_invalid_seed_returns_error(self, pipeline):
        out = json.loads(pipeline.seed_manager({"action": "set", "seed": "abc"}))
        assert "error" in out
