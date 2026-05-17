"""Tests for ResultStore: SQLite-backed result persistence with lineage."""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest

from sophia.research.result_store import ResultStore, INLINE_LIMIT_BYTES


class _FakeModel:
    """Module-level dummy model class (must be top-level for pickle)."""

    def __init__(self, coef=None):
        self.coef_ = coef if coef is not None else [1.0, 2.0]

    def __eq__(self, other):
        return isinstance(other, _FakeModel) and self.coef_ == other.coef_


@pytest.fixture
def store(tmp_path):
    return ResultStore(str(tmp_path))


@pytest.fixture
def store_with_data(tmp_path):
    s = ResultStore(str(tmp_path))
    rid1 = s.store({"a": 1}, kind="result", tool="t1")
    rid2 = s.store({"b": 2}, kind="result", tool="t2", parents=[rid1])
    rid3 = s.store({"c": 3}, kind="result", tool="t3", parents=[rid2])
    return s, rid1, rid2, rid3


# ----------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------
class TestInit:
    def test_creates_db_file(self, tmp_path):
        ResultStore(str(tmp_path))
        assert os.path.exists(os.path.join(tmp_path, ".research", "results.db"))

    def test_creates_payload_dir(self, tmp_path):
        ResultStore(str(tmp_path))
        assert os.path.isdir(os.path.join(tmp_path, ".research", "cache", "results"))

    def test_wal_mode(self, store):
        cur = store._conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0].lower() == "wal"

    def test_reopen_existing(self, tmp_path):
        s1 = ResultStore(str(tmp_path))
        rid = s1.store({"x": 1}, kind="result", tool="t")
        s1.close()
        s2 = ResultStore(str(tmp_path))
        data = s2.get(rid)
        assert data == {"x": 1}

    def test_workspace_normalized(self, tmp_path):
        s = ResultStore(str(tmp_path))
        assert os.path.isabs(s.workspace)


# ----------------------------------------------------------------------
# Store / Get basic types
# ----------------------------------------------------------------------
class TestStoreGet:
    def test_store_returns_id_with_prefix(self, store):
        rid = store.store({"x": 1}, kind="result", tool="t")
        assert rid.startswith("res_")
        assert len(rid) == 12  # "res_" + 8 hex chars

    def test_store_get_dict(self, store):
        rid = store.store({"a": 1, "b": "two"}, kind="result", tool="t")
        assert store.get(rid) == {"a": 1, "b": "two"}

    def test_store_get_list(self, store):
        rid = store.store([1, 2, 3], kind="result", tool="t")
        assert store.get(rid) == [1, 2, 3]

    def test_store_get_string(self, store):
        rid = store.store("hello world", kind="text", tool="t")
        assert store.get(rid) == "hello world"

    def test_store_get_int(self, store):
        rid = store.store(42, kind="result", tool="t")
        assert store.get(rid) == 42

    def test_store_get_float(self, store):
        rid = store.store(3.14, kind="result", tool="t")
        assert store.get(rid) == 3.14

    def test_store_get_bool(self, store):
        rid = store.store(True, kind="result", tool="t")
        assert store.get(rid) is True

    def test_store_get_none(self, store):
        rid = store.store(None, kind="result", tool="t")
        assert store.get(rid) is None

    def test_store_get_nested(self, store):
        data = {"a": [1, 2, {"nested": True}], "b": {"x": [3.0, 4.0]}}
        rid = store.store(data, kind="result", tool="t")
        assert store.get(rid) == data


# ----------------------------------------------------------------------
# DataFrame round-trip
# ----------------------------------------------------------------------
class TestDataFrame:
    def test_store_get_dataframe_simple(self, store):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        rid = store.store(df, kind="dataframe", tool="load_data")
        out = store.get(rid)
        pd.testing.assert_frame_equal(out, df)

    def test_store_get_dataframe_with_dates(self, store):
        df = pd.DataFrame({
            "t": pd.date_range("2024-01-01", periods=10, freq="D"),
            "v": np.arange(10),
        })
        rid = store.store(df, kind="dataframe", tool="t")
        out = store.get(rid)
        pd.testing.assert_frame_equal(out, df)

    def test_store_get_dataframe_with_nans(self, store):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        rid = store.store(df, kind="dataframe", tool="t")
        out = store.get(rid)
        pd.testing.assert_frame_equal(out, df)

    def test_store_get_large_dataframe(self, store):
        # ~ 10000 rows × 5 cols. Forces pickle path.
        df = pd.DataFrame(np.random.randn(10000, 5), columns=list("abcde"))
        rid = store.store(df, kind="dataframe", tool="t")
        meta = store.get_metadata(rid)
        assert meta["payload_path"] is not None
        assert meta["inline"] is False
        out = store.get(rid)
        pd.testing.assert_frame_equal(out, df)

    def test_dataframe_summary_has_shape(self, store):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        rid = store.store(df, kind="dataframe", tool="t")
        meta = store.get_metadata(rid)
        assert meta["summary"]["shape"] == [2, 2]
        assert "a" in meta["summary"]["columns"]


class TestGetDataframe:
    def test_from_dataframe(self, store):
        df = pd.DataFrame({"a": [1, 2]})
        rid = store.store(df, kind="dataframe", tool="t")
        out = store.get_dataframe(rid)
        pd.testing.assert_frame_equal(out, df)

    def test_from_list_of_dicts(self, store):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        rid = store.store(data, kind="result", tool="t")
        out = store.get_dataframe(rid)
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 2

    def test_from_ndarray(self, store):
        arr = np.array([[1, 2], [3, 4]])
        rid = store.store(arr, kind="array", tool="t")
        out = store.get_dataframe(rid)
        assert out.shape == (2, 2)

    def test_from_series(self, store):
        s = pd.Series([1, 2, 3], name="v")
        rid = store.store(s, kind="dataframe", tool="t")
        out = store.get_dataframe(rid)
        assert "v" in out.columns

    def test_from_dict_of_cols(self, store):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        rid = store.store(data, kind="result", tool="t")
        out = store.get_dataframe(rid)
        assert list(out.columns) == ["a", "b"]

    def test_from_list_of_scalars(self, store):
        rid = store.store([1, 2, 3], kind="result", tool="t")
        out = store.get_dataframe(rid)
        assert "value" in out.columns
        assert len(out) == 3

    def test_unconvertible_raises(self, store):
        rid = store.store("not a dataframe", kind="text", tool="t")
        with pytest.raises(TypeError):
            store.get_dataframe(rid)


# ----------------------------------------------------------------------
# ndarray
# ----------------------------------------------------------------------
class TestNdarray:
    def test_store_get_1d(self, store):
        arr = np.array([1.0, 2.0, 3.0])
        rid = store.store(arr, kind="array", tool="t")
        out = store.get(rid)
        np.testing.assert_array_equal(out, arr)

    def test_store_get_2d(self, store):
        arr = np.random.randn(50, 50)
        rid = store.store(arr, kind="array", tool="t")
        out = store.get(rid)
        np.testing.assert_array_equal(out, arr)

    def test_ndarray_always_pickled(self, store):
        arr = np.array([1, 2, 3])
        rid = store.store(arr, kind="array", tool="t")
        meta = store.get_metadata(rid)
        assert meta["payload_path"] is not None


# ----------------------------------------------------------------------
# Inline vs disk decision
# ----------------------------------------------------------------------
class TestInlineVsDisk:
    def test_small_dict_inline(self, store):
        rid = store.store({"x": 1}, kind="result", tool="t")
        meta = store.get_metadata(rid)
        assert meta["inline"] is True
        assert meta["payload_path"] is None

    def test_large_dict_on_disk(self, store):
        big = {f"k{i}": "x" * 100 for i in range(200)}
        rid = store.store(big, kind="result", tool="t")
        meta = store.get_metadata(rid)
        assert meta["payload_path"] is not None
        assert meta["inline"] is False

    def test_dataframe_always_on_disk(self, store):
        df = pd.DataFrame({"a": [1]})
        rid = store.store(df, kind="dataframe", tool="t")
        meta = store.get_metadata(rid)
        assert meta["payload_path"] is not None


# ----------------------------------------------------------------------
# Model kind
# ----------------------------------------------------------------------
class TestModelKind:
    def test_store_get_model_like_object(self, store):
        rid = store.store(_FakeModel(coef=[1.0, 2.0]), kind="model", tool="train")
        out = store.get(rid)
        assert out.coef_ == [1.0, 2.0]
        assert isinstance(out, _FakeModel)


# ----------------------------------------------------------------------
# Parents / lineage
# ----------------------------------------------------------------------
class TestParents:
    def test_store_with_parents(self, store):
        p = store.store({"x": 1}, kind="result", tool="t1")
        rid = store.store({"y": 2}, kind="result", tool="t2", parents=[p])
        meta = store.get_metadata(rid)
        assert meta["parents"] == [p]

    def test_parents_validated(self, store):
        with pytest.raises(ValueError, match="Parent result_id not found"):
            store.store({"y": 1}, kind="result", tool="t", parents=["res_invalid"])

    def test_multiple_parents(self, store):
        p1 = store.store({"a": 1}, kind="result", tool="t1")
        p2 = store.store({"b": 2}, kind="result", tool="t2")
        rid = store.store({"c": 3}, kind="result", tool="t3", parents=[p1, p2])
        meta = store.get_metadata(rid)
        assert set(meta["parents"]) == {p1, p2}


class TestLineage:
    def test_three_level_lineage(self, store_with_data):
        s, r1, r2, r3 = store_with_data
        lineage = s.lineage(r3)
        ids = [item["id"] for item in lineage]
        assert ids[0] == r3
        assert r2 in ids
        assert r1 in ids

    def test_lineage_depth_field(self, store_with_data):
        s, r1, r2, r3 = store_with_data
        lineage = s.lineage(r3)
        m = {item["id"]: item["depth"] for item in lineage}
        assert m[r3] == 0
        assert m[r2] == 1
        assert m[r1] == 2

    def test_lineage_single_node(self, store):
        rid = store.store({"x": 1}, kind="result", tool="t")
        lineage = store.lineage(rid)
        assert len(lineage) == 1
        assert lineage[0]["id"] == rid

    def test_lineage_unknown_id_raises(self, store):
        with pytest.raises(KeyError):
            store.lineage("res_nope")

    def test_lineage_dag_shared_ancestor(self, store):
        # diamond DAG: r1 → r2, r1 → r3, r4 ← r2 + r3
        r1 = store.store({"x": 1}, kind="result", tool="root")
        r2 = store.store({"y": 2}, kind="result", tool="branch", parents=[r1])
        r3 = store.store({"z": 3}, kind="result", tool="branch", parents=[r1])
        r4 = store.store({"w": 4}, kind="result", tool="merge", parents=[r2, r3])
        lineage = store.lineage(r4)
        ids = [item["id"] for item in lineage]
        assert ids.count(r1) == 1  # not duplicated


# ----------------------------------------------------------------------
# get / get_metadata / exists / delete
# ----------------------------------------------------------------------
class TestExists:
    def test_exists_true(self, store):
        rid = store.store({"x": 1}, kind="result", tool="t")
        assert store.exists(rid)

    def test_exists_false(self, store):
        assert not store.exists("res_nope")

    def test_exists_empty(self, store):
        assert not store.exists("")
        assert not store.exists(None)


class TestGet:
    def test_get_missing_raises(self, store):
        with pytest.raises(KeyError):
            store.get("res_missing")

    def test_get_metadata_missing_raises(self, store):
        with pytest.raises(KeyError):
            store.get_metadata("res_missing")

    def test_metadata_has_required_fields(self, store):
        rid = store.store(
            {"x": 1}, kind="result", tool="t1",
            params={"alpha": 0.05},
        )
        meta = store.get_metadata(rid)
        assert meta["id"] == rid
        assert meta["kind"] == "result"
        assert meta["tool"] == "t1"
        assert meta["params"] == {"alpha": 0.05}
        assert meta["parents"] == []
        assert "created_at" in meta
        assert "summary" in meta


class TestDelete:
    def test_delete_existing(self, store):
        rid = store.store({"x": 1}, kind="result", tool="t")
        assert store.delete(rid) is True
        assert not store.exists(rid)

    def test_delete_missing_returns_false(self, store):
        assert store.delete("res_missing") is False

    def test_delete_removes_pickle_file(self, store):
        df = pd.DataFrame({"a": [1, 2]})
        rid = store.store(df, kind="dataframe", tool="t")
        meta = store.get_metadata(rid)
        path = meta["payload_path"]
        assert os.path.exists(path)
        store.delete(rid)
        assert not os.path.exists(path)


class TestClear:
    def test_clear_returns_count(self, store):
        for i in range(5):
            store.store({"i": i}, kind="result", tool="t")
        n = store.clear()
        assert n == 5
        assert store.get_stats()["total"] == 0

    def test_clear_removes_pickles(self, store):
        df = pd.DataFrame({"a": [1]})
        rid = store.store(df, kind="dataframe", tool="t")
        path = store.get_metadata(rid)["payload_path"]
        store.clear()
        assert not os.path.exists(path)


# ----------------------------------------------------------------------
# list_by_kind
# ----------------------------------------------------------------------
class TestListByKind:
    def test_list_all(self, store):
        store.store({"a": 1}, kind="result", tool="t1")
        store.store(pd.DataFrame({"x": [1]}), kind="dataframe", tool="t2")
        store.store([1, 2], kind="result", tool="t3")
        lst = store.list_by_kind()
        assert len(lst) == 3

    def test_filter_by_kind(self, store):
        store.store({"a": 1}, kind="result", tool="t1")
        store.store(pd.DataFrame({"x": [1]}), kind="dataframe", tool="t2")
        df_only = store.list_by_kind(kind="dataframe")
        assert len(df_only) == 1
        assert df_only[0]["kind"] == "dataframe"

    def test_filter_by_tool(self, store):
        store.store({"a": 1}, kind="result", tool="ttest")
        store.store({"a": 2}, kind="result", tool="anova")
        store.store({"a": 3}, kind="result", tool="ttest")
        ttests = store.list_by_kind(tool="ttest")
        assert len(ttests) == 2

    def test_limit_applied(self, store):
        for i in range(10):
            store.store({"i": i}, kind="result", tool="t")
        lst = store.list_by_kind(limit=3)
        assert len(lst) == 3

    def test_returns_newest_first(self, store):
        rids = []
        for i in range(3):
            rids.append(store.store({"i": i}, kind="result", tool="t"))
            # Small sleep to ensure distinct timestamps even on fast systems
            import time
            time.sleep(0.01)
        lst = store.list_by_kind()
        # most recent first
        assert lst[0]["id"] == rids[-1]


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------
class TestStats:
    def test_stats_empty(self, store):
        s = store.get_stats()
        assert s["total"] == 0
        assert s["by_kind"] == {}

    def test_stats_counts(self, store):
        store.store({"a": 1}, kind="result", tool="t1")
        store.store({"b": 2}, kind="result", tool="t1")
        store.store(pd.DataFrame({"x": [1]}), kind="dataframe", tool="t2")
        s = store.get_stats()
        assert s["total"] == 3
        assert s["by_kind"]["result"] == 2
        assert s["by_kind"]["dataframe"] == 1
        assert s["by_tool"]["t1"] == 2
        assert s["by_tool"]["t2"] == 1

    def test_stats_inline_vs_disk_split(self, store):
        store.store({"x": 1}, kind="result", tool="t")  # inline
        store.store(pd.DataFrame({"x": [1, 2]}), kind="dataframe", tool="t")  # disk
        s = store.get_stats()
        assert s["inline"] == 1
        assert s["on_disk"] == 1


# ----------------------------------------------------------------------
# Validation / errors
# ----------------------------------------------------------------------
class TestValidation:
    def test_unknown_kind_raises(self, store):
        with pytest.raises(ValueError, match="Unknown kind"):
            store.store({"x": 1}, kind="bogus", tool="t")

    def test_empty_tool_raises(self, store):
        with pytest.raises(ValueError, match="tool"):
            store.store({"x": 1}, kind="result", tool="")

    def test_none_tool_raises(self, store):
        with pytest.raises(ValueError):
            store.store({"x": 1}, kind="result", tool=None)


# ----------------------------------------------------------------------
# result_id uniqueness
# ----------------------------------------------------------------------
class TestResultIdUniqueness:
    def test_many_ids_distinct(self, store):
        ids = set()
        for i in range(200):
            rid = store.store({"i": i}, kind="result", tool="t")
            assert rid not in ids
            ids.add(rid)
        assert len(ids) == 200


# ----------------------------------------------------------------------
# Persistence round-trip across reopens
# ----------------------------------------------------------------------
class TestPersistence:
    def test_persistence_across_reopens(self, tmp_path):
        s1 = ResultStore(str(tmp_path))
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        rid = s1.store(df, kind="dataframe", tool="t", params={"k": "v"})
        s1.close()

        s2 = ResultStore(str(tmp_path))
        out = s2.get(rid)
        pd.testing.assert_frame_equal(out, df)
        meta = s2.get_metadata(rid)
        assert meta["tool"] == "t"
        assert meta["params"] == {"k": "v"}

    def test_lineage_persisted(self, tmp_path):
        s1 = ResultStore(str(tmp_path))
        r1 = s1.store({"a": 1}, kind="result", tool="t1")
        r2 = s1.store({"b": 2}, kind="result", tool="t2", parents=[r1])
        s1.close()

        s2 = ResultStore(str(tmp_path))
        lineage = s2.lineage(r2)
        ids = [item["id"] for item in lineage]
        assert r1 in ids
        assert r2 in ids


# ----------------------------------------------------------------------
# Pickling: numpy / scipy outputs that aren't json-serializable inline
# ----------------------------------------------------------------------
class TestComplexPayloads:
    def test_numpy_scalars_in_dict(self, store):
        data = {
            "n": np.int64(42),
            "f": np.float64(3.14),
            "arr": np.array([1, 2, 3]),
        }
        rid = store.store(data, kind="result", tool="t")
        out = store.get(rid)
        assert int(out["n"]) == 42
        np.testing.assert_array_equal(out["arr"], np.array([1, 2, 3]))

    def test_pandas_series_payload(self, store):
        s = pd.Series([1, 2, 3], name="x")
        rid = store.store(s, kind="dataframe", tool="t")
        out = store.get(rid)
        pd.testing.assert_series_equal(out, s)
