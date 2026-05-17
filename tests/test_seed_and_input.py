"""Tests for GlobalSeed and _resolve_input."""
import os
import random

import numpy as np
import pandas as pd
import pytest

from sophia.research.seed import GlobalSeed
from sophia.research._input import (
    InputResolutionError,
    resolve_dataframe,
    resolve_parent_ids,
)
from sophia.research.result_store import ResultStore


@pytest.fixture(autouse=True)
def _reset_seed():
    """Ensure each test starts with no global seed."""
    GlobalSeed.reset()
    yield
    GlobalSeed.reset()


# ----------------------------------------------------------------------
# GlobalSeed
# ----------------------------------------------------------------------
class TestGlobalSeedSetGet:
    def test_set_returns_seed(self):
        assert GlobalSeed.set(123) == 123

    def test_get_after_set(self):
        GlobalSeed.set(7)
        assert GlobalSeed.get() == 7

    def test_get_before_set_is_none(self):
        assert GlobalSeed.get() is None

    def test_reset_clears_seed(self):
        GlobalSeed.set(99)
        GlobalSeed.reset()
        assert GlobalSeed.get() is None

    def test_set_propagates_to_numpy(self):
        GlobalSeed.set(42)
        a = np.random.rand(5)
        GlobalSeed.set(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_set_propagates_to_random(self):
        GlobalSeed.set(42)
        a = [random.random() for _ in range(5)]
        GlobalSeed.set(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_set_propagates_to_pythonhashseed(self):
        GlobalSeed.set(123)
        assert os.environ.get("PYTHONHASHSEED") == "123"

    def test_set_coerces_string_to_int(self):
        GlobalSeed.set("55")
        assert GlobalSeed.get() == 55


class TestGetOrDefault:
    def test_returns_set_seed(self):
        GlobalSeed.set(7)
        assert GlobalSeed.get_or_default() == 7

    def test_returns_default_when_unset(self):
        assert GlobalSeed.get_or_default(99) == 99

    def test_default_default_is_42(self):
        assert GlobalSeed.get_or_default() == 42

    def test_set_overrides_default(self):
        GlobalSeed.set(0)  # zero is a valid seed, must not be treated as unset
        assert GlobalSeed.get_or_default(42) == 0


class TestWithSeed:
    def test_context_sets_and_restores(self):
        GlobalSeed.set(10)
        with GlobalSeed.with_seed(99):
            assert GlobalSeed.get() == 99
        assert GlobalSeed.get() == 10

    def test_context_restores_when_unset(self):
        with GlobalSeed.with_seed(99):
            assert GlobalSeed.get() == 99
        assert GlobalSeed.get() is None

    def test_context_restores_on_exception(self):
        GlobalSeed.set(5)
        try:
            with GlobalSeed.with_seed(99):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert GlobalSeed.get() == 5


# ----------------------------------------------------------------------
# resolve_dataframe — result_id branch
# ----------------------------------------------------------------------
class TestResolveResultId:
    def test_resolve_from_result_id(self, tmp_path):
        store = ResultStore(str(tmp_path))
        df = pd.DataFrame({"a": [1, 2, 3]})
        rid = store.store(df, kind="dataframe", tool="t")
        out = resolve_dataframe({"result_id": rid}, store=store)
        pd.testing.assert_frame_equal(out, df)

    def test_missing_store_raises(self, tmp_path):
        with pytest.raises(InputResolutionError, match="no ResultStore"):
            resolve_dataframe({"result_id": "res_xyz"}, store=None)

    def test_unknown_result_id_raises(self, tmp_path):
        store = ResultStore(str(tmp_path))
        with pytest.raises(InputResolutionError, match="not found"):
            resolve_dataframe({"result_id": "res_nope"}, store=store)

    def test_uncoercible_result_raises(self, tmp_path):
        store = ResultStore(str(tmp_path))
        rid = store.store("plain text", kind="text", tool="t")
        with pytest.raises(InputResolutionError, match="DataFrame"):
            resolve_dataframe({"result_id": rid}, store=store)


# ----------------------------------------------------------------------
# resolve_dataframe — inline data branch
# ----------------------------------------------------------------------
class TestResolveInlineData:
    def test_list_of_dicts(self):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        out = resolve_dataframe({"data": data})
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 2

    def test_dict_of_cols(self):
        data = {"a": [1, 2], "b": [3, 4]}
        out = resolve_dataframe({"data": data})
        assert list(out.columns) == ["a", "b"]

    def test_dataframe_passthrough(self):
        df = pd.DataFrame({"x": [1]})
        out = resolve_dataframe({"data": df})
        pd.testing.assert_frame_equal(out, df)

    def test_series(self):
        s = pd.Series([1, 2, 3], name="v")
        out = resolve_dataframe({"data": s})
        assert "v" in out.columns

    def test_ndarray(self):
        arr = np.array([[1, 2], [3, 4]])
        out = resolve_dataframe({"data": arr})
        assert out.shape == (2, 2)

    def test_empty_list(self):
        out = resolve_dataframe({"data": []})
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

    def test_list_of_scalars(self):
        out = resolve_dataframe({"data": [1, 2, 3]})
        assert "value" in out.columns
        assert len(out) == 3


# ----------------------------------------------------------------------
# resolve_dataframe — path branch
# ----------------------------------------------------------------------
class TestResolvePath:
    def test_csv(self, tmp_path):
        p = tmp_path / "x.csv"
        p.write_text("a,b\n1,2\n3,4\n")
        out = resolve_dataframe({"path": str(p)})
        assert list(out.columns) == ["a", "b"]
        assert len(out) == 2

    def test_json(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text('[{"a":1,"b":2},{"a":3,"b":4}]')
        out = resolve_dataframe({"path": str(p)})
        assert len(out) == 2

    def test_missing_file_raises(self):
        with pytest.raises(InputResolutionError, match="not found"):
            resolve_dataframe({"path": "/tmp/does_not_exist_xyz.csv"})

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "x.bin"
        p.write_bytes(b"\x00\x01")
        with pytest.raises(InputResolutionError, match="Unsupported"):
            resolve_dataframe({"path": str(p)})

    def test_with_guard(self, tmp_path):
        from sophia.research.workspace_guard import WorkspaceGuard
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "data.csv").write_text("a,b\n1,2\n")
        guard = WorkspaceGuard(str(ws))
        out = resolve_dataframe({"path": "data.csv"}, guard=guard)
        assert list(out.columns) == ["a", "b"]


# ----------------------------------------------------------------------
# Precedence
# ----------------------------------------------------------------------
class TestPrecedence:
    def test_result_id_beats_data(self, tmp_path):
        store = ResultStore(str(tmp_path))
        rid = store.store(pd.DataFrame({"src": ["store"]}), kind="dataframe", tool="t")
        out = resolve_dataframe(
            {"result_id": rid, "data": [{"src": "inline"}]},
            store=store,
        )
        assert out["src"].iloc[0] == "store"

    def test_data_beats_path(self, tmp_path):
        p = tmp_path / "x.csv"
        p.write_text("src\npath\n")
        out = resolve_dataframe({"data": [{"src": "inline"}], "path": str(p)})
        assert out["src"].iloc[0] == "inline"


# ----------------------------------------------------------------------
# require flag
# ----------------------------------------------------------------------
class TestRequire:
    def test_returns_none_when_empty(self):
        assert resolve_dataframe({}) is None

    def test_require_raises_when_empty(self):
        with pytest.raises(InputResolutionError):
            resolve_dataframe({}, require=True)


# ----------------------------------------------------------------------
# Custom keys
# ----------------------------------------------------------------------
class TestCustomKeys:
    def test_custom_data_key(self):
        out = resolve_dataframe(
            {"records": [{"a": 1}]},
            data_key="records",
        )
        assert out.iloc[0]["a"] == 1

    def test_custom_path_key(self, tmp_path):
        p = tmp_path / "x.csv"
        p.write_text("a\n1\n")
        out = resolve_dataframe({"file": str(p)}, path_key="file")
        assert out["a"].iloc[0] == 1


# ----------------------------------------------------------------------
# resolve_parent_ids
# ----------------------------------------------------------------------
class TestResolveParentIds:
    def test_single_result_id(self):
        ids = resolve_parent_ids({"result_id": "res_abc12345"})
        assert ids == ["res_abc12345"]

    def test_list_result_ids(self):
        ids = resolve_parent_ids({"result_ids": ["res_aaa", "res_bbb"]})
        assert ids == ["res_aaa", "res_bbb"]

    def test_ignores_non_result_strings(self):
        ids = resolve_parent_ids({"result_id": "not_a_res_id"})
        assert ids == []

    def test_no_keys_returns_empty(self):
        assert resolve_parent_ids({}) == []

    def test_dedupes(self):
        ids = resolve_parent_ids(
            {"result_id": "res_aaa", "parents": ["res_aaa", "res_bbb"]}
        )
        assert ids == ["res_aaa", "res_bbb"]

    def test_custom_keys(self):
        ids = resolve_parent_ids(
            {"x_result_id": "res_aaa", "y_result_id": "res_bbb"},
        )
        assert "res_aaa" in ids
        assert "res_bbb" in ids
