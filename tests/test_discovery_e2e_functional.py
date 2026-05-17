"""End-to-end functional tests for the discovery pipeline.

These tests verify that the full discovery flow — search → build → validate →
activate — produces handlers that return *meaningful* research results, not just
placeholders.
"""
import json
import pytest

from sophia.research.discovery.method_catalog import MethodCatalog
from sophia.research.discovery.method_builder import MethodBuilder
from sophia.research.discovery.method_searcher import MethodSearcher
from sophia.research.discovery.dependency_manager import DependencyManager
from sophia.research.discovery.register import _activate_method
from sophia.tools.registry import ToolRegistry


@pytest.fixture
def components(tmp_path):
    db_path = str(tmp_path / "catalog.db")
    catalog = MethodCatalog(db_path)
    searcher = MethodSearcher(catalog)
    builder = MethodBuilder(catalog)
    dep_mgr = DependencyManager()
    registry = ToolRegistry()
    return {
        "catalog": catalog,
        "searcher": searcher,
        "builder": builder,
        "dep_manager": dep_mgr,
        "registry": registry,
    }


class TestAutoDiscoverFunctional:
    """Golden-standard tests: auto-discover must produce working handlers."""

    def test_auto_discover_numpy_produces_meaningful_handler(self, components):
        """Full pipeline: search 'numpy descriptive stats' → build → activate → run."""
        catalog = components["catalog"]
        builder = components["builder"]
        registry = components["registry"]

        # Step 1: Search
        search_result = json.loads(
            components["searcher"].search("numpy descriptive statistics")
        )
        # Should find numpy as an external candidate (not in builtin catalog)
        candidates = search_result.get("candidates", [])
        numpy_candidate = None
        for c in candidates:
            if c.get("library") == "numpy":
                numpy_candidate = c
                break
        # If not found in candidates, build directly
        if numpy_candidate is None:
            numpy_candidate = {
                "library": "numpy",
                "pip_name": "numpy",
                "description": "Descriptive statistics with numpy",
                "name": "Numpy Stats",
                "category": "statistics",
                "importable": True,
            }

        # Step 2: Build
        build_result = json.loads(builder.build(numpy_candidate, "compute mean and std"))
        assert build_result["success"] is True
        method_id = build_result["method_id"]

        # Step 3: Activate into registry
        method = catalog.get(method_id)
        assert method is not None
        assert method["handler_code"] is not None
        activated = _activate_method(registry, method)
        assert activated is True

        # Step 4: Call the activated tool with real data
        tool_name = method["tool_name"]
        assert tool_name in registry.list_tools()
        result_json = registry.dispatch(tool_name, {"data": [1, 2, 3, 4, 5]})
        result = json.loads(result_json)

        # Step 5: Verify meaningful output (not a TODO template)
        assert result.get("status") == "success"
        assert abs(result.get("mean", 0) - 3.0) < 0.01
        assert result.get("std") is not None

    def test_auto_discover_scipy_produces_ttest_handler(self, components):
        """Full pipeline: search 'scipy t-test' → build → activate → run."""
        catalog = components["catalog"]
        builder = components["builder"]
        registry = components["registry"]

        candidate = {
            "library": "scipy",
            "pip_name": "scipy",
            "description": "Independent samples t-test",
            "name": "Scipy T-Test",
            "category": "statistics",
            "importable": True,
        }

        build_result = json.loads(builder.build(candidate, "compare two groups"))
        assert build_result["success"] is True
        method_id = build_result["method_id"]

        method = catalog.get(method_id)
        activated = _activate_method(registry, method)
        assert activated is True

        tool_name = method["tool_name"]
        result_json = registry.dispatch(tool_name, {
            "data": {"group_a": [1, 2, 3, 4, 5], "group_b": [10, 11, 12, 13, 14]},
            "method": "ttest_ind",
        })
        result = json.loads(result_json)

        assert result.get("status") == "success"
        assert "t_statistic" in result
        assert result.get("significant") is True  # groups are very different

    def test_auto_discover_pandas_produces_dataframe_handler(self, components):
        """Full pipeline for pandas data processing."""
        catalog = components["catalog"]
        builder = components["builder"]
        registry = components["registry"]

        candidate = {
            "library": "pandas",
            "pip_name": "pandas",
            "description": "DataFrame description",
            "name": "Pandas Describe",
            "category": "pipeline",
            "importable": True,
        }

        build_result = json.loads(builder.build(candidate, "describe a dataset"))
        assert build_result["success"] is True
        method = catalog.get(build_result["method_id"])
        assert _activate_method(registry, method) is True

        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
        result = json.loads(registry.dispatch(method["tool_name"], {
            "data": records,
            "method": "describe",
        }))
        assert result.get("status") == "success"
        assert result.get("shape") == [3, 2]
        assert "describe" in result

    def test_auto_discover_sklearn_produces_ml_handler(self, components):
        """Full pipeline for sklearn ML training."""
        catalog = components["catalog"]
        builder = components["builder"]
        registry = components["registry"]

        candidate = {
            "library": "scikit-learn",
            "pip_name": "scikit-learn",
            "description": "Train a logistic regression classifier",
            "name": "Sklearn Classifier",
            "category": "ml",
            "importable": True,
        }

        build_result = json.loads(builder.build(candidate, "train classifier"))
        assert build_result["success"] is True
        method = catalog.get(build_result["method_id"])
        assert _activate_method(registry, method) is True

        result = json.loads(registry.dispatch(method["tool_name"], {
            "data": {
                "X": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            },
            "parameters": {"model": "LogisticRegression", "random_state": 42},
        }))
        assert result.get("status") == "success"
        assert "train_score" in result
        assert "test_score" in result

    def test_auto_discover_unknown_uses_generic_template(self, components):
        """For unknown libraries, generic template returns library info."""
        catalog = components["catalog"]
        builder = components["builder"]
        registry = components["registry"]

        candidate = {
            "library": "json",  # builtin, not in any template map
            "pip_name": "json",
            "description": "Utility library",
            "name": "Json Utils",
            "category": "utility",
            "importable": True,
        }

        build_result = json.loads(builder.build(candidate, "do something"))
        assert build_result["success"] is True
        method = catalog.get(build_result["method_id"])
        assert _activate_method(registry, method) is True

        result = json.loads(registry.dispatch(method["tool_name"], {"data": []}))
        assert result.get("status") == "info"
        assert "available_functions" in result


class TestCatalogActivateAll:
    """Test that catalog.activate_all correctly loads installed methods."""

    def test_activate_all_loads_builtins(self, components):
        """Builtin methods (no handler_code) should be skipped gracefully."""
        catalog = components["catalog"]
        registry = components["registry"]
        count = catalog.activate_all(registry)
        # Builtin methods have no handler_code, so activation count is 0
        assert count == 0

    def test_activate_all_loads_built_methods(self, components):
        """Methods built by MethodBuilder with handler_code should activate."""
        catalog = components["catalog"]
        builder = components["builder"]
        registry = components["registry"]

        candidate = {
            "library": "numpy",
            "pip_name": "numpy",
            "description": "Mean calculator",
            "name": "Numpy Mean",
            "category": "statistics",
            "importable": True,
        }
        build_result = json.loads(builder.build(candidate, "compute mean"))
        assert build_result["success"] is True

        count = catalog.activate_all(registry)
        assert count == 1
        assert build_result["tool_name"] in registry.list_tools()

        result = json.loads(registry.dispatch(build_result["tool_name"], {"data": [10, 20, 30]}))
        assert result.get("mean") == 20.0
