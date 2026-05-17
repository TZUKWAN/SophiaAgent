"""End-to-end integration tests for the full discovery tool suite.

Tests the 5 discovery tools working together via ToolRegistry.dispatch:
  method_search → method_install → method_auto_discover
  method_list_available → method_verify
"""
import json
import os
import pytest

from sophia.research.discovery.method_catalog import MethodCatalog
from sophia.research.discovery.method_builder import MethodBuilder
from sophia.research.discovery.method_searcher import MethodSearcher
from sophia.research.discovery.dependency_manager import DependencyManager
from sophia.research.discovery.register import register_discovery_tools
from sophia.tools.registry import ToolRegistry


@pytest.fixture
def discovery_env(tmp_path):
    db_path = str(tmp_path / "catalog.db")
    catalog = MethodCatalog(db_path)
    searcher = MethodSearcher(catalog)
    builder = MethodBuilder(catalog)
    dep_mgr = DependencyManager()
    registry = ToolRegistry()

    components = {
        "catalog": catalog,
        "searcher": searcher,
        "builder": builder,
        "dep_manager": dep_mgr,
    }
    register_discovery_tools(registry, components)

    return {
        "catalog": catalog,
        "searcher": searcher,
        "builder": builder,
        "dep_mgr": dep_mgr,
        "registry": registry,
    }


class TestMethodSearchTool:
    def test_search_finds_numpy(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_search", {"description": "numpy descriptive statistics"})
        result = json.loads(result_json)
        assert result.get("source") in ("catalog", "external")
        # Should find numpy among candidates or list installed methods.
        assert "candidates" in result or "methods" in result

    def test_search_with_category_filter(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_search", {
            "description": "machine learning",
            "category": "ml",
        })
        result = json.loads(result_json)
        candidates = result.get("candidates", [])
        for c in candidates:
            assert c.get("category") == "ml"

    def test_search_empty_description(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_search", {"description": ""})
        result = json.loads(result_json)
        assert result.get("success") is False or "candidates" in result


class TestMethodInstallTool:
    def test_install_whitelisted_package(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_install", {"package": "numpy"})
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["whitelisted"] is True

    def test_install_nonexistent_package(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_install", {"package": "nonexistent_pkg_abc123"})
        result = json.loads(result_json)
        assert result["success"] is False
        assert result["whitelisted"] is False

    def test_install_empty_package(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_install", {"package": ""})
        result = json.loads(result_json)
        assert result["success"] is False


class TestMethodAutoDiscoverTool:
    def test_auto_discover_numpy_pipeline(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_auto_discover", {
            "description": "numpy descriptive statistics",
            "context": "compute mean and std",
        })
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["tool_name"] is not None
        assert result["method_id"] is not None
        assert result["library"] == "numpy"

        # Verify the tool is actually callable in the registry.
        tool_name = result["tool_name"]
        assert tool_name in registry.list_tools()
        run_result = json.loads(registry.dispatch(tool_name, {"data": [1, 2, 3, 4, 5]}))
        assert run_result.get("status") == "success"
        assert abs(run_result.get("mean", 0) - 3.0) < 0.01

    def test_auto_discover_scipy_pipeline(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_auto_discover", {
            "description": "scipy t-test independent samples",
            "context": "compare two groups",
        })
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["library"] == "scipy"

        tool_name = result["tool_name"]
        run_result = json.loads(registry.dispatch(tool_name, {
            "data": {"group_a": [1, 2, 3], "group_b": [10, 11, 12]},
            "method": "ttest_ind",
        }))
        assert run_result.get("status") == "success"
        assert "t_statistic" in run_result

    def test_auto_discover_sklearn_pipeline(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_auto_discover", {
            "description": "scikit-learn logistic regression",
            "context": "train a classifier",
        })
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["library"] in ("scikit-learn", "sklearn", "statsmodels")

        tool_name = result["tool_name"]
        run_result = json.loads(registry.dispatch(tool_name, {
            "data": {
                "X": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            },
            "parameters": {"model": "LogisticRegression", "random_state": 42},
        }))
        assert run_result.get("status") == "success"
        assert "train_score" in run_result

    def test_auto_discover_empty_description(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_auto_discover", {"description": ""})
        result = json.loads(result_json)
        assert result["success"] is False

    def test_auto_discover_no_candidates(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_auto_discover", {
            "description": "xyz_nonexistent_library_99999 do something impossible",
        })
        result = json.loads(result_json)
        # Should fail at search or build step.
        assert result["success"] is False
        assert "steps" in result

    def test_auto_discover_steps_trace(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_auto_discover", {
            "description": "pandas describe dataframe",
            "context": "describe a dataset",
        })
        result = json.loads(result_json)
        assert result["success"] is True
        steps = result["steps"]
        step_names = [s["step"] for s in steps]
        assert "search" in step_names
        assert "select_candidate" in step_names
        assert "build" in step_names
        assert "activate" in step_names


class TestMethodListAvailableTool:
    def test_list_all_methods(self, discovery_env):
        registry = discovery_env["registry"]
        # First auto-discover something to populate catalog.
        registry.dispatch("method_auto_discover", {
            "description": "numpy mean",
            "context": "compute mean",
        })
        result_json = registry.dispatch("method_list_available", {})
        result = json.loads(result_json)
        assert result["total"] > 0
        assert "methods" in result
        assert "stats" in result

    def test_list_filter_by_category(self, discovery_env):
        registry = discovery_env["registry"]
        registry.dispatch("method_auto_discover", {
            "description": "numpy statistics",
            "context": "compute stats",
        })
        result_json = registry.dispatch("method_list_available", {"category": "statistics"})
        result = json.loads(result_json)
        for m in result.get("methods", []):
            assert m.get("category") == "statistics"

    def test_list_filter_by_status(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_list_available", {"status": "installed"})
        result = json.loads(result_json)
        for m in result.get("methods", []):
            assert m.get("status") == "installed"


class TestMethodVerifyTool:
    def test_verify_builtin_method(self, discovery_env):
        registry = discovery_env["registry"]
        catalog = discovery_env["catalog"]
        # Builtin methods have no handler_code; verify should pass with note.
        builtins = catalog.list_methods(source="builtin")
        if not builtins:
            pytest.skip("No builtin methods in catalog")
        method = builtins[0]
        result_json = registry.dispatch("method_verify", {"method_id": method["id"]})
        result = json.loads(result_json)
        assert result["valid"] is True
        assert "Built-in method" in result.get("note", "")

    def test_verify_auto_discovered_method(self, discovery_env):
        registry = discovery_env["registry"]
        catalog = discovery_env["catalog"]
        # Auto-discover a method first.
        ad_result = json.loads(registry.dispatch("method_auto_discover", {
            "description": "numpy descriptive statistics",
            "context": "compute mean and std",
        }))
        method_id = ad_result["method_id"]
        result_json = registry.dispatch("method_verify", {"method_id": method_id})
        result = json.loads(result_json)
        assert result["valid"] is True
        assert result["syntax_ok"] is True
        assert result["exec_ok"] is True
        assert result["handle_callable"] is True
        assert result["schema_valid"] is True

        # Verify catalog was updated with verification status.
        method = catalog.get(method_id)
        assert method["verified"] == 1

    def test_verify_nonexistent_method(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_verify", {
            "method_id": "nonexistent_id_12345",
        })
        result = json.loads(result_json)
        assert result["valid"] is False
        assert result["success"] is False

    def test_verify_by_tool_name(self, discovery_env):
        registry = discovery_env["registry"]
        ad_result = json.loads(registry.dispatch("method_auto_discover", {
            "description": "numpy mean",
            "context": "compute mean",
        }))
        tool_name = ad_result["tool_name"]
        result_json = registry.dispatch("method_verify", {"tool_name": tool_name})
        result = json.loads(result_json)
        assert result["valid"] is True

    def test_verify_no_identifier(self, discovery_env):
        registry = discovery_env["registry"]
        result_json = registry.dispatch("method_verify", {})
        result = json.loads(result_json)
        assert result["success"] is False


class TestDiscoveryToolSuiteIntegration:
    def test_full_workflow_search_install_discover_verify_list(self, discovery_env):
        registry = discovery_env["registry"]

        # 1. Search
        search_result = json.loads(registry.dispatch("method_search", {
            "description": "scipy t-test",
        }))
        assert "candidates" in search_result

        # 2. Install (idempotent for already-installed)
        install_result = json.loads(registry.dispatch("method_install", {
            "package": "scipy",
        }))
        assert install_result["success"] is True

        # 3. Auto-discover
        discover_result = json.loads(registry.dispatch("method_auto_discover", {
            "description": "scipy t-test",
            "context": "compare groups",
        }))
        assert discover_result["success"] is True
        method_id = discover_result["method_id"]
        tool_name = discover_result["tool_name"]

        # 4. Verify
        verify_result = json.loads(registry.dispatch("method_verify", {
            "method_id": method_id,
        }))
        assert verify_result["valid"] is True

        # 5. List
        list_result = json.loads(registry.dispatch("method_list_available", {}))
        assert list_result["total"] > 0
        ids = [m["id"] for m in list_result["methods"]]
        assert method_id in ids

        # 6. Run the discovered tool
        run_result = json.loads(registry.dispatch(tool_name, {
            "data": {"group_a": [1, 2, 3], "group_b": [4, 5, 6]},
            "method": "ttest_ind",
        }))
        assert run_result.get("status") == "success"

    def test_multiple_auto_discoveries_add_tools(self, discovery_env):
        registry = discovery_env["registry"]
        initial_count = len(registry.list_tools())

        descriptions = [
            ("numpy descriptive statistics", "compute stats"),
            ("scipy t-test", "compare groups"),
            ("pandas describe", "describe data"),
        ]
        for desc, ctx in descriptions:
            result = json.loads(registry.dispatch("method_auto_discover", {
                "description": desc,
                "context": ctx,
            }))
            assert result["success"] is True

        final_count = len(registry.list_tools())
        assert final_count >= initial_count + 3

    def test_discovered_tool_persistence(self, discovery_env):
        registry = discovery_env["registry"]
        catalog = discovery_env["catalog"]

        # Discover and get method_id.
        result = json.loads(registry.dispatch("method_auto_discover", {
            "description": "numpy mean",
            "context": "compute mean",
        }))
        method_id = result["method_id"]

        # Method should be in catalog.
        method = catalog.get(method_id)
        assert method is not None
        assert method["handler_code"] is not None
        assert method["tool_schema"] is not None

        # Should survive a new registry + register_discovery_tools cycle.
        registry2 = ToolRegistry()
        components = {
            "catalog": catalog,
            "searcher": discovery_env["searcher"],
            "builder": discovery_env["builder"],
            "dep_manager": discovery_env["dep_mgr"],
        }
        register_discovery_tools(registry2, components)

        # Re-verify with new registry.
        verify_result = json.loads(registry2.dispatch("method_verify", {
            "method_id": method_id,
        }))
        assert verify_result["valid"] is True
