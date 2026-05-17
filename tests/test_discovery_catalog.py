"""Tests for MethodCatalog: persistent method index."""
import json
import os
import tempfile
import pytest

from sophia.research.discovery.method_catalog import MethodCatalog


@pytest.fixture
def catalog(tmp_path):
    """Create a fresh MethodCatalog with a temporary DB."""
    db_path = str(tmp_path / "test_catalog.db")
    return MethodCatalog(db_path)


@pytest.fixture
def catalog_no_seed(tmp_path):
    """Create a catalog and verify it has seeded data."""
    db_path = str(tmp_path / "test_catalog2.db")
    cat = MethodCatalog(db_path)
    return cat


class TestMethodCatalogInit:
    def test_creates_database_file(self, tmp_path):
        db_path = str(tmp_path / "new.db")
        cat = MethodCatalog(db_path)
        assert os.path.exists(db_path)

    def test_seeds_builtin_methods(self, catalog):
        stats = catalog.get_stats()
        assert stats["total"] >= 77  # All 77 built-in tools

    def test_builtin_methods_are_installed(self, catalog):
        methods = catalog.list_methods(status="installed")
        assert len(methods) >= 77
        for m in methods:
            assert m["status"] == "installed"
            assert m["source"] == "builtin"

    def test_seeding_is_idempotent(self, tmp_path):
        db_path = str(tmp_path / "idem.db")
        cat1 = MethodCatalog(db_path)
        count1 = cat1.get_stats()["total"]
        cat2 = MethodCatalog(db_path)
        count2 = cat2.get_stats()["total"]
        assert count1 == count2


class TestMethodCatalogAdd:
    def test_add_new_method(self, catalog):
        method = {
            "name": "Test Method",
            "category": "test",
            "description": "A test method",
            "status": "known",
            "keywords": ["test", "unit"],
        }
        mid = catalog.add(method)
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_add_with_custom_id(self, catalog):
        method = {
            "id": "my_custom_id",
            "name": "Custom",
            "category": "test",
        }
        mid = catalog.add(method)
        assert mid == "my_custom_id"

    def test_add_with_auto_id(self, catalog):
        method = {"name": "Auto ID", "category": "test"}
        mid = catalog.add(method)
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_add_with_dependencies(self, catalog):
        method = {
            "name": "Dep Method",
            "category": "test",
            "dependencies": ["numpy", "pandas"],
        }
        mid = catalog.add(method)
        retrieved = catalog.get(mid)
        assert retrieved["dependencies"] == ["numpy", "pandas"]

    def test_add_with_handler_code(self, catalog):
        code = "def handle(args):\n    return 'ok'"
        method = {
            "name": "Handler Method",
            "category": "test",
            "handler_code": code,
        }
        mid = catalog.add(method)
        retrieved = catalog.get(mid)
        assert retrieved["handler_code"] == code

    def test_add_with_tool_schema(self, catalog):
        schema = {"description": "test", "parameters": {"type": "object"}}
        method = {
            "name": "Schema Method",
            "category": "test",
            "tool_schema": schema,
        }
        mid = catalog.add(method)
        retrieved = catalog.get(mid)
        assert retrieved["tool_schema"] == schema


class TestMethodCatalogGet:
    def test_get_existing_method(self, catalog):
        method = catalog.get("ttest")
        assert method is not None
        assert method["name"] == "T-Test"
        assert method["category"] == "statistics"

    def test_get_nonexistent_returns_none(self, catalog):
        assert catalog.get("nonexistent_xyz_123") is None

    def test_get_by_tool_name(self, catalog):
        method = catalog.get_by_tool("research_ttest")
        assert method is not None
        assert method["id"] == "ttest"

    def test_get_by_tool_nonexistent(self, catalog):
        assert catalog.get_by_tool("nonexistent_tool") is None


class TestMethodCatalogSearch:
    def test_search_by_name(self, catalog):
        results = catalog.search("T-Test")
        assert len(results) >= 1
        names = [r["name"] for r in results]
        assert any("T-Test" in n for n in names)

    def test_search_by_description(self, catalog):
        results = catalog.search("Bayesian")
        assert len(results) >= 1

    def test_search_with_category_filter(self, catalog):
        results = catalog.search("test", category="statistics")
        assert len(results) >= 1
        for r in results:
            assert r["category"] == "statistics"

    def test_search_returns_empty_for_no_match(self, catalog):
        results = catalog.search("zzzzz_nonexistent_method_xyz")
        assert len(results) == 0

    def test_search_case_insensitive(self, catalog):
        # SQL LIKE is case-insensitive by default in SQLite for ASCII
        results_lower = catalog.search("anova")
        results_upper = catalog.search("ANOVA")
        assert len(results_lower) == len(results_upper)


class TestMethodCatalogUpdate:
    def test_update_name(self, catalog):
        method = {
            "id": "updateme",
            "name": "Original",
            "category": "test",
        }
        catalog.add(method)
        success = catalog.update("updateme", name="Updated")
        assert success is True
        updated = catalog.get("updateme")
        assert updated["name"] == "Updated"

    def test_update_status(self, catalog):
        method = {"id": "status_test", "name": "Status", "category": "test", "status": "known"}
        catalog.add(method)
        catalog.update("status_test", status="installed")
        updated = catalog.get("status_test")
        assert updated["status"] == "installed"

    def test_update_nonexistent_returns_false(self, catalog):
        assert catalog.update("nonexistent_xyz", name="Nope") is False

    def test_update_sets_timestamp(self, catalog):
        method = {"id": "ts_test", "name": "TS", "category": "test"}
        catalog.add(method)
        original = catalog.get("ts_test")
        catalog.update("ts_test", name="TS Updated")
        updated = catalog.get("ts_test")
        assert updated["updated_at"] >= original["updated_at"]

    def test_update_ignores_invalid_fields(self, catalog):
        method = {"id": "ignore_test", "name": "Ignore", "category": "test"}
        catalog.add(method)
        result = catalog.update("ignore_test", invalid_field="value")
        assert result is False


class TestMethodCatalogList:
    def test_list_all(self, catalog):
        methods = catalog.list_methods()
        assert len(methods) >= 77

    def test_list_by_category(self, catalog):
        methods = catalog.list_methods(category="statistics")
        assert len(methods) >= 11
        for m in methods:
            assert m["category"] == "statistics"

    def test_list_by_status(self, catalog):
        methods = catalog.list_methods(status="installed")
        assert len(methods) >= 77

    def test_list_by_source(self, catalog):
        methods = catalog.list_methods(source="builtin")
        assert len(methods) >= 77
        for m in methods:
            assert m["source"] == "builtin"

    def test_list_combined_filters(self, catalog):
        methods = catalog.list_methods(category="causal", status="installed")
        assert len(methods) >= 8
        for m in methods:
            assert m["category"] == "causal"
            assert m["status"] == "installed"


class TestMethodCatalogActivate:
    def test_activate_skips_methods_without_handler(self, catalog):
        # Builtin methods have no handler_code, so activate_all returns 0
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()
        count = catalog.activate_all(reg)
        assert count == 0

    def test_activate_registers_methods_with_handler(self, catalog):
        from sophia.tools.registry import ToolRegistry
        reg = ToolRegistry()

        handler_code = (
            "import json\n"
            "def handle(args):\n"
            "    return json.dumps({'result': 'test'})\n"
        )
        schema = {
            "description": "Test tool",
            "parameters": {"type": "object", "properties": {"data": {"type": "array"}}},
        }
        catalog.add({
            "id": "activatable",
            "name": "Activatable",
            "category": "test",
            "status": "installed",
            "tool_name": "research_activatable",
            "handler_code": handler_code,
            "tool_schema": schema,
        })

        count = catalog.activate_all(reg)
        assert count == 1
        assert "research_activatable" in reg.list_tools()


class TestMethodCatalogStats:
    def test_stats_has_by_category(self, catalog):
        stats = catalog.get_stats()
        assert "by_category" in stats
        assert "statistics" in stats["by_category"]
        assert stats["by_category"]["statistics"] >= 11

    def test_stats_has_by_status(self, catalog):
        stats = catalog.get_stats()
        assert "by_status" in stats
        assert "installed" in stats["by_status"]

    def test_stats_total_matches(self, catalog):
        stats = catalog.get_stats()
        total_from_parts = sum(stats["by_category"].values())
        assert stats["total"] == total_from_parts
