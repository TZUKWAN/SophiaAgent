"""Tests for Plugin system."""
from sophia.plugins import PluginInterface, PluginManager
from sophia.tools.registry import ToolRegistry


class MockPlugin(PluginInterface):
    def name(self):
        return "mock_plugin"

    def register(self, registry, **kwargs):
        registry.register(
            "mock_tool", "A mock tool",
            {"type": "object", "properties": {"x": {"type": "integer"}}},
            lambda args: '{"result": "mock"}',
        )


class TestPluginManager:
    def test_register_plugin(self):
        pm = PluginManager()
        p = MockPlugin()
        pm.register_plugin(p)
        assert "mock_plugin" in pm.list_plugins()

    def test_get_plugin(self):
        pm = PluginManager()
        pm.register_plugin(MockPlugin())
        assert pm.get_plugin("mock_plugin") is not None
        assert pm.get_plugin("nonexistent") is None

    def test_list_plugins(self):
        pm = PluginManager()
        pm.register_plugin(MockPlugin())
        assert pm.list_plugins() == ["mock_plugin"]

    def test_register_all(self):
        pm = PluginManager()
        pm.register_plugin(MockPlugin())
        reg = ToolRegistry()
        pm.register_all(reg)
        assert "mock_tool" in reg.list_tools()

    def test_load_from_module(self):
        import types
        mod = types.ModuleType("test_mod")
        mod.MockPlugin = MockPlugin
        pm = PluginManager()
        pm.load_from_module(mod)
        assert "mock_plugin" in pm.list_plugins()

    def test_load_from_nonexistent_dir(self):
        pm = PluginManager()
        result = pm.load_from_directory("/nonexistent/path")
        assert result == []
