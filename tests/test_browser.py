"""Tests for sophia.browser.BrowserTools and register_browser_tools."""

import json
import pytest

from sophia.browser import BrowserTools, HAS_PLAYWRIGHT, register_browser_tools
from sophia.tools.registry import ToolRegistry


class TestHasPlaywright:
    """Tests for HAS_PLAYWRIGHT detection flag."""

    def test_is_boolean(self):
        assert isinstance(HAS_PLAYWRIGHT, bool)

    def test_in_test_env_likely_false(self):
        # In a typical test environment playwright is not installed
        # We just check the flag exists and is usable
        if not HAS_PLAYWRIGHT:
            assert HAS_PLAYWRIGHT is False


class TestBrowserToolsNavigate:
    """Tests for BrowserTools.navigate without playwright."""

    def test_navigate_without_playwright_raises(self):
        if HAS_PLAYWRIGHT:
            pytest.skip("playwright is installed, skipping no-playwright test")
        bt = BrowserTools()
        with pytest.raises(RuntimeError, match="playwright not installed"):
            bt.navigate("https://example.com")

    def test_extract_without_playwright_raises(self):
        if HAS_PLAYWRIGHT:
            pytest.skip("playwright is installed, skipping no-playwright test")
        bt = BrowserTools()
        with pytest.raises(RuntimeError, match="playwright not installed"):
            bt.extract("https://example.com")

    def test_screenshot_without_playwright_raises(self):
        if HAS_PLAYWRIGHT:
            pytest.skip("playwright is installed, skipping no-playwright test")
        bt = BrowserTools()
        with pytest.raises(RuntimeError, match="playwright not installed"):
            bt.screenshot("https://example.com")


class TestRegisterBrowserTools:
    """Tests for register_browser_tools."""

    def test_registers_three_tools(self):
        registry = ToolRegistry()
        register_browser_tools(registry)
        tools = registry.list_tools()
        assert "browser_navigate" in tools
        assert "browser_extract" in tools
        assert "browser_screenshot" in tools

    def test_tool_schemas_valid(self):
        registry = ToolRegistry()
        register_browser_tools(registry)
        schemas = registry.get_schemas()

        tool_names = [s["function"]["name"] for s in schemas]
        assert "browser_navigate" in tool_names
        assert "browser_extract" in tool_names
        assert "browser_screenshot" in tool_names

    def test_navigate_dispatch_graceful_error_without_playwright(self):
        if HAS_PLAYWRIGHT:
            pytest.skip("playwright is installed, skipping no-playwright test")
        registry = ToolRegistry()
        register_browser_tools(registry)

        result_json = registry.dispatch("browser_navigate", {"url": "https://example.com"})
        result = json.loads(result_json)
        assert "error" in result

    def test_extract_dispatch_graceful_error_without_playwright(self):
        if HAS_PLAYWRIGHT:
            pytest.skip("playwright is installed, skipping no-playwright test")
        registry = ToolRegistry()
        register_browser_tools(registry)

        result_json = registry.dispatch("browser_extract", {"url": "https://example.com"})
        result = json.loads(result_json)
        assert "error" in result

    def test_screenshot_dispatch_graceful_error_without_playwright(self):
        if HAS_PLAYWRIGHT:
            pytest.skip("playwright is installed, skipping no-playwright test")
        registry = ToolRegistry()
        register_browser_tools(registry)

        result_json = registry.dispatch("browser_screenshot", {"url": "https://example.com"})
        result = json.loads(result_json)
        assert "error" in result

    def test_navigate_schema_has_required_url(self):
        registry = ToolRegistry()
        register_browser_tools(registry)
        schemas = registry.get_schemas()
        nav_schema = [s for s in schemas if s["function"]["name"] == "browser_navigate"][0]
        assert "url" in nav_schema["function"]["parameters"]["required"]

    def test_extract_schema_optional_selector(self):
        registry = ToolRegistry()
        register_browser_tools(registry)
        schemas = registry.get_schemas()
        ext_schema = [s for s in schemas if s["function"]["name"] == "browser_extract"][0]
        # selector is optional
        assert "selector" in ext_schema["function"]["parameters"]["properties"]
        required = ext_schema["function"]["parameters"].get("required", [])
        assert "selector" not in required


class TestBrowserToolsClose:
    """Tests for BrowserTools.close."""

    def test_close_without_browser(self):
        bt = BrowserTools()
        # Should be a no-op when nothing was opened
        bt.close()
        assert bt._browser is None
        assert bt._playwright is None

    def test_close_idempotent(self):
        bt = BrowserTools()
        bt.close()
        bt.close()
        assert bt._browser is None
