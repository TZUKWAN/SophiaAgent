"""Browser automation tools for SophiaAgent.

Provides web browsing capabilities via optional playwright dependency.
"""

import json
import logging
from typing import Any, Dict, Optional

from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

HAS_PLAYWRIGHT = False
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    pass


class BrowserTools:
    def __init__(self, workspace: str = ""):
        self.workspace = workspace
        self._playwright = None
        self._browser = None

    def _ensure_browser(self):
        if not HAS_PLAYWRIGHT:
            raise RuntimeError("playwright not installed. Install with: pip install playwright && playwright install")
        if self._browser is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)

    def navigate(self, url: str) -> Dict:
        self._ensure_browser()
        page = self._browser.new_page()
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=30000)
            title = page.title()
            content = page.content()
            return {
                "url": url,
                "title": title,
                "status": response.status if response else None,
                "content_length": len(content),
                "content_preview": content[:2000],
            }
        except Exception as e:
            return {"error": str(e), "url": url}
        finally:
            page.close()

    def extract(self, url: str, selector: str = "body") -> Dict:
        self._ensure_browser()
        page = self._browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            element = page.query_selector(selector)
            if element:
                text = element.inner_text()
                return {"url": url, "selector": selector, "text": text[:5000]}
            return {"error": f"Selector '{selector}' not found", "url": url}
        except Exception as e:
            return {"error": str(e), "url": url}
        finally:
            page.close()

    def screenshot(self, url: str, path: str = "") -> Dict:
        self._ensure_browser()
        page = self._browser.new_page()
        try:
            page.goto(url, wait_until="load", timeout=30000)
            import os
            if not path:
                path = os.path.join(self.workspace, f"screenshot_{hash(url) % 10000}.png")
            page.screenshot(path=path)
            return {"url": url, "screenshot_path": path}
        except Exception as e:
            return {"error": str(e), "url": url}
        finally:
            page.close()

    def close(self):
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None


def register_browser_tools(registry: ToolRegistry, workspace: str = ""):
    browser = BrowserTools(workspace)

    def _navigate(args):
        result = browser.navigate(args["url"])
        return json.dumps(result, ensure_ascii=False)

    registry.register(
        "browser_navigate",
        "Navigate to a URL and get page content preview",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to"},
            },
            "required": ["url"],
        },
        _navigate,
    )

    def _extract(args):
        result = browser.extract(args["url"], args.get("selector", "body"))
        return json.dumps(result, ensure_ascii=False)

    registry.register(
        "browser_extract",
        "Extract text content from a webpage using CSS selector",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to extract from"},
                "selector": {"type": "string", "description": "CSS selector (default: body)"},
            },
            "required": ["url"],
        },
        _extract,
    )

    def _screenshot(args):
        result = browser.screenshot(args["url"], args.get("path", ""))
        return json.dumps(result, ensure_ascii=False)

    registry.register(
        "browser_screenshot",
        "Take a screenshot of a webpage",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to screenshot"},
                "path": {"type": "string", "description": "Output file path"},
            },
            "required": ["url"],
        },
        _screenshot,
    )
