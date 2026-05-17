"""Plugin system for SophiaAgent extensibility."""
import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name."""

    @abstractmethod
    def register(self, registry: ToolRegistry, **kwargs) -> None:
        """Register tools into the registry."""


class PluginManager:
    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}

    def register_plugin(self, plugin: PluginInterface) -> None:
        self._plugins[plugin.name()] = plugin
        logger.info("Plugin registered: %s", plugin.name())

    def load_from_module(self, module) -> Optional[PluginInterface]:
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (inspect.isclass(obj) and issubclass(obj, PluginInterface)
                    and obj is not PluginInterface):
                plugin = obj()
                self.register_plugin(plugin)
                return plugin
        return None

    def load_from_directory(self, plugin_dir: str) -> List[PluginInterface]:
        loaded = []
        path = Path(plugin_dir)
        if not path.exists():
            return loaded
        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                plugin = self.load_from_module(mod)
                if plugin:
                    loaded.append(plugin)
            except Exception as e:
                logger.warning("Failed to load plugin from %s: %s", py_file, e)
        return loaded

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        return list(self._plugins.keys())

    def register_all(self, registry: ToolRegistry, **kwargs) -> None:
        for plugin in self._plugins.values():
            try:
                plugin.register(registry, **kwargs)
            except Exception as e:
                logger.warning("Plugin %s registration failed: %s", plugin.name(), e)
