"""Self-evolving method discovery system."""

from sophia.research.discovery.method_catalog import MethodCatalog
from sophia.research.discovery.method_searcher import MethodSearcher
from sophia.research.discovery.method_builder import MethodBuilder
from sophia.research.discovery.dependency_manager import DependencyManager
from sophia.research.discovery.register import register_discovery_tools

__all__ = [
    "MethodCatalog",
    "MethodSearcher",
    "MethodBuilder",
    "DependencyManager",
    "register_discovery_tools",
]
