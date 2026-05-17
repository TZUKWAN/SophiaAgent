import json

from sophia.swarm.orchestrator import FilteredToolRegistry
from sophia.tools.registry import ToolRegistry


def test_filtered_registry_exposes_only_allowed_schemas_and_dispatches():
    registry = ToolRegistry()
    registry.register("allowed", "allowed", {"type": "object"}, lambda args: {"ok": True})
    registry.register("blocked", "blocked", {"type": "object"}, lambda args: {"bad": True})
    filtered = FilteredToolRegistry(registry, ["allowed"])
    names = [schema["function"]["name"] for schema in filtered.get_schemas()]
    assert names == ["allowed"]
    assert json.loads(filtered.dispatch("allowed", {}))["ok"] is True
    assert "not allowed" in json.loads(filtered.dispatch("blocked", {}))["error"]
