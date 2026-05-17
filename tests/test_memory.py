"""Tests for Memory system."""
import json
from sophia.hooks import HookManager
from sophia.memory import MemoryManager, register_memory_tools
from sophia.tools.registry import ToolRegistry


def _make_mgr(tmp_path, hooks=None):
    db = str(tmp_path / "test.db")
    return MemoryManager(db, hooks or HookManager())


class TestMemoryManager:
    def test_store(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mid = mgr.store("s1", "topic", "Digital humanities research")
        assert mid > 0

    def test_recall_by_keyword(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.store("s1", "topic1", "Digital humanities research")
        mgr.store("s1", "topic2", "Machine learning applications")
        results = mgr.recall("s1", "digital")
        assert len(results) == 1
        assert "digital" in results[0].content.lower()

    def test_recall_by_category(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.store("s1", "p1", "Prefers Chinese", category="preference")
        mgr.store("s1", "f1", "AI is growing", category="fact")
        results = mgr.recall("s1", "", category="preference")
        assert all(r.category == "preference" for r in results)

    def test_get_entry(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mid = mgr.store("s1", "k1", "content here")
        entry = mgr.get(mid)
        assert entry is not None
        assert entry.key == "k1"

    def test_update_entry(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mid = mgr.store("s1", "k1", "old content")
        updated = mgr.update(mid, content="new content")
        assert updated.content == "new content"

    def test_delete_entry(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mid = mgr.store("s1", "k1", "bye")
        assert mgr.delete(mid) is True
        assert mgr.get(mid) is None

    def test_access_count_increments(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mid = mgr.store("s1", "k1", "searchable content")
        mgr.recall("s1", "searchable")
        entry = mgr.get(mid)
        assert entry.access_count >= 1

    def test_build_context(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.store("s1", "pref", "Chinese language", category="preference")
        mgr.store("s1", "fact", "AI growth", category="fact")
        ctx = mgr.build_context("s1", "Chinese")
        assert "[Memory Context]" in ctx
        assert "Chinese" in ctx

    def test_search_across_fields(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.store("s1", "network_analysis", "Using Gephi for visualization", tags=["network", "gephi"])
        results = mgr.search("s1", "gephi")
        assert len(results) == 1

    def test_get_all(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.store("s1", "a", "a")
        mgr.store("s1", "b", "b")
        mgr.store("s2", "c", "c")
        all_entries = mgr.get_all("s1")
        assert len(all_entries) == 2

    def test_hook_emission(self, tmp_path):
        hooks = HookManager()
        events = []
        hooks.register("memory.store", lambda ctx: (events.append("store"), ctx)[1])
        mgr = _make_mgr(tmp_path, hooks)
        mgr.store("s1", "k", "v")
        assert "store" in events


class TestMemoryTools:
    def test_store_tool(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        reg = ToolRegistry()
        register_memory_tools(reg, mgr)
        result = json.loads(reg.dispatch("memory_store", {
            "key": "topic", "content": "test content",
        }))
        assert result["action"] == "stored"

    def test_recall_tool(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        reg = ToolRegistry()
        register_memory_tools(reg, mgr)
        reg.dispatch("memory_store", {"key": "k1", "content": "digital humanities"})
        result = json.loads(reg.dispatch("memory_recall", {"query": "digital"}))
        assert len(result) == 1

    def test_search_tool(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        reg = ToolRegistry()
        register_memory_tools(reg, mgr)
        reg.dispatch("memory_store", {"key": "k1", "content": "AI research"})
        result = json.loads(reg.dispatch("memory_search", {"query": "AI"}))
        assert len(result) == 1

    def test_delete_tool(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        reg = ToolRegistry()
        register_memory_tools(reg, mgr)
        r = json.loads(reg.dispatch("memory_store", {"key": "k1", "content": "temp"}))
        result = json.loads(reg.dispatch("memory_delete", {"entry_id": r["id"]}))
        assert result["deleted"] is True
