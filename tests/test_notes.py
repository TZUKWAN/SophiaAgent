"""Tests for ZettelkastenStore."""

import pytest

from sophia.research.notes import ZettelkastenStore, NOTE_TYPES


@pytest.fixture
def store(tmp_workspace):
    return ZettelkastenStore(tmp_workspace)


class TestNoteCreate:
    def test_create_concept_note(self, store):
        result = store.create(
            title="社会资本",
            content="社会资本是指个体或群体通过社会网络获取资源的能力。",
            note_type="concept",
            tags=["社会学", "理论"],
        )
        assert result["success"] is True
        assert result["note"]["note_type"] == "concept"
        assert result["note"]["title"] == "社会资本"

    def test_create_evidence_note(self, store):
        result = store.create(
            title="实证发现：社会资本与幸福感",
            content="来源：Smith et al. (2020)\n数据：N=1250\n方法：OLS回归\n结论：社会资本显著正向影响幸福感",
            note_type="evidence",
            source_type="paper",
            source_id="smith2020",
        )
        assert result["success"] is True
        assert result["note"]["note_type"] == "evidence"
        assert result["note"]["source_type"] == "paper"

    def test_create_comment_note(self, store):
        result = store.create(
            title="对社会资本测量的质疑",
            content="现有研究对社会资本的测量过于简化，忽视了结构性维度。",
            note_type="comment",
        )
        assert result["success"] is True
        assert result["note"]["note_type"] == "comment"

    def test_invalid_note_type(self, store):
        result = store.create(
            title="Test",
            content="Content",
            note_type="invalid",
        )
        assert "error" in result

    def test_create_with_links(self, store):
        note1 = store.create(title="Note 1", content="Content 1", note_type="concept")
        note1_id = note1["note"]["id"]

        result = store.create(
            title="Note 2",
            content="Content 2 referencing [[note_id]]",
            note_type="concept",
            links=[note1_id],
        )
        assert result["success"] is True
        assert note1_id in result["note"]["links"]

        # Check backlink
        note1_reloaded = store.get(note1_id)
        assert result["note"]["id"] in note1_reloaded["backlinks"]


class TestNoteSearch:
    def test_search_by_query(self, store):
        store.create(title="社会资本理论", content="关于社会资本的讨论", note_type="concept")
        store.create(title="幸福感研究", content="关于幸福感的讨论", note_type="concept")

        results = store.search(query="社会资本")
        assert len(results) == 1
        assert results[0]["title"] == "社会资本理论"

    def test_search_by_tags(self, store):
        store.create(title="Note A", content="Content", note_type="concept", tags=["tag1", "tag2"])
        store.create(title="Note B", content="Content", note_type="concept", tags=["tag2"])

        results = store.search(query="", tags=["tag1"])
        assert len(results) == 1
        assert results[0]["title"] == "Note A"

    def test_search_by_note_type(self, store):
        store.create(title="Concept", content="Content", note_type="concept")
        store.create(title="Evidence", content="Content", note_type="evidence")

        results = store.search(query="", note_type="evidence")
        assert len(results) == 1
        assert results[0]["title"] == "Evidence"

    def test_search_by_linked_to(self, store):
        note1 = store.create(title="Source", content="Content", note_type="concept")
        note1_id = note1["note"]["id"]
        note2 = store.create(title="Linked", content="Content", note_type="concept", links=[note1_id])

        results = store.search(query="", linked_to=note1_id)
        assert len(results) == 1
        assert results[0]["title"] == "Linked"


class TestNoteUpdate:
    def test_update_title(self, store):
        result = store.create(title="Old Title", content="Content", note_type="concept")
        note_id = result["note"]["id"]

        update_result = store.update(note_id=note_id, title="New Title")
        assert update_result["success"] is True
        assert update_result["note"]["title"] == "New Title"

    def test_update_links(self, store):
        note1 = store.create(title="Note 1", content="Content", note_type="concept")
        note1_id = note1["note"]["id"]
        note2 = store.create(title="Note 2", content="Content", note_type="concept")
        note2_id = note2["note"]["id"]

        store.update(note_id=note1_id, links=[note2_id])

        note2_reloaded = store.get(note2_id)
        assert note1_id in note2_reloaded["backlinks"]

    def test_update_inline_links(self, store):
        note1 = store.create(title="Note 1", content="Content", note_type="concept")
        note1_id = note1["note"]["id"]

        store.update(note_id=note1_id, content=f"See also [[{note1_id}]]")
        note1_reloaded = store.get(note1_id)
        assert note1_id in note1_reloaded["links"]


class TestNoteDelete:
    def test_delete_note(self, store):
        result = store.create(title="To Delete", content="Content", note_type="concept")
        note_id = result["note"]["id"]

        delete_result = store.delete(note_id)
        assert delete_result["success"] is True
        assert store.get(note_id) is None

    def test_delete_removes_backlinks(self, store):
        note1 = store.create(title="Note 1", content="Content", note_type="concept")
        note1_id = note1["note"]["id"]
        note2 = store.create(title="Note 2", content="Content", note_type="concept", links=[note1_id])
        note2_id = note2["note"]["id"]

        store.delete(note2_id)
        note1_reloaded = store.get(note1_id)
        assert note2_id not in note1_reloaded["backlinks"]


class TestLinkGraph:
    def test_empty_graph(self, store):
        graph = store.get_link_graph()
        assert graph["node_count"] == 0
        assert graph["edge_count"] == 0

    def test_graph_with_links(self, store):
        note1 = store.create(title="Note 1", content="Content", note_type="concept")
        note1_id = note1["note"]["id"]
        note2 = store.create(title="Note 2", content="Content", note_type="concept", links=[note1_id])

        graph = store.get_link_graph()
        assert graph["node_count"] == 2
        assert graph["edge_count"] == 2  # forward + back


class TestFromPaperElements:
    def test_auto_generate_evidence(self, store):
        elements = {
            "research_question": ["Does X affect Y?"],
            "core_arguments": ["X positively affects Y"],
            "methods": ["Regression"],
            "data_sources": ["Panel data"],
            "main_findings": ["Significant effect"],
            "limitations": ["Endogeneity"],
            "theoretical_framework": ["Theory Z"],
            "sample_size": "5000",
        }
        result = store.from_paper_elements(
            elements=elements,
            paper_title="Test Paper",
            paper_id="test2024",
        )
        assert result["success"] is True
        assert result["note"]["note_type"] == "evidence"
        assert "Test Paper" in result["note"]["tags"]
        assert "5000" in result["note"]["content"]

    def test_from_paper_with_chinese(self, store):
        elements = {
            "research_question": ["社会资本如何影响居民幸福感？"],
            "main_findings": ["社会资本显著正向影响幸福感"],
            "sample_size": "1250",
        }
        result = store.from_paper_elements(
            elements=elements,
            paper_title="社会资本与幸福感研究",
        )
        assert result["success"] is True
        assert "社会资本" in result["note"]["content"]
