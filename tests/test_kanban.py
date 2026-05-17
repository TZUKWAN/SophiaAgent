"""Tests for Kanban Board."""
import json
from sophia.kanban import KanbanBoard, register_kanban_tools
from sophia.tools.registry import ToolRegistry


def _make_board(tmp_path):
    return KanbanBoard(str(tmp_path / "test.db"))


class TestKanbanBoard:
    def test_create_card(self, tmp_path):
        board = _make_board(tmp_path)
        card = board.create_card("s1", "Research AI ethics")
        assert card["title"] == "Research AI ethics"
        assert card["status"] == "backlog"

    def test_move_card(self, tmp_path):
        board = _make_board(tmp_path)
        card = board.create_card("s1", "Task")
        moved = board.move_card(card["id"], "in_progress")
        assert moved["status"] == "in_progress"

    def test_move_invalid_status(self, tmp_path):
        board = _make_board(tmp_path)
        card = board.create_card("s1", "Task")
        result = board.move_card(card["id"], "invalid_status")
        assert result is None

    def test_update_card(self, tmp_path):
        board = _make_board(tmp_path)
        card = board.create_card("s1", "Old title")
        updated = board.update_card(card["id"], title="New title")
        assert updated["title"] == "New title"

    def test_get_board(self, tmp_path):
        board = _make_board(tmp_path)
        board.create_card("s1", "T1", status="backlog")
        board.create_card("s1", "T2", status="todo")
        board.create_card("s1", "T3", status="done")
        b = board.get_board("s1")
        assert len(b["backlog"]) == 1
        assert len(b["todo"]) == 1
        assert len(b["done"]) == 1

    def test_search_cards(self, tmp_path):
        board = _make_board(tmp_path)
        board.create_card("s1", "Research AI", description="About artificial intelligence")
        board.create_card("s1", "Write paper", description="Draft the paper")
        results = board.search_cards("s1", "AI")
        assert len(results) == 1

    def test_delete_card(self, tmp_path):
        board = _make_board(tmp_path)
        card = board.create_card("s1", "Delete me")
        assert board.delete_card(card["id"]) is True
        assert board.get_card(card["id"]) is None

    def test_get_card_not_found(self, tmp_path):
        board = _make_board(tmp_path)
        assert board.get_card("nonexistent") is None

    def test_card_with_tags(self, tmp_path):
        board = _make_board(tmp_path)
        card = board.create_card("s1", "Tagged", tags=["ai", "ethics"])
        assert "ai" in card["tags"]
        assert "ethics" in card["tags"]


class TestKanbanTools:
    def test_create_tool(self, tmp_path):
        board = _make_board(tmp_path)
        reg = ToolRegistry()
        register_kanban_tools(reg, board)
        result = json.loads(reg.dispatch("kanban_create", {"title": "Test card"}))
        assert result["title"] == "Test card"

    def test_move_tool(self, tmp_path):
        board = _make_board(tmp_path)
        reg = ToolRegistry()
        register_kanban_tools(reg, board)
        r = json.loads(reg.dispatch("kanban_create", {"title": "T"}))
        result = json.loads(reg.dispatch("kanban_move", {
            "card_id": r["id"], "status": "todo",
        }))
        assert result["status"] == "todo"

    def test_board_tool(self, tmp_path):
        board = _make_board(tmp_path)
        reg = ToolRegistry()
        register_kanban_tools(reg, board)
        reg.dispatch("kanban_create", {"title": "T1"})
        result = json.loads(reg.dispatch("kanban_board", {}))
        assert result["backlog"] == 1
