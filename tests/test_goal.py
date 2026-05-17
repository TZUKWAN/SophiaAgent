"""Tests for GoalManager and goal tools."""

import json

import pytest

from sophia.goal import Goal, GoalManager, register_goal_tools
from sophia.hooks import HookEvent, HookManager
from sophia.tools.registry import ToolRegistry


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    """Return a temporary database path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def hooks():
    """Create a fresh HookManager."""
    return HookManager()


@pytest.fixture
def goal_mgr(db_path, hooks):
    """Create a GoalManager with a temporary database."""
    return GoalManager(db_path, hooks)


@pytest.fixture
def registry(goal_mgr):
    """Create a ToolRegistry with goal tools registered."""
    reg = ToolRegistry()
    register_goal_tools(reg, goal_mgr)
    return reg


# ── 1. Create a goal ─────────────────────────────────────────


class TestCreateGoal:
    def test_create_basic_goal(self, goal_mgr):
        goal = goal_mgr.create(
            session_id="sess1",
            title="Research topic",
            description="Find relevant papers on LLM agents",
        )
        assert isinstance(goal, Goal)
        assert goal.id is not None
        assert len(goal.id) == 12
        assert goal.session_id == "sess1"
        assert goal.title == "Research topic"
        assert goal.description == "Find relevant papers on LLM agents"
        assert goal.status == "pending"
        assert goal.priority == 3
        assert goal.progress == 0.0
        assert goal.parent_id is None
        assert goal.deadline is None
        assert goal.result is None
        assert goal.created_at is not None
        assert goal.updated_at is not None

    def test_create_with_priority_and_deadline(self, goal_mgr):
        goal = goal_mgr.create(
            session_id="sess1",
            title="Urgent task",
            priority=1,
            deadline="2026-12-31T23:59:59",
        )
        assert goal.priority == 1
        assert goal.deadline == "2026-12-31T23:59:59"

    def test_create_emits_hook(self, goal_mgr, hooks):
        events = []
        hooks.register(
            HookEvent.GOAL_CREATED,
            lambda ctx: (events.append(ctx), ctx)[1],
        )
        goal_mgr.create(session_id="s1", title="Hook test")
        assert len(events) == 1
        assert events[0]["goal"]["title"] == "Hook test"
        assert events[0]["goal"]["status"] == "pending"


# ── 2. Parent-child relationships ────────────────────────────


class TestParentChild:
    def test_create_child_goal(self, goal_mgr):
        parent = goal_mgr.create(session_id="s1", title="Parent goal")
        child = goal_mgr.create(
            session_id="s1",
            title="Child goal",
            parent_id=parent.id,
        )
        assert child.parent_id == parent.id

        children = goal_mgr.get_children(parent.id)
        assert len(children) == 1
        assert children[0].id == child.id
        assert children[0].parent_id == parent.id

    def test_multiple_children(self, goal_mgr):
        parent = goal_mgr.create(session_id="s1", title="Parent")
        c1 = goal_mgr.create(session_id="s1", title="Child 1", parent_id=parent.id)
        c2 = goal_mgr.create(session_id="s1", title="Child 2", parent_id=parent.id)
        c3 = goal_mgr.create(session_id="s1", title="Child 3", parent_id=parent.id)

        children = goal_mgr.get_children(parent.id)
        assert len(children) == 3
        child_ids = {c.id for c in children}
        assert c1.id in child_ids
        assert c2.id in child_ids
        assert c3.id in child_ids

    def test_nested_tree(self, goal_mgr):
        root = goal_mgr.create(session_id="s1", title="Root")
        mid = goal_mgr.create(session_id="s1", title="Mid", parent_id=root.id)
        leaf = goal_mgr.create(session_id="s1", title="Leaf", parent_id=mid.id)

        assert leaf.parent_id == mid.id
        assert mid.parent_id == root.id

        root_children = goal_mgr.get_children(root.id)
        assert len(root_children) == 1
        assert root_children[0].id == mid.id

        mid_children = goal_mgr.get_children(mid.id)
        assert len(mid_children) == 1
        assert mid_children[0].id == leaf.id


# ── 3. Update goal status ────────────────────────────────────


class TestUpdateGoal:
    def test_update_status(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="To update")
        updated = goal_mgr.update(goal.id, status="active")
        assert updated.status == "active"

    def test_update_priority(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="Priority test")
        updated = goal_mgr.update(goal.id, priority=1)
        assert updated.priority == 1

    def test_update_multiple_fields(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="Multi update")
        updated = goal_mgr.update(
            goal.id, status="active", priority=2, description="Updated desc"
        )
        assert updated.status == "active"
        assert updated.priority == 2
        assert updated.description == "Updated desc"

    def test_update_invalid_status_raises(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="Bad status")
        with pytest.raises(ValueError, match="Invalid status"):
            goal_mgr.update(goal.id, status="nonexistent")

    def test_update_emits_hook(self, goal_mgr, hooks):
        events = []
        hooks.register(
            HookEvent.GOAL_UPDATED,
            lambda ctx: (events.append(ctx), ctx)[1],
        )
        goal = goal_mgr.create(session_id="s1", title="Hook update")
        goal_mgr.update(goal.id, status="active")
        assert len(events) == 1
        assert events[0]["goal"]["status"] == "active"

    def test_update_nothing_returns_same(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="No update")
        result = goal_mgr.update(goal.id)
        assert result.id == goal.id
        assert result.status == goal.status

    def test_status_transitions_full_lifecycle(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="Lifecycle")
        assert goal.status == "pending"

        goal = goal_mgr.update(goal.id, status="active")
        assert goal.status == "active"

        goal = goal_mgr.update(goal.id, status="completed")
        assert goal.status == "completed"


# ── 4. Complete a goal ───────────────────────────────────────


class TestCompleteGoal:
    def test_complete_sets_status_and_progress(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="To complete")
        completed = goal_mgr.complete(goal.id, result="Done successfully")
        assert completed.status == "completed"
        assert completed.progress == 1.0
        assert completed.result == "Done successfully"

    def test_complete_emits_hook(self, goal_mgr, hooks):
        events = []
        hooks.register(
            HookEvent.GOAL_COMPLETED,
            lambda ctx: (events.append(ctx), ctx)[1],
        )
        goal = goal_mgr.create(session_id="s1", title="Hook complete")
        goal_mgr.complete(goal.id)
        assert len(events) == 1
        assert events[0]["goal"]["status"] == "completed"
        assert events[0]["goal"]["progress"] == 1.0

    def test_complete_propagates_progress_to_parent(self, goal_mgr):
        parent = goal_mgr.create(session_id="s1", title="Parent")
        c1 = goal_mgr.create(session_id="s1", title="C1", parent_id=parent.id)
        c2 = goal_mgr.create(session_id="s1", title="C2", parent_id=parent.id)

        goal_mgr.complete(c1.id)
        parent = goal_mgr.get(parent.id)
        assert parent.progress == 0.5

        goal_mgr.complete(c2.id)
        parent = goal_mgr.get(parent.id)
        assert parent.progress == 1.0


# ── 5. Fail a goal ───────────────────────────────────────────


class TestFailGoal:
    def test_fail_sets_status_and_reason(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="To fail")
        failed = goal_mgr.fail(goal.id, reason="Timeout exceeded")
        assert failed.status == "failed"
        assert failed.result == "Timeout exceeded"

    def test_fail_emits_hook(self, goal_mgr, hooks):
        events = []
        hooks.register(
            HookEvent.GOAL_FAILED,
            lambda ctx: (events.append(ctx), ctx)[1],
        )
        goal = goal_mgr.create(session_id="s1", title="Hook fail")
        goal_mgr.fail(goal.id, reason="test failure")
        assert len(events) == 1
        assert events[0]["goal"]["status"] == "failed"
        assert events[0]["goal"]["result"] == "test failure"

    def test_fail_propagates_progress_to_parent(self, goal_mgr):
        parent = goal_mgr.create(session_id="s1", title="Parent")
        c1 = goal_mgr.create(session_id="s1", title="C1", parent_id=parent.id)
        c2 = goal_mgr.create(session_id="s1", title="C2", parent_id=parent.id)

        goal_mgr.complete(c1.id)
        parent = goal_mgr.get(parent.id)
        assert parent.progress == 0.5

        goal_mgr.fail(c2.id)
        parent = goal_mgr.get(parent.id)
        # Failed child is not completed, so progress stays 0.5
        assert parent.progress == 0.5


# ── 6. Get tree ──────────────────────────────────────────────


class TestGetTree:
    def test_get_tree_returns_all_session_goals(self, goal_mgr):
        g1 = goal_mgr.create(session_id="s1", title="G1")
        g2 = goal_mgr.create(session_id="s1", title="G2", parent_id=g1.id)
        g3 = goal_mgr.create(session_id="s1", title="G3", parent_id=g1.id)
        # Different session
        goal_mgr.create(session_id="s2", title="Other")

        tree = goal_mgr.get_tree("s1")
        assert len(tree) == 3
        titles = {g.title for g in tree}
        assert titles == {"G1", "G2", "G3"}

    def test_get_tree_empty_session(self, goal_mgr):
        tree = goal_mgr.get_tree("nonexistent")
        assert tree == []


# ── 7. Get active goals ──────────────────────────────────────


class TestGetActive:
    def test_get_active_filters_by_status(self, goal_mgr):
        g1 = goal_mgr.create(session_id="s1", title="Pending")
        g2 = goal_mgr.create(session_id="s1", title="Active one")
        goal_mgr.update(g2.id, status="active")
        g3 = goal_mgr.create(session_id="s1", title="Active two")
        goal_mgr.update(g3.id, status="active")
        g4 = goal_mgr.create(session_id="s1", title="Completed")
        goal_mgr.complete(g4.id)

        active = goal_mgr.get_active("s1")
        assert len(active) == 2
        active_titles = {g.title for g in active}
        assert "Active one" in active_titles
        assert "Active two" in active_titles

    def test_get_active_empty(self, goal_mgr):
        active = goal_mgr.get_active("s1")
        assert active == []


# ── 8. Compute progress from children ────────────────────────


class TestComputeProgress:
    def test_no_children_returns_own_progress(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="Leaf")
        progress = goal_mgr.compute_progress(goal.id)
        assert progress == 0.0

    def test_progress_from_completed_children(self, goal_mgr):
        parent = goal_mgr.create(session_id="s1", title="Parent")
        c1 = goal_mgr.create(session_id="s1", title="C1", parent_id=parent.id)
        c2 = goal_mgr.create(session_id="s1", title="C2", parent_id=parent.id)
        c3 = goal_mgr.create(session_id="s1", title="C3", parent_id=parent.id)

        # No children completed
        progress = goal_mgr.compute_progress(parent.id)
        assert progress == 0.0

        # Complete 1 of 3
        goal_mgr.complete(c1.id)
        progress = goal_mgr.compute_progress(parent.id)
        assert progress == pytest.approx(1.0 / 3.0)

        # Complete 2 of 3
        goal_mgr.complete(c2.id)
        progress = goal_mgr.compute_progress(parent.id)
        assert progress == pytest.approx(2.0 / 3.0)

        # Complete all
        goal_mgr.complete(c3.id)
        progress = goal_mgr.compute_progress(parent.id)
        assert progress == 1.0

    def test_progress_persists_in_db(self, goal_mgr):
        parent = goal_mgr.create(session_id="s1", title="Parent")
        c1 = goal_mgr.create(session_id="s1", title="C1", parent_id=parent.id)
        c2 = goal_mgr.create(session_id="s1", title="C2", parent_id=parent.id)

        goal_mgr.complete(c1.id)
        # Progress is auto-propagated via complete()
        parent = goal_mgr.get(parent.id)
        assert parent.progress == 0.5

    def test_deep_tree_progress_propagation(self, goal_mgr):
        """Verify progress propagates up through multiple levels."""
        root = goal_mgr.create(session_id="s1", title="Root")
        mid = goal_mgr.create(session_id="s1", title="Mid", parent_id=root.id)
        leaf1 = goal_mgr.create(session_id="s1", title="Leaf1", parent_id=mid.id)
        leaf2 = goal_mgr.create(session_id="s1", title="Leaf2", parent_id=mid.id)

        goal_mgr.complete(leaf1.id)

        mid = goal_mgr.get(mid.id)
        assert mid.progress == 0.5

        root = goal_mgr.get(root.id)
        # Root has 1 child (mid) which is 0.5 complete, but compute_progress
        # only counts fully completed children. So root = 0/1 = 0.0
        assert root.progress == 0.0

        goal_mgr.complete(leaf2.id)

        mid = goal_mgr.get(mid.id)
        assert mid.progress == 1.0

        root = goal_mgr.get(root.id)
        # mid is not marked completed (only its children are), so root still 0.0
        assert root.progress == 0.0


# ── 9. Delete goal (cascade) ─────────────────────────────────


class TestDeleteGoal:
    def test_delete_single_goal(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="To delete")
        goal_mgr.delete(goal.id)
        assert goal_mgr.get(goal.id) is None

    def test_delete_cascades_to_children(self, goal_mgr):
        parent = goal_mgr.create(session_id="s1", title="Parent")
        c1 = goal_mgr.create(session_id="s1", title="C1", parent_id=parent.id)
        c2 = goal_mgr.create(session_id="s1", title="C2", parent_id=parent.id)
        grandchild = goal_mgr.create(session_id="s1", title="GC", parent_id=c1.id)

        goal_mgr.delete(parent.id)

        assert goal_mgr.get(parent.id) is None
        assert goal_mgr.get(c1.id) is None
        assert goal_mgr.get(c2.id) is None
        assert goal_mgr.get(grandchild.id) is None

    def test_delete_preserves_other_goals(self, goal_mgr):
        g1 = goal_mgr.create(session_id="s1", title="Keep")
        g2 = goal_mgr.create(session_id="s1", title="Remove")
        goal_mgr.delete(g2.id)
        assert goal_mgr.get(g1.id) is not None
        assert goal_mgr.get(g2.id) is None

    def test_delete_with_propagation(self, goal_mgr):
        parent = goal_mgr.create(session_id="s1", title="Parent")
        c1 = goal_mgr.create(session_id="s1", title="C1", parent_id=parent.id)
        c2 = goal_mgr.create(session_id="s1", title="C2", parent_id=parent.id)

        goal_mgr.complete(c1.id)
        parent = goal_mgr.get(parent.id)
        assert parent.progress == 0.5

        goal_mgr.delete_with_propagation(c2.id)
        parent = goal_mgr.get(parent.id)
        # Now only 1 child (c1, completed), so progress should be 1.0
        assert parent.progress == 1.0


# ── 10. Tool registration and dispatch ───────────────────────


class TestGoalTools:
    def test_goal_create_tool(self, registry):
        result = json.loads(registry.dispatch("goal_create", {
            "session_id": "s1",
            "title": "Tool created goal",
            "description": "Via tool dispatch",
            "priority": 2,
        }))
        assert "error" not in result
        assert result["title"] == "Tool created goal"
        assert result["session_id"] == "s1"
        assert result["status"] == "pending"
        assert result["priority"] == 2
        assert result["description"] == "Via tool dispatch"

    def test_goal_create_missing_session(self, registry):
        result = json.loads(registry.dispatch("goal_create", {
            "title": "No session",
        }))
        assert "error" in result

    def test_goal_create_missing_title(self, registry):
        result = json.loads(registry.dispatch("goal_create", {
            "session_id": "s1",
        }))
        assert "error" in result

    def test_goal_update_tool(self, registry):
        created = json.loads(registry.dispatch("goal_create", {
            "session_id": "s1", "title": "Original",
        }))
        goal_id = created["id"]

        result = json.loads(registry.dispatch("goal_update", {
            "goal_id": goal_id,
            "status": "active",
            "priority": 1,
        }))
        assert result["status"] == "active"
        assert result["priority"] == 1

    def test_goal_update_missing_goal_id(self, registry):
        result = json.loads(registry.dispatch("goal_update", {
            "status": "active",
        }))
        assert "error" in result

    def test_goal_update_no_fields(self, registry):
        created = json.loads(registry.dispatch("goal_create", {
            "session_id": "s1", "title": "Test",
        }))
        result = json.loads(registry.dispatch("goal_update", {
            "goal_id": created["id"],
        }))
        assert "error" in result

    def test_goal_list_tool(self, registry):
        registry.dispatch("goal_create", {
            "session_id": "s1", "title": "G1",
        })
        registry.dispatch("goal_create", {
            "session_id": "s1", "title": "G2",
        })
        registry.dispatch("goal_create", {
            "session_id": "s2", "title": "Other session",
        })

        result = json.loads(registry.dispatch("goal_list", {
            "session_id": "s1",
        }))
        assert result["total"] == 2
        assert len(result["goals"]) == 2

    def test_goal_list_missing_session(self, registry):
        result = json.loads(registry.dispatch("goal_list", {}))
        assert "error" in result

    def test_goal_complete_tool(self, registry):
        created = json.loads(registry.dispatch("goal_create", {
            "session_id": "s1", "title": "To finish",
        }))
        goal_id = created["id"]

        result = json.loads(registry.dispatch("goal_complete", {
            "goal_id": goal_id,
            "result": "All done",
        }))
        assert result["status"] == "completed"
        assert result["progress"] == 1.0
        assert result["result"] == "All done"

    def test_goal_complete_missing_id(self, registry):
        result = json.loads(registry.dispatch("goal_complete", {}))
        assert "error" in result

    def test_goal_fail_tool(self, registry):
        created = json.loads(registry.dispatch("goal_create", {
            "session_id": "s1", "title": "Will fail",
        }))
        goal_id = created["id"]

        result = json.loads(registry.dispatch("goal_fail", {
            "goal_id": goal_id,
            "reason": "Timeout",
        }))
        assert result["status"] == "failed"
        assert result["result"] == "Timeout"

    def test_goal_fail_missing_id(self, registry):
        result = json.loads(registry.dispatch("goal_fail", {}))
        assert "error" in result

    def test_all_tools_registered(self, registry):
        tools = registry.list_tools()
        assert "goal_create" in tools
        assert "goal_update" in tools
        assert "goal_list" in tools
        assert "goal_complete" in tools
        assert "goal_fail" in tools

    def test_dispatch_unknown_tool(self, registry):
        result = json.loads(registry.dispatch("goal_nonexistent", {}))
        assert "error" in result

    def test_get_schemas(self, registry):
        schemas = registry.get_schemas()
        goal_tools = [
            s for s in schemas
            if s["function"]["name"].startswith("goal_")
        ]
        assert len(goal_tools) == 5
        for s in goal_tools:
            assert "parameters" in s["function"]
            assert "properties" in s["function"]["parameters"]


# ── Edge cases ────────────────────────────────────────────────


class TestEdgeCases:
    def test_get_nonexistent_goal(self, goal_mgr):
        assert goal_mgr.get("nonexistent_id") is None

    def test_get_children_of_leaf(self, goal_mgr):
        leaf = goal_mgr.create(session_id="s1", title="Leaf")
        children = goal_mgr.get_children(leaf.id)
        assert children == []

    def test_cancel_goal(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="Cancel me")
        cancelled = goal_mgr.cancel(goal.id)
        assert cancelled.status == "cancelled"

    def test_cancel_propagates_to_parent(self, goal_mgr):
        parent = goal_mgr.create(session_id="s1", title="Parent")
        c1 = goal_mgr.create(session_id="s1", title="C1", parent_id=parent.id)
        c2 = goal_mgr.create(session_id="s1", title="C2", parent_id=parent.id)

        goal_mgr.complete(c1.id)
        goal_mgr.cancel(c2.id)

        parent = goal_mgr.get(parent.id)
        # 1 completed out of 2 (cancelled doesn't count as completed)
        assert parent.progress == 0.5

    def test_cross_session_isolation(self, goal_mgr):
        s1_goal = goal_mgr.create(session_id="s1", title="S1 Goal")
        s2_goal = goal_mgr.create(session_id="s2", title="S2 Goal")

        s1_tree = goal_mgr.get_tree("s1")
        s2_tree = goal_mgr.get_tree("s2")

        assert len(s1_tree) == 1
        assert s1_tree[0].id == s1_goal.id
        assert len(s2_tree) == 1
        assert s2_tree[0].id == s2_goal.id

    def test_complete_with_empty_result(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="No result")
        completed = goal_mgr.complete(goal.id)
        assert completed.result == ""
        assert completed.status == "completed"

    def test_fail_with_empty_reason(self, goal_mgr):
        goal = goal_mgr.create(session_id="s1", title="No reason")
        failed = goal_mgr.fail(goal.id)
        assert failed.result == ""
        assert failed.status == "failed"
