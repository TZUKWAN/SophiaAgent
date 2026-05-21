"""Tests for NVivo-style CodingProject and CodingTree.

Covers:
1. 3-level coding tree (employment > flexible employment > platform economy)
2. Two coders, Kappa > 0 when they agree
3. Memo attached to a node
4. Saturation curve shows decreasing trend
5. Data persistence (save and reload)
"""
import json
import os
import shutil
import tempfile

import pytest

from sophia.research.qualitative import CodingProject, CodingTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(result: str) -> dict:
    return json.loads(result)


@pytest.fixture
def project():
    """Fresh CodingProject instance (no store, so _final returns plain JSON)."""
    return CodingProject()


@pytest.fixture
def workspace(tmp_path):
    """Temporary workspace directory for persistence tests."""
    return str(tmp_path)


# ===================================================================
# 1. 3-level coding tree
# ===================================================================

class TestCodingTree:

    def test_three_level_hierarchy(self):
        """Build: Root > employment > flexible employment > platform economy."""
        tree = CodingTree()
        l1 = tree.create_node("employment", color="#FF0000")
        l2 = tree.create_node("flexible employment", parent_id=l1["id"], color="#00FF00")
        l3 = tree.create_node("platform economy", parent_id=l2["id"], color="#0000FF")

        # Verify all three levels exist
        assert len(tree.list_nodes()) == 3

        # Verify hierarchy via tree export
        exported = tree.to_tree()
        assert exported["name"] == "Root"
        assert len(exported["children"]) == 1
        assert exported["children"][0]["name"] == "employment"
        assert len(exported["children"][0]["children"]) == 1
        assert exported["children"][0]["children"][0]["name"] == "flexible employment"
        assert len(exported["children"][0]["children"][0]["children"]) == 1
        assert exported["children"][0]["children"][0]["children"][0]["name"] == "platform economy"

    def test_create_node_attributes(self):
        tree = CodingTree()
        node = tree.create_node("test code", color="#ABCDEF", description="A test node")
        assert node["name"] == "test code"
        assert node["color"] == "#ABCDEF"
        assert node["description"] == "A test node"
        assert "id" in node
        assert "created_at" in node
        assert "modified_at" in node

    def test_delete_node_cascades(self):
        tree = CodingTree()
        l1 = tree.create_node("parent")
        l2 = tree.create_node("child", parent_id=l1["id"])
        assert len(tree.list_nodes()) == 2
        tree.delete_node(l1["id"])
        assert len(tree.list_nodes()) == 0

    def test_rename_node(self):
        tree = CodingTree()
        node = tree.create_node("old name")
        renamed = tree.rename_node(node["id"], "new name")
        assert renamed["name"] == "new name"
        assert tree.get_node(node["id"])["name"] == "new name"

    def test_delete_root_raises(self):
        tree = CodingTree()
        with pytest.raises(ValueError, match="Cannot delete the root"):
            tree.delete_node("root")

    def test_find_by_name(self):
        tree = CodingTree()
        tree.create_node("alpha")
        tree.create_node("beta")
        tree.create_node("alpha")  # duplicate name
        found = tree.find_by_name("alpha")
        assert len(found) == 2

    def test_tree_roundtrip(self):
        tree = CodingTree()
        a = tree.create_node("A")
        b = tree.create_node("B", parent_id=a["id"])
        exported = tree.to_tree()
        restored = CodingTree.from_tree(exported)
        assert len(restored.list_nodes()) == 2
        assert restored.get_node(a["id"])["name"] == "A"
        assert restored.get_node(b["id"])["name"] == "B"


# ===================================================================
# 2. Two coders, Kappa > 0 when they agree
# ===================================================================

class TestReliabilityReport:

    def test_kappa_positive_when_coders_agree(self, project):
        """Two coders code the same text segments with the same codes.
        Kappa should be > 0 (ideally 1.0 for perfect agreement)."""
        texts = [
            "The rise of platform economy has transformed employment patterns.",
            "Flexible work arrangements are becoming increasingly common.",
            "Digital platforms create new forms of precarious employment.",
        ]
        # Create project
        res = _parse(project.create_project({
            "project_name": "Employment Study",
            "texts": texts,
        }))
        pid = res["project_id"]

        # Build coding tree: employment > flexible employment > platform economy
        l1 = _parse(project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "employment",
        }))["node"]
        l2 = _parse(project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "flexible employment",
            "parent_id": l1["id"],
        }))["node"]
        l3 = _parse(project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "platform economy",
            "parent_id": l2["id"],
        }))["node"]

        # Both coders code the same segments with same codes
        code_id_emp = l1["id"]
        code_id_flex = l2["id"]
        code_id_plat = l3["id"]

        # Coder 1 assignments
        for code_id, ti, start, end in [
            (code_id_emp, 0, 13, 29),
            (code_id_plat, 0, 13, 29),
            (code_id_flex, 1, 0, 26),
            (code_id_emp, 2, 0, 16),
            (code_id_plat, 2, 17, 50),
        ]:
            _parse(project.assign_code({
                "project_id": pid, "code_id": code_id,
                "coder_id": "coder1", "text_index": ti,
                "start": start, "end": end,
            }))

        # Coder 2: same codes on same segments (perfect agreement)
        for code_id, ti, start, end in [
            (code_id_emp, 0, 13, 29),
            (code_id_plat, 0, 13, 29),
            (code_id_flex, 1, 0, 26),
            (code_id_emp, 2, 0, 16),
            (code_id_plat, 2, 17, 50),
        ]:
            _parse(project.assign_code({
                "project_id": pid, "code_id": code_id,
                "coder_id": "coder2", "text_index": ti,
                "start": start, "end": end,
            }))

        report = _parse(project.reliability_report({
            "project_id": pid,
            "coder1_id": "coder1",
            "coder2_id": "coder2",
        }))
        assert "error" not in report
        assert report["kappa"] > 0
        assert report["agreement_rate"] > 0
        assert "confusion_matrix" in report

    def test_kappa_perfect_agreement(self, project):
        """Simpler test: both coders assign identical code to same segment."""
        _parse(project.create_project({
            "project_name": "Simple Kappa Test",
            "texts": ["Some text to code."],
        }))
        # Get the first project_id from list
        listing = _parse(project.list_projects({}))
        pid = listing["projects"][0]["project_id"]

        node = _parse(project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "Code A",
        }))["node"]

        # Both coders assign the same code
        for coder in ["c1", "c2"]:
            _parse(project.assign_code({
                "project_id": pid, "code_id": node["id"],
                "coder_id": coder, "text_index": 0, "start": 0, "end": 9,
            }))

        report = _parse(project.reliability_report({
            "project_id": pid, "coder1_id": "c1", "coder2_id": "c2",
        }))
        assert report["kappa"] == 1.0
        assert report["kappa_interpretation"] == "almost perfect"


# ===================================================================
# 3. Memo attached to a node
# ===================================================================

class TestMemo:

    def test_add_memo_to_node(self, project):
        res = _parse(project.create_project({
            "project_name": "Memo Test",
            "texts": ["Interview excerpt about work."],
        }))
        pid = res["project_id"]

        node = _parse(project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "Work Theme",
        }))["node"]

        memo_res = _parse(project.add_memo({
            "project_id": pid,
            "code_id": node["id"],
            "content": "This code captures all mentions of work-related themes including employment type and job satisfaction.",
        }))
        assert "error" not in memo_res
        assert memo_res["memo_id"]
        assert memo_res["memo"]["content"].startswith("This code captures")

        # Verify memo is stored on the project
        proj = project._projects[pid]
        assert node["id"] in proj["memos"]
        assert len(proj["memos"][node["id"]]) == 1

    def test_multiple_memos_on_same_node(self, project):
        res = _parse(project.create_project({"project_name": "Multi Memo"}))
        pid = res["project_id"]

        node = _parse(project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "Node",
        }))["node"]

        for i in range(3):
            _parse(project.add_memo({
                "project_id": pid,
                "code_id": node["id"],
                "content": f"Memo number {i+1}",
            }))

        proj = project._projects[pid]
        assert len(proj["memos"][node["id"]]) == 3


# ===================================================================
# 4. Saturation curve shows decreasing trend
# ===================================================================

class TestSaturationCurve:

    def test_saturation_decreasing_trend(self, project):
        """Simulate coding where initial texts introduce many new codes
        and later texts mostly use existing codes."""
        res = _parse(project.create_project({
            "project_name": "Saturation Test",
            "texts": [
                "Text about employment.",
                "Text about education.",
                "Text about health.",
                "Text about employment again.",
                "Text about employment and education.",
                "Text about employment once more.",
                "Text about health and education.",
                "Text about employment.",
            ],
        }))
        pid = res["project_id"]

        # Create codes
        codes = {}
        for name in ["employment", "education", "health"]:
            node = _parse(project.edit_tree({
                "project_id": pid, "action": "add", "node_name": name,
            }))["node"]
            codes[name] = node["id"]

        # Simulate coding: early texts introduce new codes, later repeat
        # Text 0: employment (new)
        # Text 1: education (new)
        # Text 2: health (new)
        # Text 3: employment (repeat)
        # Text 4: employment + education (repeat)
        # Text 5: employment (repeat)
        # Text 6: health + education (repeat)
        # Text 7: employment (repeat)
        assignments = [
            (0, "employment"),
            (1, "education"),
            (2, "health"),
            (3, "employment"),
            (4, "employment"),
            (4, "education"),
            (5, "employment"),
            (6, "health"),
            (6, "education"),
            (7, "employment"),
        ]

        for text_idx, code_name in assignments:
            _parse(project.assign_code({
                "project_id": pid,
                "code_id": codes[code_name],
                "coder_id": "coder1",
                "text_index": text_idx,
                "start": 0,
                "end": 10,
            }))

        sat = _parse(project.saturation_curve({
            "project_id": pid,
        }))
        assert "error" not in sat
        assert sat["total_unique_codes"] == 3
        assert sat["trend"] == "decreasing"

        curve = sat["curve"]
        # First few should have new_codes > 0, last few should have 0
        first_half_new = sum(c["new_codes"] for c in curve[: len(curve) // 2])
        second_half_new = sum(c["new_codes"] for c in curve[len(curve) // 2 :])
        assert first_half_new > second_half_new

        # Marginal rate should generally decrease
        assert curve[0]["marginal_rate"] >= curve[-1]["marginal_rate"]

    def test_saturation_by_text_unit(self, project):
        res = _parse(project.create_project({
            "project_name": "Text Saturation",
            "texts": ["A", "B", "C", "D"],
        }))
        pid = res["project_id"]

        codes = {}
        for name in ["X", "Y", "Z"]:
            node = _parse(project.edit_tree({
                "project_id": pid, "action": "add", "node_name": name,
            }))["node"]
            codes[name] = node["id"]

        # Text 0: X (new), Text 1: Y (new), Text 2: X+Z (Z new), Text 3: X (repeat)
        for ti, code_name in [(0, "X"), (1, "Y"), (2, "X"), (2, "Z"), (3, "X")]:
            _parse(project.assign_code({
                "project_id": pid, "code_id": codes[code_name],
                "coder_id": "c1", "text_index": ti, "start": 0, "end": 1,
            }))

        sat = _parse(project.saturation_curve({
            "project_id": pid, "unit_type": "text",
        }))
        assert sat["total_unique_codes"] == 3
        assert sat["unit_type"] == "text"


# ===================================================================
# 5. Data persistence (save and reload)
# ===================================================================

class TestPersistence:

    def test_save_and_reload(self, project, workspace):
        """Create project, save to disk, load in new instance."""
        # Create project with workspace
        res = _parse(project.create_project({
            "project_name": "Persistence Test",
            "workspace": workspace,
            "texts": ["First interview text.", "Second interview text."],
            "description": "A test for persistence",
        }))
        pid = res["project_id"]

        # Add tree nodes
        l1 = _parse(project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "employment",
        }))["node"]
        l2 = _parse(project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "flexible employment",
            "parent_id": l1["id"],
        }))["node"]

        # Add assignment
        _parse(project.assign_code({
            "project_id": pid, "code_id": l1["id"],
            "coder_id": "coder1", "text_index": 0, "start": 0, "end": 5,
        }))

        # Add memo
        _parse(project.add_memo({
            "project_id": pid, "code_id": l1["id"],
            "content": "Key theme in interviews",
        }))

        # Verify file was saved
        save_dir = os.path.join(workspace, ".sophia", "coding_projects")
        assert os.path.exists(save_dir)
        files = os.listdir(save_dir)
        assert any(f.startswith("proj_") and f.endswith(".json") for f in files)

        # Reload in a fresh CodingProject instance
        fresh_project = CodingProject()
        load_res = _parse(fresh_project.load_project({
            "project_id": pid,
            "workspace": workspace,
        }))
        assert "error" not in load_res
        assert load_res["project_name"] == "Persistence Test"
        assert load_res["n_texts"] == 2
        assert load_res["n_nodes"] == 2
        assert load_res["n_assignments"] == 1
        assert load_res["n_memos"] == 1

        # Verify the tree structure is intact
        tree_res = _parse(fresh_project.get_tree({"project_id": pid}))
        tree = tree_res["tree"]
        assert len(tree["children"]) == 1
        assert tree["children"][0]["name"] == "employment"
        assert len(tree["children"][0]["children"]) == 1
        assert tree["children"][0]["children"][0]["name"] == "flexible employment"

        # Can continue coding after reload
        new_node = _parse(fresh_project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "platform economy",
            "parent_id": l2["id"],
        }))
        assert new_node["node"]["name"] == "platform economy"


# ===================================================================
# Additional edge-case tests
# ===================================================================

class TestEdgeCases:

    def test_create_project_no_name_error(self, project):
        res = _parse(project.create_project({"project_name": ""}))
        assert "error" in res

    def test_edit_tree_invalid_project(self, project):
        res = _parse(project.edit_tree({
            "project_id": "nonexistent", "action": "add", "node_name": "X",
        }))
        assert "error" in res

    def test_edit_tree_unknown_action(self, project):
        res = _parse(project.create_project({"project_name": "test"}))
        pid = res["project_id"]
        res2 = _parse(project.edit_tree({
            "project_id": pid, "action": "invalid", "node_name": "X",
        }))
        assert "error" in res2

    def test_assign_code_invalid_text_index(self, project):
        res = _parse(project.create_project({
            "project_name": "t", "texts": ["hello"],
        }))
        pid = res["project_id"]
        node = _parse(project.edit_tree({
            "project_id": pid, "action": "add", "node_name": "C",
        }))["node"]
        res2 = _parse(project.assign_code({
            "project_id": pid, "code_id": node["id"],
            "coder_id": "c1", "text_index": 5, "start": 0, "end": 3,
        }))
        assert "error" in res2

    def test_add_memo_nonexistent_code(self, project):
        res = _parse(project.create_project({"project_name": "t"}))
        pid = res["project_id"]
        res2 = _parse(project.add_memo({
            "project_id": pid, "code_id": "fake_id", "content": "memo",
        }))
        assert "error" in res2

    def test_reliability_no_assignments(self, project):
        res = _parse(project.create_project({"project_name": "t"}))
        pid = res["project_id"]
        res2 = _parse(project.reliability_report({
            "project_id": pid, "coder1_id": "c1", "coder2_id": "c2",
        }))
        assert "error" in res2

    def test_list_projects(self, project):
        _parse(project.create_project({"project_name": "P1"}))
        _parse(project.create_project({"project_name": "P2"}))
        listing = _parse(project.list_projects({}))
        assert len(listing["projects"]) == 2
