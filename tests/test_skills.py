"""Tests for Skill Store."""
import json
from sophia.skills import SkillManager
from sophia.tools.registry import ToolRegistry


def _make_mgr(tmp_path):
    return SkillManager(str(tmp_path / "test.db"))


SIMPLE_SKILL = {
    "name": "Echo Skill",
    "description": "Echoes back the input",
    "tool_schemas": [{
        "name": "echo_tool",
        "description": "Echo input",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
    }],
    "handler_code": "def handle(args):\n    import json\n    return json.dumps({'echo': args.get('text', '')})\n",
}


class TestSkillManager:
    def test_install(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        sid = mgr.install(SIMPLE_SKILL)
        assert sid == "echo_skill"

    def test_uninstall(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        sid = mgr.install(SIMPLE_SKILL)
        assert mgr.uninstall(sid) is True
        assert mgr.get_skill(sid) is None

    def test_list_skills(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.install(SIMPLE_SKILL)
        skills = mgr.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "Echo Skill"

    def test_list_by_category(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        mgr.install({**SIMPLE_SKILL, "category": "test"})
        skills = mgr.list_skills(category="test")
        assert len(skills) == 1
        skills_other = mgr.list_skills(category="other")
        assert len(skills_other) == 0

    def test_get_skill(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        sid = mgr.install(SIMPLE_SKILL)
        skill = mgr.get_skill(sid)
        assert skill is not None
        assert skill["name"] == "Echo Skill"

    def test_get_skill_not_found(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        assert mgr.get_skill("nonexistent") is None

    def test_register_skill_tools(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        sid = mgr.install(SIMPLE_SKILL)
        reg = ToolRegistry()
        count = mgr.register_skill_tools(sid, reg)
        assert count == 1
        assert "echo_tool" in reg.list_tools()

    def test_register_skill_tools_dispatch(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        sid = mgr.install(SIMPLE_SKILL)
        reg = ToolRegistry()
        mgr.register_skill_tools(sid, reg)
        result = json.loads(reg.dispatch("echo_tool", {"text": "hello"}))
        assert result["echo"] == "hello"

    def test_register_nonexistent_skill(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        reg = ToolRegistry()
        count = mgr.register_skill_tools("nonexistent", reg)
        assert count == 0
