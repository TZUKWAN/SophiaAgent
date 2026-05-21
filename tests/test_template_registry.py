"""Tests for TemplateRegistry and template tools."""

from __future__ import annotations

import json
import pytest

from sophia.prompts.templates.registry import TemplateRegistry
from sophia.tools.registry import ToolRegistry
from sophia.tools.templates import register_template_tools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    """Load templates from the real templates directory."""
    return TemplateRegistry()


@pytest.fixture
def tool_registry(registry):
    """ToolRegistry with template tools registered."""
    tools = ToolRegistry()
    register_template_tools(tools, registry)
    return tools


# ---------------------------------------------------------------------------
# TemplateRegistry tests
# ---------------------------------------------------------------------------

class TestTemplateRegistryLoading:
    def test_loads_all_templates(self, registry):
        templates = registry.list_templates()
        assert len(templates) == 21

    def test_lists_all_disciplines(self, registry):
        disciplines = registry.list_disciplines()
        expected = ["education", "history", "literature", "politics_law", "psychology", "sociology"]
        assert disciplines == expected


class TestTemplateRegistryRetrieval:
    def test_get_history_paper(self, registry):
        tmpl = registry.get_template("history/outline_history_paper")
        assert tmpl is not None
        assert tmpl["name"] == "历史学学术论文大纲"
        assert "outline" in tmpl
        assert "section_prompts" in tmpl
        assert "checklist" in tmpl

    def test_get_literature_theory_frameworks(self, registry):
        tmpl = registry.get_template("literature/theory_frameworks")
        assert tmpl is not None
        assert tmpl["name"] == "文学批评理论框架库"
        frameworks = tmpl["frameworks"]
        assert len(frameworks) == 10
        ids = {f["id"] for f in frameworks}
        expected_ids = {
            "psychoanalysis", "feminism", "postcolonialism", "new_historicism",
            "structuralism", "poststructuralism", "reader_response",
            "cultural_materialism", "ecocriticism", "deconstruction",
        }
        assert ids == expected_ids

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get_template("nonexistent/foo") is None

    def test_list_by_discipline(self, registry):
        history_templates = registry.list_templates(discipline="history")
        assert len(history_templates) == 4
        for t in history_templates:
            assert t["discipline"] == "history"

        psych_templates = registry.list_templates(discipline="psychology")
        assert len(psych_templates) == 3

    def test_list_no_filter_returns_all(self, registry):
        all_templates = registry.list_templates()
        assert len(all_templates) == 21


class TestTemplateRegistryOutlineHelpers:
    def test_get_outline(self, registry):
        outline = registry.get_outline("history/outline_history_paper")
        assert outline == ["问题的提出", "史料来源与批判", "考证分析", "讨论", "结论"]

    def test_get_outline_missing_returns_empty(self, registry):
        assert registry.get_outline("nonexistent/foo") == []

    def test_get_section_prompt(self, registry):
        prompt = registry.get_section_prompt("history/outline_history_paper", "考证分析")
        assert "考证" in prompt
        assert len(prompt) > 20

    def test_get_section_prompt_missing(self, registry):
        assert registry.get_section_prompt("nonexistent/foo", "考证分析") == ""
        assert registry.get_section_prompt("history/outline_history_paper", "不存在") == ""

    def test_get_checklist(self, registry):
        checklist = registry.get_checklist("history/outline_history_paper")
        assert len(checklist) >= 5
        assert any("史料" in item for item in checklist)

    def test_get_checklist_missing_returns_empty(self, registry):
        assert registry.get_checklist("nonexistent/foo") == []


class TestTemplateRegistryRecommendation:
    def test_recommend_by_keyword(self, registry):
        results = registry.recommend_templates("我想写一篇关于明清档案考证的历史论文")
        assert len(results) > 0
        top = results[0]
        assert "template_id" in top
        assert "match_score" in top
        assert "reason" in top
        # History templates should rank high
        history_results = [r for r in results if r["discipline"] == "history"]
        assert len(history_results) > 0

    def test_recommend_with_discipline_filter(self, registry):
        results = registry.recommend_templates("教育行动研究", discipline="education")
        assert len(results) > 0
        for r in results:
            assert r["discipline"] == "education"

    def test_recommend_no_match(self, registry):
        results = registry.recommend_templates("xyzabc123 unrelated topic")
        # Should still potentially match something via description or tags,
        # but if truly no match, score stays 0 and results may be empty.
        # We just assert it doesn't crash.
        assert isinstance(results, list)

    def test_recommend_returns_limited_results(self, registry):
        results = registry.recommend_templates("历史论文")
        assert len(results) <= 10


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

class TestTemplateListTool:
    def test_list_all(self, tool_registry):
        result = tool_registry.dispatch("template_list", {})
        data = json.loads(result)
        assert data["count"] == 21
        assert len(data["disciplines"]) == 6
        assert len(data["templates"]) == 21

    def test_list_by_discipline(self, tool_registry):
        result = tool_registry.dispatch("template_list", {"discipline": "sociology"})
        data = json.loads(result)
        assert data["count"] == 4
        for t in data["templates"]:
            assert t["discipline"] == "sociology"


class TestTemplateGetTool:
    def test_get_valid(self, tool_registry):
        result = tool_registry.dispatch("template_get", {"template_id": "politics_law/outline_legal_argument"})
        data = json.loads(result)
        assert "error" not in data
        assert data["name"] == "法律论证结构模板"
        assert data["template_id"] == "politics_law/outline_legal_argument"
        assert "outline" in data
        assert "checklist" in data

    def test_get_missing(self, tool_registry):
        result = tool_registry.dispatch("template_get", {"template_id": "foo/bar"})
        data = json.loads(result)
        assert "error" in data

    def test_get_no_id(self, tool_registry):
        result = tool_registry.dispatch("template_get", {})
        data = json.loads(result)
        assert "error" in data


class TestTemplateRecommendTool:
    def test_recommend(self, tool_registry):
        result = tool_registry.dispatch("template_recommend", {
            "research_question": "心理学实验设计中的APA格式报告",
        })
        data = json.loads(result)
        assert "error" not in data
        assert data["research_question"] == "心理学实验设计中的APA格式报告"
        assert data["count"] > 0
        top = data["recommendations"][0]
        assert "match_score" in top

    def test_recommend_with_filter(self, tool_registry):
        result = tool_registry.dispatch("template_recommend", {
            "research_question": "论文",
            "discipline": "literature",
            "top_n": 2,
        })
        data = json.loads(result)
        assert data["count"] <= 2
        for r in data["recommendations"]:
            assert r["discipline"] == "literature"

    def test_recommend_no_question(self, tool_registry):
        result = tool_registry.dispatch("template_recommend", {})
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# Integration: verify template JSON validity
# ---------------------------------------------------------------------------

class TestTemplateJsonValidity:
    def test_all_templates_are_valid_json(self, registry):
        for tid in [t["template_id"] for t in registry.list_templates()]:
            tmpl = registry.get_template(tid)
            assert tmpl is not None
            assert "name" in tmpl
            # Verify we can serialize back to JSON
            serialized = json.dumps(tmpl, ensure_ascii=False)
            assert len(serialized) > 0
