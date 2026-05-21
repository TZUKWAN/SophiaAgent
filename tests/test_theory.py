"""Tests for TheoryMapper: map_theories, trace_concept, compare_schools."""

import json
import os

import pytest

from sophia.research.theory import TheoryMapper


@pytest.fixture
def mapper():
    """Create a TheoryMapper with the built-in knowledge base."""
    return TheoryMapper()


class TestMapTheories:
    """Tests for TheoryMapper.map_theories."""

    def test_map_theories_basic(self, mapper):
        result = mapper.map_theories("社会资本与社区发展")
        assert result["topic"] == "社会资本与社区发展"
        assert "theories" in result
        assert "relations" in result
        assert "recommended" in result
        # Should match social_capital theory
        tids = [t["theory_id"] for t in result["theories"]]
        assert "social_capital" in tids

    def test_map_theories_with_discipline(self, mapper):
        result = mapper.map_theories("教育不平等", discipline="education")
        assert result["topic"] == "教育不平等"
        tids = [t["theory_id"] for t in result["theories"]]
        # Should match education theories
        assert len(tids) > 0

    def test_map_theories_empty_topic(self, mapper):
        result = mapper.map_theories("")
        assert "error" in result
        assert result["error"] == "topic is required"

    def test_map_theories_relevance_scores(self, mapper):
        result = mapper.map_theories("社会资本")
        for t in result["theories"]:
            assert 0 <= t["relevance_score"] <= 1.0
            assert "relation_to_topic" in t

    def test_map_theories_relations(self, mapper):
        result = mapper.map_theories("社会资本 网络 信任")
        relations = result["relations"]
        for r in relations:
            assert r["type"] in {"extends", "contradicts", "complements", "influences"}
            assert 0 <= r["strength"] <= 1.0


class TestExportTheoryMap:
    """Tests for TheoryMapper.export_theory_map."""

    def test_export_mermaid(self, mapper):
        data = mapper.map_theories("社会资本")
        output = mapper.export_theory_map(data, format="mermaid")
        assert "graph TD" in output
        assert "TOPIC" in output

    def test_export_tikz(self, mapper):
        data = mapper.map_theories("社会资本")
        output = mapper.export_theory_map(data, format="tikz")
        assert "\\begin{tikzpicture}" in output
        assert "\\end{tikzpicture}" in output

    def test_export_dot(self, mapper):
        data = mapper.map_theories("社会资本")
        output = mapper.export_theory_map(data, format="dot")
        assert "digraph TheoryMap" in output
        assert "}" in output

    def test_export_invalid_format(self, mapper):
        data = mapper.map_theories("社会资本")
        output = mapper.export_theory_map(data, format="invalid")
        assert "Error" in output


class TestTraceConcept:
    """Tests for TheoryMapper.trace_concept."""

    def test_trace_precomputed_social_capital(self, mapper):
        result = mapper.trace_concept("社会资本")
        assert result["concept"] == "社会资本"
        assert len(result["evolution_stages"]) > 0
        assert len(result["current_debates"]) > 0
        assert len(result["cross_disciplinary_usage"]) > 0

    def test_trace_precomputed_involution(self, mapper):
        result = mapper.trace_concept("内卷")
        assert result["concept"] == "内卷"
        assert len(result["evolution_stages"]) >= 3

    def test_trace_precomputed_digital_labor(self, mapper):
        result = mapper.trace_concept("数字劳动")
        assert result["concept"] == "数字劳动"
        assert "传播学" in result["cross_disciplinary_usage"]

    def test_trace_precomputed_cultural_capital(self, mapper):
        result = mapper.trace_concept("文化资本")
        assert result["concept"] == "文化资本"

    def test_trace_precomputed_governance(self, mapper):
        result = mapper.trace_concept("治理")
        assert result["concept"] == "治理"

    def test_trace_precomputed_globalization(self, mapper):
        result = mapper.trace_concept("全球化")
        assert result["concept"] == "全球化"

    def test_trace_precomputed_identity(self, mapper):
        result = mapper.trace_concept("身份认同")
        assert result["concept"] == "身份认同"

    def test_trace_precomputed_post_truth(self, mapper):
        result = mapper.trace_concept("后真相")
        assert result["concept"] == "后真相"

    def test_trace_unknown_concept_no_llm(self, mapper):
        result = mapper.trace_concept("一个不存在的概念XYZ")
        assert "note" in result
        assert "暂无预计算历史" in result["note"]

    def test_trace_empty_concept(self, mapper):
        result = mapper.trace_concept("")
        assert "error" in result

    def test_trace_concept_stages_structure(self, mapper):
        result = mapper.trace_concept("社会资本")
        for stage in result["evolution_stages"]:
            assert "period" in stage
            assert "discipline" in stage
            assert "definition" in stage
            assert "key_authors" in stage
            assert "seminal_works" in stage
            assert "shift_description" in stage


class TestCompareSchools:
    """Tests for TheoryMapper.compare_schools."""

    def test_compare_two_theories(self, mapper):
        result = mapper.compare_schools(["social_capital", "field_theory"])
        assert "comparison_table" in result
        assert "dimensions" in result
        assert "markdown" in result
        assert len(result["comparison_table"]) > 0
        assert len(result["dimensions"]) > 0

    def test_compare_multiple_theories(self, mapper):
        result = mapper.compare_schools(["social_capital", "cultural_capital", "field_theory"])
        table = result["comparison_table"]
        # Check that all theory_ids appear as keys in rows
        for row in table:
            assert "social_capital" in row or row["dimension"] == "维度"
            assert "field_theory" in row or row["dimension"] == "维度"

    def test_compare_empty_list(self, mapper):
        result = mapper.compare_schools([])
        assert "error" in result

    def test_compare_invalid_theory(self, mapper):
        result = mapper.compare_schools(["nonexistent_theory"])
        assert "error" in result

    def test_compare_markdown_output(self, mapper):
        result = mapper.compare_schools(["social_capital", "field_theory"])
        md = result["markdown"]
        assert "|" in md
        assert "社会资本" in md or "social_capital" in md

    def test_compare_dimensions_coverage(self, mapper):
        result = mapper.compare_schools(["social_capital", "field_theory"])
        dims = result["dimensions"]
        expected = [
            "本体论假设",
            "认识论立场",
            "方法论偏好",
            "核心概念",
            "代表学者",
            "经典文献",
            "主要批评",
            "适用场景",
            "局限性",
        ]
        for d in expected:
            assert d in dims


class TestTheoryMapperKB:
    """Tests for knowledge base loading."""

    def test_kb_loaded(self, mapper):
        theories = mapper.list_theories()
        assert len(theories) >= 30

    def test_disciplines(self, mapper):
        discs = mapper.list_disciplines()
        assert "sociology" in discs
        assert "education" in discs
        assert "politics" in discs
        assert "psychology" in discs
        assert "communication" in discs

    def test_get_theory(self, mapper):
        t = mapper.get_theory("social_capital")
        assert t is not None
        assert t["theory_id"] == "social_capital"
        assert "name_cn" in t
        assert "founders" in t

    def test_list_theories_by_discipline(self, mapper):
        soc = mapper.list_theories("sociology")
        assert len(soc) >= 6
        edu = mapper.list_theories("education")
        assert len(edu) >= 5
        pol = mapper.list_theories("politics")
        assert len(pol) >= 5
        psy = mapper.list_theories("psychology")
        assert len(psy) >= 4
        com = mapper.list_theories("communication")
        assert len(com) >= 4
