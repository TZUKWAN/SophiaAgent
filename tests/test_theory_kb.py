"""Tests for theory knowledge base JSON format and coverage."""

import json
import os

import pytest


KB_PATH = os.path.join(os.path.dirname(__file__), "..", "sophia", "research", "data", "theory_kb.json")


@pytest.fixture
def kb():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestKBCoverage:
    """Verify KB has sufficient coverage."""

    def test_at_least_30_theories(self, kb):
        theories = kb.get("theories", [])
        assert len(theories) >= 30, f"Expected at least 30 theories, got {len(theories)}"

    def test_at_least_5_disciplines(self, kb):
        disciplines = kb.get("disciplines", [])
        assert len(disciplines) >= 5, f"Expected at least 5 disciplines, got {len(disciplines)}"

    def test_sociology_coverage(self, kb):
        theories = [t for t in kb["theories"] if t["discipline"] == "sociology"]
        names = {t["name_cn"] for t in theories}
        required = {"社会资本理论", "场域理论", "社会交换理论", "符号互动论", "结构功能主义", "冲突理论"}
        missing = required - names
        assert not missing, f"Missing sociology theories: {missing}"

    def test_education_coverage(self, kb):
        theories = [t for t in kb["theories"] if t["discipline"] == "education"]
        names = {t["name_cn"] for t in theories}
        required = {"建构主义学习理论", "自我效能理论", "社会认知理论", "文化再生产理论"}
        missing = required - names
        assert not missing, f"Missing education theories: {missing}"

    def test_politics_coverage(self, kb):
        theories = [t for t in kb["theories"] if t["discipline"] == "politics"]
        names = {t["name_cn"] for t in theories}
        required = {"国家能力理论", "制度主义", "民主化理论", "公共选择理论"}
        missing = required - names
        assert not missing, f"Missing politics theories: {missing}"

    def test_psychology_coverage(self, kb):
        theories = [t for t in kb["theories"] if t["discipline"] == "psychology"]
        names = {t["name_cn"] for t in theories}
        required = {"依恋理论", "认知失调理论", "社会认同理论", "自我决定理论"}
        missing = required - names
        assert not missing, f"Missing psychology theories: {missing}"

    def test_communication_coverage(self, kb):
        theories = [t for t in kb["theories"] if t["discipline"] == "communication"]
        names = {t["name_cn"] for t in theories}
        required = {"议程设置理论", "框架理论", "使用与满足理论", "数字鸿沟理论"}
        missing = required - names
        assert not missing, f"Missing communication theories: {missing}"


class TestKBFormat:
    """Verify each theory entry has required fields."""

    def test_all_theories_have_required_fields(self, kb):
        required_fields = [
            "theory_id",
            "name_en",
            "name_cn",
            "discipline",
            "founders",
            "key_concepts",
            "core_propositions",
            "related_theories",
            "competing_theories",
            "methodological_implications",
        ]
        for t in kb["theories"]:
            for field in required_fields:
                assert field in t, f"Theory {t.get('theory_id', '?')} missing field: {field}"

    def test_theory_id_unique(self, kb):
        ids = [t["theory_id"] for t in kb["theories"]]
        assert len(ids) == len(set(ids)), f"Duplicate theory_ids found"

    def test_disciplines_valid(self, kb):
        valid = set(kb.get("disciplines", []))
        for t in kb["theories"]:
            assert t["discipline"] in valid, f"Theory {t['theory_id']} has invalid discipline: {t['discipline']}"

    def test_founders_is_list(self, kb):
        for t in kb["theories"]:
            assert isinstance(t["founders"], list), f"Theory {t['theory_id']}: founders must be a list"
            assert len(t["founders"]) > 0, f"Theory {t['theory_id']}: founders must not be empty"

    def test_key_concepts_is_list(self, kb):
        for t in kb["theories"]:
            assert isinstance(t["key_concepts"], list), f"Theory {t['theory_id']}: key_concepts must be a list"
            assert len(t["key_concepts"]) > 0, f"Theory {t['theory_id']}: key_concepts must not be empty"

    def test_core_propositions_is_list(self, kb):
        for t in kb["theories"]:
            assert isinstance(t["core_propositions"], list), f"Theory {t['theory_id']}: core_propositions must be a list"
            assert len(t["core_propositions"]) > 0, f"Theory {t['theory_id']}: core_propositions must not be empty"

    def test_related_theories_is_list(self, kb):
        for t in kb["theories"]:
            assert isinstance(t["related_theories"], list), f"Theory {t['theory_id']}: related_theories must be a list"

    def test_competing_theories_is_list(self, kb):
        for t in kb["theories"]:
            assert isinstance(t["competing_theories"], list), f"Theory {t['theory_id']}: competing_theories must be a list"

    def test_methodological_implications_is_string(self, kb):
        for t in kb["theories"]:
            assert isinstance(t["methodological_implications"], str), f"Theory {t['theory_id']}: methodological_implications must be a string"
            assert len(t["methodological_implications"]) > 0, f"Theory {t['theory_id']}: methodological_implications must not be empty"

    def test_classic_works_present(self, kb):
        for t in kb["theories"]:
            assert "classic_works" in t, f"Theory {t['theory_id']} missing classic_works"
            assert isinstance(t["classic_works"], list), f"Theory {t['theory_id']}: classic_works must be a list"
            assert len(t["classic_works"]) > 0, f"Theory {t['theory_id']}: classic_works must not be empty"

    def test_name_cn_not_empty(self, kb):
        for t in kb["theories"]:
            assert t["name_cn"] and isinstance(t["name_cn"], str), f"Theory {t['theory_id']}: name_cn must be a non-empty string"

    def test_name_en_not_empty(self, kb):
        for t in kb["theories"]:
            assert t["name_en"] and isinstance(t["name_en"], str), f"Theory {t['theory_id']}: name_en must be a non-empty string"
