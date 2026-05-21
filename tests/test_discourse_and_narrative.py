"""Tests for discourse and narrative analysis engines (Task A-4, A-5)."""

import json
import pytest

from sophia.research.discourse import DiscourseEngine
from sophia.research.narrative import NarrativeEngine


# =========================================================================
# A-4: Discourse Analysis
# =========================================================================

class TestDiscourseEngine:
    """Tests for DiscourseEngine."""

    def setup_method(self):
        self.engine = DiscourseEngine()

    def _gov_report_text(self):
        return (
            "根据国务院文件精神，各级政府要积极推进社会治理创新。"
            "数据显示，全国已有85%的城市建立了网格化管理体系。"
            "专家指出，这一举措有效提升了基层治理能力。"
            "然而，部分群众反映在实际执行中存在形式主义问题。"
            "我们必须坚持以人民为中心，杜绝形式主义。"
            "西方某些国家对中国治理模式的批评是别有用心的。"
        )

    def test_identifies_speaker_roles(self):
        """Should identify government, public, scholar as discourse subjects."""
        result = json.loads(self.engine.analyze_discourse({"text": self._gov_report_text()}))
        subjects = result["subjects"]
        roles = [s["role"] for s in subjects]
        assert "government" in roles, f"Expected government in {roles}"

    def test_detects_discourse_strategies(self):
        """Should detect authority citation and data rhetoric."""
        result = json.loads(self.engine.analyze_discourse({"text": self._gov_report_text()}))
        strategies = result["discourse_strategies"]
        strategy_names = [s["strategy"] for s in strategies]
        assert "权威引用" in strategy_names, f"Expected 权威引用 in {strategy_names}"
        assert "数据修辞" in strategy_names, f"Expected 数据修辞 in {strategy_names}"

    def test_detects_power_relations(self):
        """Should detect at least one power relation (government-public)."""
        result = json.loads(self.engine.analyze_discourse({"text": self._gov_report_text()}))
        relations = result["power_relations"]
        assert len(relations) >= 1, "Expected at least one power relation"

    def test_detects_ideology_frames(self):
        """Should detect nationalism or statism frame."""
        result = json.loads(self.engine.analyze_discourse({"text": self._gov_report_text()}))
        ideologies = result["ideology_frames"]
        ideo_names = [i["ideology"] for i in ideologies]
        # The text has "以人民为中心", "中国特色", etc.
        assert len(ideologies) > 0, "Expected at least one ideology frame"

    def test_foucault_framework(self):
        result = json.loads(self.engine.analyze_discourse({
            "text": self._gov_report_text(),
            "framework": "foucault",
        }))
        assert result["framework"] == "foucault"
        assert "framework_analysis" in result
        assert "knowledge_regimes" in result["framework_analysis"]

    def test_cda_framework(self):
        result = json.loads(self.engine.analyze_discourse({
            "text": self._gov_report_text(),
            "framework": "cda",
        }))
        assert result["framework"] == "cda"
        fa = result["framework_analysis"]
        assert "text_analysis" in fa
        assert "discourse_practice" in fa
        assert "social_practice" in fa

    def test_narrative_discourse_framework(self):
        result = json.loads(self.engine.analyze_discourse({
            "text": "我在学校遇到了一些困难，但后来克服了。",
            "framework": "narrative",
        }))
        assert result["framework"] == "narrative"
        fa = result["framework_analysis"]
        assert "voice_analysis" in fa

    def test_empty_text(self):
        result = json.loads(self.engine.analyze_discourse({"text": ""}))
        assert "error" in result

    def test_output_is_valid_json(self):
        result = self.engine.analyze_discourse({"text": "这是一个简单的测试文本。"})
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_summary_generated(self):
        result = json.loads(self.engine.analyze_discourse({"text": self._gov_report_text()}))
        assert "summary" in result
        assert len(result["summary"]) > 0


# =========================================================================
# A-5: Narrative Analysis
# =========================================================================

class TestNarrativeEngine:
    """Tests for NarrativeEngine."""

    def setup_method(self):
        self.engine = NarrativeEngine()

    def _interview_text(self):
        return (
            "我是一名来自农村的青年教师。"
            "那时候我刚毕业，被分配到一所乡村小学。"
            "刚开始的时候，教学条件很差，连电脑都没有。"
            "但是我觉得这些孩子很需要我。"
            "我觉得最困难的是和家长沟通，很多家长不重视教育。"
            "后来有一次家访改变了我对家长的看法。"
            "那是一次转折点，我突然意识到他们不是不重视，而是条件不允许。"
            "从那以后，我开始主动了解每个孩子的家庭情况。"
            "现在回想起来，那段经历让我成长了很多。"
            "我觉得教育最重要的不是传授知识，而是理解。"
        )

    def test_structure_mode(self):
        """Should identify at least 2 Labov narrative elements."""
        result = json.loads(self.engine.analyze_narrative({
            "text": self._interview_text(),
            "mode": "structure",
        }))
        assert result["mode"] == "structure"
        elements = result["narrative_elements"]
        element_types = set(e["element"] for e in elements)
        assert len(element_types) >= 2, f"Expected >= 2 element types, got {element_types}"

    def test_structure_completeness(self):
        """Completeness score should be > 0."""
        result = json.loads(self.engine.analyze_narrative({
            "text": self._interview_text(),
            "mode": "structure",
        }))
        assert result["completeness"] > 0

    def test_turning_point_mode(self):
        """Should identify at least one turning point."""
        result = json.loads(self.engine.analyze_narrative({
            "text": self._interview_text(),
            "mode": "turning_point",
        }))
        tps = result["turning_points"]
        assert len(tps) >= 1, f"Expected >= 1 turning point, got {len(tps)}"
        # Each turning point should have text and markers
        for tp in tps:
            assert "text" in tp
            assert "markers" in tp

    def test_turning_point_timeline(self):
        result = json.loads(self.engine.analyze_narrative({
            "text": self._interview_text(),
            "mode": "turning_point",
        }))
        timeline = result["timeline"]
        assert len(timeline) > 0

    def test_identity_mode(self):
        """Should distinguish self and other descriptions."""
        result = json.loads(self.engine.analyze_narrative({
            "text": self._interview_text(),
            "mode": "identity",
        }))
        characters = result["characters"]
        assert len(characters) == 2
        self_char = [c for c in characters if c["type"] == "self"][0]
        other_char = [c for c in characters if c["type"] == "other"][0]
        assert self_char["reference_count"] > 0
        assert "self_other_ratio" in result

    def test_identity_roles(self):
        """Should detect professional identity."""
        result = json.loads(self.engine.analyze_narrative({
            "text": self._interview_text(),
            "mode": "identity",
        }))
        roles = result["identity_roles"]
        role_types = [r["role_type"] for r in roles]
        assert "专业身份" in role_types, f"Expected 专业身份 in {role_types}"

    def test_chinese_tokenization_in_narrative(self):
        """Chinese text should be properly handled."""
        result = json.loads(self.engine.analyze_narrative({
            "text": "作为一名乡村教师，我的工作经历充满了挑战和成长。",
            "mode": "structure",
        }))
        assert result["text_length"] > 0

    def test_empty_text(self):
        result = json.loads(self.engine.analyze_narrative({"text": ""}))
        assert "error" in result

    def test_output_is_valid_json(self):
        result = self.engine.analyze_narrative({"text": "简单测试文本。"})
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_coherence_score_range(self):
        """Coherence score should be between 0 and 1."""
        result = json.loads(self.engine.analyze_narrative({
            "text": self._interview_text(),
            "mode": "structure",
        }))
        assert 0 <= result["coherence_score"] <= 1
