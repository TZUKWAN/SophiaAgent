"""Tests for Phase H: Enhanced Research Design."""

import pytest
from sophia.research.design_enhanced import EnhancedDesignEngine


@pytest.fixture
def engine():
    return EnhancedDesignEngine()


# ---------------------------------------------------------------------------
# H-1: Design Templates
# ---------------------------------------------------------------------------

class TestDesignTemplates:
    def test_get_experimental_template(self, engine):
        result = engine.get_design_template({"design_type": "experimental"})
        assert result["name"] == "实验研究设计"
        assert len(result["sections"]) >= 5
        assert "common_methods" in result
        assert "quality_criteria" in result

    def test_get_qualitative_template(self, engine):
        result = engine.get_design_template({"design_type": "qualitative"})
        assert result["name"] == "质性研究设计"
        assert any(s["title"] == "研究范式" for s in result["sections"])

    def test_get_mixed_methods_template(self, engine):
        result = engine.get_design_template({"design_type": "mixed_methods"})
        assert result["name"] == "混合方法研究设计"
        assert "quantitative_component" not in result  # template only, not full design

    def test_unknown_design_type(self, engine):
        result = engine.get_design_template({"design_type": "nonexistent"})
        assert "error" in result
        assert "available" in result

    def test_discipline_notes(self, engine):
        result = engine.get_design_template({
            "design_type": "experimental",
            "discipline": "教育学",
        })
        assert "discipline_notes" in result
        assert len(result["discipline_notes"]) > 0

    def test_rq_alignment_experimental(self, engine):
        result = engine.get_design_template({
            "design_type": "experimental",
            "research_question": "翻转课堂对学生成绩有何影响？",
        })
        assert result["rq_alignment"]["aligned"] is True

    def test_rq_alignment_qualitative_with_quant_rq(self, engine):
        result = engine.get_design_template({
            "design_type": "qualitative",
            "research_question": "有多少学生支持这个政策？",
        })
        assert result["rq_alignment"]["aligned"] is False
        assert len(result["rq_alignment"]["issues"]) > 0

    def test_list_design_types(self, engine):
        types = engine.list_design_types()
        assert len(types) >= 5
        names = [t["name"] for t in types]
        assert "实验研究设计" in names
        assert "质性研究设计" in names


# ---------------------------------------------------------------------------
# H-3: Mixed Methods Design
# ---------------------------------------------------------------------------

class TestMixedMethodsDesign:
    def test_convergent_parallel(self, engine):
        result = engine.mixed_methods_design({
            "design_subtype": "convergent_parallel",
            "research_question": "教师职业倦怠的现状与机制",
            "quantitative_focus": "倦怠水平测量与影响因素分析",
            "qualitative_focus": "倦怠经历的深层意义建构",
        })
        assert result["name"] == "聚敛式并行设计"
        assert result["timing"] == "concurrent"
        assert result["priority"] == "equal"
        assert "visual_diagram" in result
        assert len(result["quality_checklist"]) > 0
        assert "quantitative_component" in result
        assert "qualitative_component" in result

    def test_explanatory_sequential(self, engine):
        result = engine.mixed_methods_design({
            "design_subtype": "explanatory_sequential",
            "research_question": "测试",
            "quantitative_focus": "测试",
            "qualitative_focus": "测试",
        })
        assert result["name"] == "解释性顺序设计"
        assert result["timing"] == "sequential"
        assert result["priority"] == "quantitative"

    def test_exploratory_sequential(self, engine):
        result = engine.mixed_methods_design({
            "design_subtype": "exploratory_sequential",
            "research_question": "测试",
            "quantitative_focus": "测试",
            "qualitative_focus": "测试",
        })
        assert result["name"] == "探索性顺序设计"
        assert result["priority"] == "qualitative"

    def test_unknown_subtype(self, engine):
        result = engine.mixed_methods_design({
            "design_subtype": "nonexistent",
            "research_question": "测试",
            "quantitative_focus": "测试",
            "qualitative_focus": "测试",
        })
        assert "error" in result


# ---------------------------------------------------------------------------
# H-4: Design Quality Assessment
# ---------------------------------------------------------------------------

class TestDesignQualityAssessment:
    def test_excellent_design(self, engine):
        result = engine.assess_design_quality({
            "design_type": "experimental",
            "research_question": "翻转课堂对大学生学业成绩有何影响？",
            "methods": ["t-test", "ANOVA"],
            "sample_size": 200,
            "has_control_group": True,
            "has_randomization": True,
            "has_pretest": True,
            "has_ethics_approval": True,
            "has_power_analysis": True,
            "has_pilot": True,
        })
        assert result["total_score"] >= 80
        assert result["grade"] == "A"
        assert len(result["breakdown"]) == 5
        assert len(result["recommendations"]) > 0

    def test_weak_design(self, engine):
        result = engine.assess_design_quality({
            "design_type": "experimental",
            "research_question": "教育",
            "methods": ["观察"],
            "sample_size": 10,
            "has_control_group": False,
            "has_randomization": False,
            "has_pretest": False,
            "has_ethics_approval": False,
            "has_power_analysis": False,
            "has_pilot": False,
        })
        assert result["total_score"] < 70
        assert len(result["findings"]) > 0

    def test_survey_design(self, engine):
        result = engine.assess_design_quality({
            "design_type": "survey",
            "research_question": "大学生手机依赖的现状调查",
            "methods": ["描述统计", "相关分析"],
            "sample_size": 500,
            "has_ethics_approval": True,
        })
        assert result["grade"] in ("A", "B", "C", "D")
        assert "breakdown" in result

    def test_no_power_analysis_warning(self, engine):
        result = engine.assess_design_quality({
            "design_type": "experimental",
            "research_question": "测试",
            "sample_size": 50,
            "has_power_analysis": False,
        })
        assert any(f["dimension"] == "统计严谨性" for f in result["findings"])


# ---------------------------------------------------------------------------
# H-2: Method Fit Check
# ---------------------------------------------------------------------------

class TestMethodFitCheck:
    def test_good_fit(self, engine):
        result = engine.check_method_fit({
            "research_question": "翻转课堂对学生成绩的影响",
            "design_type": "experimental",
            "proposed_method": "t-test",
            "data_description": {"N": 100},
        })
        assert result["fit_score"] >= 0.6
        assert result["fit_level"] in ("高度匹配", "基本匹配", "部分匹配", "匹配度低")
        assert len(result["reasons"]) > 0

    def test_poor_fit(self, engine):
        result = engine.check_method_fit({
            "research_question": "农民工城市融入的叙事研究",
            "design_type": "qualitative",
            "proposed_method": "ANOVA",
        })
        assert result["fit_score"] < 0.6
        assert len(result["concerns"]) > 0

    def test_small_sample_concern(self, engine):
        result = engine.check_method_fit({
            "research_question": "测试",
            "design_type": "survey",
            "proposed_method": "结构方程模型",
            "data_description": {"N": 20},
        })
        assert any("样本量" in c for c in result["concerns"])

    def test_causal_rq_without_causal_method(self, engine):
        result = engine.check_method_fit({
            "research_question": "政策A对经济增长的影响",
            "design_type": "observational",
            "proposed_method": "相关分析",
        })
        assert any("因果" in c for c in result["concerns"])
