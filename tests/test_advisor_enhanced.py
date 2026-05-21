"""Tests for Phase H: Methodology advisor enhancements."""

import pytest
from sophia.research.advisor import MethodologyAdvisor


@pytest.fixture
def advisor():
    return MethodologyAdvisor()


# ---------------------------------------------------------------------------
# H-1: Research question diagnosis
# ---------------------------------------------------------------------------

class TestDiagnoseQuestion:
    def test_causal_question(self, advisor):
        result = advisor.diagnose_question("数字经济发展对城市创新效率的影响研究")
        assert "diagnosis" in result
        diag = result["diagnosis"]
        assert diag["question_type"] == "解释性"
        assert diag["paradigm"] == "实证主义"
        recs = result["recommended_methods"]
        assert len(recs) >= 1
        # Should recommend causal inference methods
        method_names = [r["method"] for r in recs]
        assert any("回归" in m or "差分" in m or "模型" in m for m in method_names)

    def test_exploratory_question(self, advisor):
        result = advisor.diagnose_question("高校青年教师如何理解和应对职业倦怠？")
        diag = result["diagnosis"]
        assert diag["question_type"] == "探索性"
        assert diag["paradigm"] == "解释主义"
        recs = result["recommended_methods"]
        method_names = [r["method"] for r in recs]
        assert any("访谈" in m for m in method_names)

    def test_descriptive_question(self, advisor):
        result = advisor.diagnose_question("当前农村留守儿童的心理健康状况调查")
        diag = result["diagnosis"]
        assert diag["question_type"] == "描述性"
        recs = result["recommended_methods"]
        method_names = [r["method"] for r in recs]
        assert any("问卷" in m for m in method_names)

    def test_longitudinal_detection(self, advisor):
        result = advisor.diagnose_question("近十年来农民工收入结构的变化趋势研究")
        diag = result["diagnosis"]
        assert diag["time_dimension"] == "纵向"

    def test_cross_sectional_default(self, advisor):
        result = advisor.diagnose_question("城市居民幸福感现状分析")
        diag = result["diagnosis"]
        assert diag["time_dimension"] == "横截面"

    def test_individual_level(self, advisor):
        result = advisor.diagnose_question("大学生学习动机与学业成绩的关系")
        diag = result["diagnosis"]
        assert diag["analysis_level"] == "个体"

    def test_social_level(self, advisor):
        result = advisor.diagnose_question("社会保障政策对农村贫困的影响")
        diag = result["diagnosis"]
        assert diag["analysis_level"] == "社会"

    def test_diagnosis_structure(self, advisor):
        result = advisor.diagnose_question("测试问题")
        assert "question" in result
        assert "diagnosis" in result
        assert "recommended_methods" in result
        for rec in result["recommended_methods"]:
            assert "method" in rec
            assert "score" in rec
            assert "reason" in rec


# ---------------------------------------------------------------------------
# H-2: Mixed method design
# ---------------------------------------------------------------------------

class TestMixedMethodDesign:
    def test_convergent_design(self, advisor):
        result = advisor.design_mixed_method(
            qual_question="教师如何理解教育公平？",
            quant_question="教育公平感知与教学效能感的关系",
            priority="equal",
        )
        assert result["design_type"] == "聚合式设计"
        assert "qual_phase" in result
        assert "quant_phase" in result
        assert "integration_points" in result
        assert len(result["integration_points"]) >= 2
        assert "timeline" in result
        assert "validation_strategy" in result

    def test_exploratory_design(self, advisor):
        result = advisor.design_mixed_method(
            qual_question="探索影响学生参与度的未知因素",
            quant_question="验证探索阶段发现的假设",
            priority="equal",
        )
        assert result["design_type"] == "探索性顺序设计"
        assert "探索" in result["rationale"] or "假设" in result["rationale"]

    def test_explanatory_design(self, advisor):
        result = advisor.design_mixed_method(
            qual_question="解释为什么某些学生成绩异常低",
            quant_question="学生成绩的影响因素分析",
            priority="equal",
        )
        assert result["design_type"] == "解释性顺序设计"

    def test_embedded_design(self, advisor):
        result = advisor.design_mixed_method(
            qual_question="实验中的访谈补充",
            quant_question="教育干预的效果评估",
            priority="quantitative",
        )
        assert result["design_type"] == "嵌入式设计"

    def test_mixed_method_structure(self, advisor):
        result = advisor.design_mixed_method(
            qual_question="Q1",
            quant_question="Q2",
        )
        assert "design_type" in result
        assert "rationale" in result
        assert "qual_phase" in result
        assert "quant_phase" in result
        assert "integration_points" in result
        assert "timeline" in result
        assert "validation_strategy" in result
        # Check qual_phase structure
        qp = result["qual_phase"]
        assert "methods" in qp
        assert "data" in qp
        assert "analysis" in qp


# ---------------------------------------------------------------------------
# H-3: Sampling strategy recommendation
# ---------------------------------------------------------------------------

class TestRecommendSampling:
    def test_qualitative_recommends_purposive(self, advisor):
        result = advisor.recommend_sampling(
            research_design="质性访谈研究",
            population="高校青年教师",
        )
        assert result["is_qualitative"] is True
        recs = result["recommended"]
        assert len(recs) >= 1
        method_names = [r["strategy"] for r in recs]
        assert "目的性抽样" in method_names

    def test_quantitative_recommends_random(self, advisor):
        result = advisor.recommend_sampling(
            research_design="量化问卷调查",
            population="大学生群体",
        )
        assert result["is_quantitative"] is True
        recs = result["recommended"]
        method_names = [r["strategy"] for r in recs]
        assert any("随机" in m or "分层" in m or "整群" in m for m in method_names)

    def test_hidden_population_recommends_snowball(self, advisor):
        result = advisor.recommend_sampling(
            research_design="质性研究",
            population="难以接触的群体",
            constraints=["难以接触", "敏感话题"],
        )
        recs = result["recommended"]
        method_names = [r["strategy"] for r in recs]
        assert "滚雪球抽样" in method_names

    def test_grounded_theory_recommends_theoretical(self, advisor):
        result = advisor.recommend_sampling(
            research_design="扎根理论研究",
        )
        recs = result["recommended"]
        method_names = [r["strategy"] for r in recs]
        assert "理论抽样" in method_names

    def test_resource_limited(self, advisor):
        result = advisor.recommend_sampling(
            research_design="量化研究",
            constraints=["资源有限", "预算不足"],
        )
        recs = result["recommended"]
        method_names = [r["strategy"] for r in recs]
        assert "便利抽样" in method_names

    def test_stratified_hint(self, advisor):
        result = advisor.recommend_sampling(
            research_design="量化研究",
            constraints=["需要分层"],
        )
        recs = result["recommended"]
        method_names = [r["strategy"] for r in recs]
        assert "分层抽样" in method_names

    def test_sampling_structure(self, advisor):
        result = advisor.recommend_sampling("测试设计")
        recs = result["recommended"]
        for rec in recs:
            assert "strategy" in rec
            assert "description" in rec
            assert "when_to_use" in rec
            assert "sample_size_guidance" in rec
            assert "pros" in rec
            assert "cons" in rec
            assert "estimated_cost" in rec

    def test_sample_size_is_specific(self, advisor):
        result = advisor.recommend_sampling("质性访谈")
        recs = result["recommended"]
        for rec in recs:
            guidance = rec["sample_size_guidance"]
            # Should not be vague like "enough"
            assert "足够多" not in guidance
            assert len(guidance) > 5
