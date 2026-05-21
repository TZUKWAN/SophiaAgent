"""Tests for Phase I: Research Ethics and IRB Support."""

import pytest
from sophia.research.ethics_support import EthicsSupportEngine


@pytest.fixture
def engine():
    return EthicsSupportEngine()


# ---------------------------------------------------------------------------
# I-1: Ethics Review Checklist
# ---------------------------------------------------------------------------

class TestEthicsChecklist:
    def test_basic_survey(self, engine):
        result = engine.ethics_checklist({"study_type": "survey"})
        assert result["study_type"] == "survey"
        assert len(result["dimensions"]) == 6
        assert "risk_level" in result
        assert "recommendations" in result
        assert result["irb_required"] is False

    def test_vulnerable_population(self, engine):
        result = engine.ethics_checklist({
            "study_type": "interview",
            "involves_vulnerable": True,
        })
        dim_ids = [d["dimension_id"] for d in result["dimensions"]]
        assert "vulnerable_populations" in dim_ids
        vul_dim = next(d for d in result["dimensions"] if d["dimension_id"] == "vulnerable_populations")
        assert len(vul_dim["items"]) > 0
        assert any("弱势群体" in r for r in result["recommendations"])

    def test_deception_flag(self, engine):
        result = engine.ethics_checklist({
            "study_type": "experiment",
            "involves_deception": True,
        })
        rb_dim = next(d for d in result["dimensions"] if d["dimension_id"] == "risk_benefit")
        assert any("debriefing" in item.get("note", "") for item in rb_dim["items"])
        assert any("debriefing" in r for r in result["recommendations"])

    def test_data_linkable_flag(self, engine):
        result = engine.ethics_checklist({
            "study_type": "survey",
            "data_linkable": True,
        })
        pc_dim = next(d for d in result["dimensions"] if d["dimension_id"] == "privacy_confidentiality")
        assert any("去标识化" in item.get("note", "") for item in pc_dim["items"])

    def test_funding_conflict_flag(self, engine):
        result = engine.ethics_checklist({
            "study_type": "survey",
            "has_funding_conflict": True,
        })
        rc_dim = next(d for d in result["dimensions"] if d["dimension_id"] == "researcher_conduct")
        assert any("利益冲突" in item.get("note", "") for item in rc_dim["items"])
        assert any("利益冲突" in r for r in result["recommendations"])

    def test_cross_border_recommendation(self, engine):
        result = engine.ethics_checklist({
            "study_type": "survey",
            "cross_border": True,
        })
        assert any("跨境" in r for r in result["recommendations"])

    def test_high_risk_triggers_irb(self, engine):
        result = engine.ethics_checklist({
            "study_type": "experiment",
            "involves_vulnerable": True,
            "involves_deception": True,
            "involves_sensitive_topics": True,
        })
        assert result["irb_required"] is True
        assert result["risk_level"] in ("moderate", "high")


# ---------------------------------------------------------------------------
# I-2: Informed Consent Generator
# ---------------------------------------------------------------------------

class TestConsentGenerator:
    def test_adult_survey_template(self, engine):
        result = engine.generate_consent({
            "template_type": "adult_survey",
            "study_title": "大学生学习动机调查",
            "researcher_name": "张三",
            "researcher_contact": "zhangsan@university.edu.cn",
            "estimated_duration": "15",
            "study_purpose": "了解大学生的学习动机及其影响因素",
        })
        assert result["title"] == "大学生学习动机调查"
        assert result["template_type"] == "adult_survey"
        assert len(result["sections"]) > 0
        assert result["signature_required"] is True
        headings = [s["heading"] for s in result["sections"]]
        assert "研究目的" in headings
        assert "自愿参与与退出" in headings
        content = " ".join(s["content"] for s in result["sections"])
        assert "张三" in content
        assert "zhangsan@university.edu.cn" in content
        assert "15" in content
        assert "学习动机" in content

    def test_adult_interview_template(self, engine):
        result = engine.generate_consent({
            "template_type": "adult_interview",
            "study_title": "教师职业倦怠访谈",
            "researcher_name": "李四",
            "researcher_contact": "lisi@university.edu.cn",
            "estimated_duration": "60",
            "study_purpose": "深入了解教师职业倦怠的经历和感受",
            "recording_consent": "同意",
        })
        assert result["template_type"] == "adult_interview"
        headings = [s["heading"] for s in result["sections"]]
        assert "录音同意" in headings
        content = " ".join(s["content"] for s in result["sections"])
        assert "录音" in content
        assert "同意" in content

    def test_minor_survey_template(self, engine):
        result = engine.generate_consent({
            "template_type": "minor_survey",
            "study_title": "中学生网络使用行为研究",
            "researcher_name": "王五",
            "researcher_contact": "wangwu@university.edu.cn",
            "estimated_duration": "20",
            "study_purpose": "了解中学生的网络使用行为模式",
            "age_range": "12-15岁",
            "activity_description": "完成一份关于网络使用习惯的问卷",
        })
        assert result["template_type"] == "minor_survey"
        headings = [s["heading"] for s in result["sections"]]
        assert "监护人同意" in headings
        assert "未成年人知情" in headings
        content = " ".join(s["content"] for s in result["sections"])
        assert "12-15岁" in content
        assert "监护人" in content

    def test_experiment_template(self, engine):
        result = engine.generate_consent({
            "template_type": "experiment",
            "study_title": "认知负荷对阅读理解的影响",
            "researcher_name": "赵六",
            "researcher_contact": "zhaoliu@university.edu.cn",
            "estimated_duration": "45",
            "study_purpose": "检验不同认知负荷条件下阅读理解表现的差异",
            "compensation": "课程学分2分",
            "risks": "无明显风险，可能出现轻微疲劳",
        })
        assert result["template_type"] == "experiment"
        headings = [s["heading"] for s in result["sections"]]
        assert "实验流程" in headings
        assert "补偿" in headings
        content = " ".join(s["content"] for s in result["sections"])
        assert "课程学分" in content
        assert "疲劳" in content

    def test_unknown_template_type(self, engine):
        result = engine.generate_consent({
            "template_type": "nonexistent",
            "study_title": "测试",
            "researcher_name": "测试",
            "researcher_contact": "test@test.com",
            "study_purpose": "测试",
        })
        assert "error" in result
        assert "available" in result

    def test_custom_sections(self, engine):
        result = engine.generate_consent({
            "template_type": "adult_survey",
            "study_title": "测试",
            "researcher_name": "测试",
            "researcher_contact": "test@test.com",
            "study_purpose": "测试",
            "custom_sections": [
                {"heading": "额外声明", "content": "这是自定义内容"},
            ],
        })
        headings = [s["heading"] for s in result["sections"]]
        assert "额外声明" in headings
        content = next(s["content"] for s in result["sections"] if s["heading"] == "额外声明")
        assert "自定义内容" in content

    def test_list_consent_templates(self, engine):
        templates = engine.list_consent_templates()
        assert len(templates) == 4
        types = [t["type"] for t in templates]
        assert "adult_survey" in types
        assert "minor_survey" in types
        assert "experiment" in types


# ---------------------------------------------------------------------------
# I-3: Risk Level Assessment
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    def test_minimal_risk(self, engine):
        result = engine.assess_risk({
            "study_type": "secondary_data",
            "involves_vulnerable": False,
            "involves_deception": False,
        })
        assert result["risk_level"] == "minimal"
        assert result["risk_score"] <= 2
        assert "豁免审查" in result["irb_review_track"]

    def test_low_risk_survey(self, engine):
        result = engine.assess_risk({
            "study_type": "survey",
            "involves_sensitive_topics": True,
        })
        assert result["risk_level"] == "low"
        assert result["risk_score"] <= 5
        assert "快速审查" in result["irb_review_track"]

    def test_moderate_risk_experiment(self, engine):
        result = engine.assess_risk({
            "study_type": "experiment",
            "involves_sensitive_topics": True,
            "data_linkable": True,
            "cross_border": True,
            "data_sharing": True,
        })
        assert result["risk_level"] in ("moderate", "high")
        assert "全面审查" in result["irb_review_track"]
        assert len(result["risk_factors"]) > 0
        assert len(result["mitigation_suggestions"]) > 0

    def test_high_risk(self, engine):
        result = engine.assess_risk({
            "study_type": "experiment",
            "involves_vulnerable": True,
            "involves_deception": True,
            "physical_intervention": True,
            "involves_sensitive_topics": True,
        })
        assert result["risk_level"] == "high"
        assert result["risk_score"] > 8
        assert "全面审查" in result["irb_review_track"]
        assert any("弱势群体" in f for f in result["risk_factors"])
        assert any("欺骗" in f for f in result["risk_factors"])
        assert any("身体干预" in f for f in result["risk_factors"])

    def test_mitigation_suggestions(self, engine):
        result = engine.assess_risk({
            "study_type": "experiment",
            "involves_deception": True,
            "involves_sensitive_topics": True,
            "cross_border": True,
        })
        mitigations = result["mitigation_suggestions"]
        assert any("debriefing" in m for m in mitigations)
        assert any("心理咨询" in m or "心理" in m for m in mitigations)
        assert any("SCC" in m or "标准合同" in m for m in mitigations)

    def test_default_args(self, engine):
        result = engine.assess_risk({})
        assert "risk_score" in result
        assert "risk_level" in result
        assert "irb_review_track" in result
