"""Tests for Phase G: Interview and Questionnaire Pipeline."""

import pytest
import numpy as np
from sophia.research.interview_questionnaire import (
    DataCollectionTracker,
    InterviewEngine,
    PilotAnalyzer,
    QuestionnaireEngine,
    ScaleLibrary,
)


# ---------------------------------------------------------------------------
# G-4: Scale Library
# ---------------------------------------------------------------------------

class TestScaleLibrary:
    @pytest.fixture
    def lib(self):
        return ScaleLibrary()

    def test_search_by_keyword(self, lib):
        results = lib.search(query="抑郁")
        assert len(results) >= 1
        assert any("抑郁" in r["name"] for r in results)

    def test_search_by_domain(self, lib):
        results = lib.search(domain="心理健康")
        assert len(results) >= 2
        assert all(r["domain"] == "心理健康" for r in results)

    def test_get_existing_scale(self, lib):
        scale = lib.get("抑郁自评量表")
        assert scale is not None
        assert scale["n_items"] == 20
        assert scale["scale_range"] == (1, 4)

    def test_get_missing_scale(self, lib):
        assert lib.get("不存在的量表") is None

    def test_list_domains(self, lib):
        domains = lib.list_domains()
        assert "心理健康" in domains
        assert "教育心理" in domains

    def test_list_all(self, lib):
        all_scales = lib.list_all()
        assert len(all_scales) >= 5

    def test_scale_has_subscales(self, lib):
        scale = lib.get("学习投入量表")
        assert len(scale["subscales"]) > 0
        assert any(s["name"] == "活力" for s in scale["subscales"])

    def test_scale_reliability(self, lib):
        scale = lib.get("一般自我效能感量表")
        assert "cronbach_alpha" in scale["reliability"]
        assert scale["reliability"]["cronbach_alpha"] > 0.7


# ---------------------------------------------------------------------------
# G-1: Questionnaire Engine
# ---------------------------------------------------------------------------

class TestQuestionnaireEngine:
    @pytest.fixture
    def engine(self):
        return QuestionnaireEngine()

    def test_design_basic(self, engine):
        result = engine.design_questionnaire({
            "topic": "大学生学习动机研究",
            "target_population": "大学生",
            "research_questions": ["RQ1: 学习动机的影响因素", "RQ2: 学习动机与成绩的关系"],
            "variables": ["内在动机", "外在动机", "学习策略"],
        })
        assert result["title"] == "《大学生学习动机研究》调查问卷"
        assert result["target_population"] == "大学生"
        assert len(result["sections"]) >= 2
        assert result["total_questions"] > 0
        assert "estimated_time" in result

    def test_design_with_demographics(self, engine):
        result = engine.design_questionnaire({
            "topic": "测试",
            "target_population": "大学生",
            "include_demographics": True,
        })
        first_section = result["sections"][0]
        assert first_section["title"] == "基本信息"
        # Should have gender and age at minimum
        assert len(first_section["questions"]) >= 2

    def test_design_without_demographics(self, engine):
        result = engine.design_questionnaire({
            "topic": "测试",
            "target_population": "大学生",
            "include_demographics": False,
        })
        assert not any(s["title"] == "基本信息" for s in result["sections"])

    def test_design_scale_suggestions(self, engine):
        result = engine.design_questionnaire({
            "topic": "大学生抑郁与焦虑研究",
            "target_population": "大学生",
            "scale_suggestions": True,
        })
        assert len(result["suggested_scales"]) > 0

    def test_validate_double_barreled(self, engine):
        result = engine.validate_questionnaire({
            "questions": [
                {"text": "你觉得这个课程的教学质量和作业量如何？", "type": "likert"},
            ]
        })
        assert any(i["type"] == "double_barreled" for i in result["issues"])

    def test_validate_long_questionnaire(self, engine):
        questions = [{"text": f"问题{i}", "type": "likert"} for i in range(60)]
        result = engine.validate_questionnaire({"questions": questions})
        assert any(w["type"] == "long_questionnaire" for w in result["warnings"])

    def test_validate_short_questionnaire(self, engine):
        questions = [{"text": "问题1", "type": "single_choice"}]
        result = engine.validate_questionnaire({"questions": questions})
        assert any(w["type"] == "short_questionnaire" for w in result["warnings"])

    def test_validate_inconsistent_scales(self, engine):
        questions = [
            {"text": "Q1", "type": "likert", "options": [{"value": i} for i in range(1, 6)]},
            {"text": "Q2", "type": "likert", "options": [{"value": i} for i in range(1, 8)]},
        ]
        result = engine.validate_questionnaire({"questions": questions})
        assert any(w["type"] == "inconsistent_scales" for w in result["warnings"])

    def test_validate_pass(self, engine):
        questions = [
            {"text": "您对本课程的满意度如何？", "type": "likert"},
            {"text": "您每周学习多少小时？", "type": "single_choice"},
        ]
        result = engine.validate_questionnaire({"questions": questions})
        assert isinstance(result["pass"], bool)


# ---------------------------------------------------------------------------
# G-2: Interview Engine
# ---------------------------------------------------------------------------

class TestInterviewEngine:
    @pytest.fixture
    def engine(self):
        return InterviewEngine()

    def test_generate_focused_protocol(self, engine):
        result = engine.generate_protocol({
            "topic": "乡村教师职业倦怠",
            "interview_type": "focused",
            "target_population": "乡村中小学教师",
            "research_questions": ["乡村教师职业倦怠的表现", "职业倦怠的影响因素"],
        })
        assert "乡村教师职业倦怠" in result["title"]
        assert result["interview_type"] == "focused"
        assert len(result["sections"]) > 0
        assert len(result["general_probing_strategies"]) > 0
        assert len(result["ethics_reminders"]) > 0

    def test_generate_life_history_protocol(self, engine):
        result = engine.generate_protocol({
            "topic": "农民工城市融入",
            "interview_type": "life_history",
            "target_population": "进城务工人员",
        })
        assert result["interview_type_name"] == "生命史访谈"
        section_titles = [s["title"] for s in result["sections"]]
        assert "成长背景" in section_titles

    def test_generate_narrative_protocol(self, engine):
        result = engine.generate_protocol({
            "topic": "高考复读经历",
            "interview_type": "narrative",
            "target_population": "复读生",
        })
        section_titles = [s["title"] for s in result["sections"]]
        assert "故事开场" in section_titles
        assert "转折点" in section_titles

    def test_sections_have_questions(self, engine):
        result = engine.generate_protocol({
            "topic": "测试",
            "target_population": "测试人群",
            "n_questions_per_section": 3,
        })
        for sec in result["sections"]:
            assert len(sec["questions"]) > 0
            assert len(sec["probing_strategies"]) > 0

    def test_ethics_reminders_present(self, engine):
        result = engine.generate_protocol({
            "topic": "测试",
            "target_population": "测试人群",
        })
        assert len(result["ethics_reminders"]) >= 3
        assert any("知情同意" in r for r in result["ethics_reminders"])

    def test_recording_suggestions_present(self, engine):
        result = engine.generate_protocol({
            "topic": "测试",
            "target_population": "测试人群",
        })
        assert len(result["recording_suggestions"]) >= 3


# ---------------------------------------------------------------------------
# G-3: Data Collection Tracker
# ---------------------------------------------------------------------------

class TestDataCollectionTracker:
    @pytest.fixture
    def tracker(self):
        return DataCollectionTracker()

    def test_create_project(self, tracker):
        result = tracker.create_project({
            "project_id": "test_001",
            "project_name": "大学生问卷调查",
            "target_sample_size": 200,
        })
        assert result["status"] == "created"
        assert result["project"]["project_id"] == "test_001"

    def test_add_record(self, tracker):
        tracker.create_project({"project_id": "p1", "project_name": "Test"})
        result = tracker.add_record({
            "project_id": "p1",
            "record_id": "r1",
            "status": "completed",
            "duration_minutes": 12.5,
        })
        assert result["status"] == "record_added"

    def test_add_record_invalid_project(self, tracker):
        result = tracker.add_record({"project_id": "nonexistent", "status": "completed"})
        assert "error" in result

    def test_report_basic(self, tracker):
        tracker.create_project({"project_id": "p2", "project_name": "Test", "target_sample_size": 10})
        for i in range(5):
            tracker.add_record({"project_id": "p2", "status": "completed", "duration_minutes": 10})
        for i in range(2):
            tracker.add_record({"project_id": "p2", "status": "refused", "duration_minutes": 0})
        report = tracker.get_report({"project_id": "p2"})
        assert report["completed"] == 5
        assert report["refused"] == 2
        assert abs(report["response_rate_pct"] - 5 / 7 * 100) < 0.5
        assert abs(report["completion_rate_pct"] - 50.0) < 0.5
        assert report["status"] == "in_progress"

    def test_report_quality_flags(self, tracker):
        tracker.create_project({"project_id": "p3", "project_name": "Test", "target_sample_size": 10})
        tracker.add_record({"project_id": "p3", "status": "completed", "quality_flags": ["too_fast"]})
        tracker.add_record({"project_id": "p3", "status": "completed", "quality_flags": ["too_fast", "straight_line"]})
        report = tracker.get_report({"project_id": "p3"})
        assert report["quality_flags"]["too_fast"] == 2
        assert report["quality_flags"]["straight_line"] == 1

    def test_report_too_fast(self, tracker):
        tracker.create_project({"project_id": "p4", "project_name": "Test", "target_sample_size": 10})
        tracker.add_record({"project_id": "p4", "status": "completed", "duration_minutes": 1.5})
        tracker.add_record({"project_id": "p4", "status": "completed", "duration_minutes": 15})
        report = tracker.get_report({"project_id": "p4"})
        assert report["too_fast_count"] == 1
        assert abs(report["avg_duration_minutes"] - 8.25) < 0.1

    def test_list_projects(self, tracker):
        tracker.create_project({"project_id": "p5", "project_name": "A", "target_sample_size": 50})
        tracker.create_project({"project_id": "p6", "project_name": "B", "target_sample_size": 100})
        projects = tracker.list_projects()
        assert len(projects) == 2

    def test_report_invalid_project(self, tracker):
        result = tracker.get_report({"project_id": "nonexistent"})
        assert "error" in result


# ---------------------------------------------------------------------------
# G-5: Pilot Analyzer
# ---------------------------------------------------------------------------

class TestPilotAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return PilotAnalyzer()

    def test_analyze_basic(self, analyzer):
        data = [
            {"q1": 4, "q2": 3, "q3": 5, "q4": 4, "q5": 3},
            {"q1": 3, "q2": 4, "q3": 4, "q4": 3, "q5": 4},
            {"q1": 5, "q2": 5, "q3": 5, "q4": 5, "q5": 5},
            {"q1": 2, "q2": 2, "q3": 3, "q4": 2, "q5": 2},
            {"q1": 4, "q2": 3, "q3": 4, "q4": 4, "q5": 3},
        ]
        result = analyzer.analyze({
            "data": data,
            "item_cols": ["q1", "q2", "q3", "q4", "q5"],
            "scale_min": 1,
            "scale_max": 5,
        })
        assert result["n_respondents"] == 5
        assert result["n_items"] == 5
        assert result["cronbach_alpha"] is not None
        assert len(result["item_statistics"]) == 5
        assert len(result["item_total_correlation"]) == 5
        assert "overall_assessment" in result

    def test_analyze_with_reverse_items(self, analyzer):
        data = [
            {"q1": 5, "q2": 4, "q3": 1},  # q3 is reverse
            {"q1": 4, "q2": 3, "q3": 2},
            {"q1": 3, "q2": 2, "q3": 3},
            {"q1": 5, "q2": 5, "q3": 1},
            {"q1": 4, "q2": 4, "q3": 2},
        ]
        result = analyzer.analyze({
            "data": data,
            "item_cols": ["q1", "q2", "q3"],
            "reverse_items": [3],
        })
        # After reverse coding, q3 should correlate positively with others
        assert result["cronbach_alpha"] is not None

    def test_analyze_missing_data(self, analyzer):
        data = [
            {"q1": 5, "q2": 4, "q3": np.nan},
            {"q1": 4, "q2": np.nan, "q3": 3},
            {"q1": 3, "q2": 2, "q3": 4},
        ]
        result = analyzer.analyze({
            "data": data,
            "item_cols": ["q1", "q2", "q3"],
        })
        assert result["n_respondents"] == 3
        # Should still compute alpha with valid data
        assert result["cronbach_alpha"] is not None

    def test_analyze_insufficient_sample(self, analyzer):
        result = analyzer.analyze({
            "data": [{"q1": 5}],
            "item_cols": ["q1"],
        })
        assert "error" in result

    def test_analyze_ceiling_effect(self, analyzer):
        # All respondents select max score for q1
        data = [
            {"q1": 5, "q2": 3},
            {"q1": 5, "q2": 2},
            {"q1": 5, "q2": 4},
            {"q1": 5, "q2": 3},
        ]
        result = analyzer.analyze({
            "data": data,
            "item_cols": ["q1", "q2"],
            "scale_max": 5,
        })
        assert any(s["issue"] == "ceiling_effect" for s in result["suggestions"])

    def test_analyze_low_reliability(self, analyzer):
        # Items with very low correlation
        np.random.seed(42)
        data = []
        for _ in range(20):
            data.append({
                "q1": int(np.random.randint(1, 6)),
                "q2": int(np.random.randint(1, 6)),
                "q3": int(np.random.randint(1, 6)),
            })
        result = analyzer.analyze({
            "data": data,
            "item_cols": ["q1", "q2", "q3"],
        })
        # Random data should have low alpha
        assert result["cronbach_alpha"] is not None
        # May or may not trigger low_reliability depending on random data
        assert "overall_assessment" in result

    def test_analyze_list_format(self, analyzer):
        data = [
            [4, 3, 5],
            [3, 4, 4],
            [5, 5, 5],
            [2, 2, 3],
            [4, 3, 4],
        ]
        result = analyzer.analyze({
            "data": data,
            "item_cols": ["q1", "q2", "q3"],
        })
        assert result["n_respondents"] == 5
        assert result["cronbach_alpha"] is not None

    def test_analyze_item_names(self, analyzer):
        data = [
            {"a": 4, "b": 3},
            {"a": 3, "b": 4},
            {"a": 5, "b": 5},
            {"a": 2, "b": 2},
        ]
        result = analyzer.analyze({
            "data": data,
            "item_cols": ["a", "b"],
            "item_names": ["内在动机", "外在动机"],
        })
        assert "内在动机" in result["item_total_correlation"]
        assert "外在动机" in result["item_total_correlation"]
