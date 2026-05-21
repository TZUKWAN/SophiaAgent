"""Tests for Phase J: Journal matching and submission guide."""

import pytest
from sophia.research.journal_db import JournalDatabase


@pytest.fixture
def db():
    return JournalDatabase()


# ---------------------------------------------------------------------------
# J-1: Search and match
# ---------------------------------------------------------------------------

class TestJournalSearch:
    def test_search_by_name(self, db):
        results = db.search("教育研究")
        assert len(results) > 0
        assert any("教育研究" in j.get("name_cn", "") for j in results)

    def test_search_by_discipline(self, db):
        results = db.search("教育学")
        assert len(results) > 0
        assert any(j.get("discipline") == "教育学" for j in results)

    def test_search_by_keyword_in_scope(self, db):
        results = db.search("传播理论")
        assert len(results) > 0

    def test_search_no_results(self, db):
        results = db.search("nonexistent_journal_xyz")
        assert len(results) == 0

    def test_search_limit(self, db):
        results = db.search("学报", limit=5)
        assert len(results) <= 5


class TestJournalMatch:
    def test_match_education_paper(self, db):
        result = db.match({
            "title": "翻转课堂对大学生学习动机的影响研究",
            "abstract": "本研究通过准实验设计，探讨翻转课堂模式对大学生学习动机的影响。",
            "keywords": ["翻转课堂", "学习动机", "大学生", "教学改革"],
            "discipline": "教育学",
            "method_type": "准实验",
        })
        assert "matches" in result
        assert len(result["matches"]) > 0
        # Top match should be education journal
        top = result["matches"][0]
        assert top["discipline"] == "教育学"
        assert top["match_score"] > 0

    def test_match_sociology_paper(self, db):
        result = db.match({
            "title": "农民工城市融入的社会资本研究",
            "abstract": "基于田野调查，分析农民工在城市中的社会网络与融入机制。",
            "keywords": ["农民工", "社会资本", "城市融入", "田野调查"],
            "discipline": "社会学",
            "method_type": "田野调查",
        })
        assert len(result["matches"]) > 0
        disciplines = [m["discipline"] for m in result["matches"]]
        assert "社会学" in disciplines

    def test_match_psychology_paper(self, db):
        result = db.match({
            "title": "工作压力与职业倦怠的关系：一个有调节的中介模型",
            "abstract": "采用问卷调查法，探讨工作压力对职业倦怠的影响机制。",
            "keywords": ["工作压力", "职业倦怠", "中介效应", "问卷调查"],
            "discipline": "心理学",
            "method_type": "问卷调查",
        })
        assert len(result["matches"]) > 0
        disciplines = [m["discipline"] for m in result["matches"]]
        assert "心理学" in disciplines

    def test_match_top_n(self, db):
        result = db.match({
            "title": "测试",
            "abstract": "测试",
            "top_n": 5,
        })
        assert len(result["matches"]) <= 5

    def test_match_returns_snippet(self, db):
        result = db.match({
            "title": "测试",
            "abstract": "测试",
        })
        if result["matches"]:
            assert "scope_snippet" in result["matches"][0]
            assert "..." in result["matches"][0]["scope_snippet"]


class TestJournalList:
    def test_list_all(self, db):
        results = db.list_journals()
        assert len(results) > 0

    def test_list_by_discipline(self, db):
        results = db.list_journals(discipline="教育学")
        assert len(results) > 0
        assert all(j["discipline"] == "教育学" for j in results)

    def test_list_disciplines(self, db):
        disciplines = db.list_disciplines()
        assert len(disciplines) > 0
        assert "教育学" in disciplines


# ---------------------------------------------------------------------------
# J-2: Submission guide
# ---------------------------------------------------------------------------

class TestSubmissionGuide:
    def test_guide_by_id(self, db):
        result = db.get_submission_guide({"journal_id": "jiaoyu-yanjiu"})
        assert "journal_info" in result
        assert result["journal_info"]["name_cn"] == "教育研究"
        assert "format_checklist" in result
        assert len(result["format_checklist"]) > 0
        assert "common_rejection_reasons" in result
        assert len(result["common_rejection_reasons"]) > 0
        assert "writing_tips" in result

    def test_guide_by_name(self, db):
        result = db.get_submission_guide({"journal_name": "心理学报"})
        assert "journal_info" in result
        assert result["journal_info"]["name_cn"] == "心理学报"

    def test_guide_unknown(self, db):
        result = db.get_submission_guide({"journal_id": "nonexistent"})
        assert "error" in result

    def test_guide_has_rejection_reasons(self, db):
        result = db.get_submission_guide({"journal_id": "xinli-xuebao"})
        reasons = result.get("common_rejection_reasons", [])
        assert len(reasons) > 0
        # Should mention common issues
        assert any("实验" in r or "统计" in r or "方法" in r for r in reasons)

    def test_guide_has_writing_tips(self, db):
        result = db.get_submission_guide({"journal_id": "shehuixue-yanjiu"})
        tips = result.get("writing_tips", [])
        assert len(tips) > 0

    def test_guide_has_word_count(self, db):
        result = db.get_submission_guide({"journal_id": "jingji-yanjiu"})
        checklist = result.get("format_checklist", [])
        assert any("字数" in c or "字" in c for c in checklist)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_loads_data(self, db):
        # Database should have loaded from file
        journals = db.list_journals()
        assert len(journals) >= 50

    def test_journal_has_required_fields(self, db):
        journals = db.list_journals()
        for journal in journals[:10]:
            assert "id" in journal
            assert "name_cn" in journal
            assert "discipline" in journal
