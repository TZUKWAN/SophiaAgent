"""Tests for Phase K: Academic presentation slide generator."""

import pytest
from sophia.research.ppt_generator import HTMLSlideRenderer, SlidePlanner


@pytest.fixture
def planner():
    return SlidePlanner()


@pytest.fixture
def renderer():
    return HTMLSlideRenderer()


# ---------------------------------------------------------------------------
# K-1: Slide structure planning
# ---------------------------------------------------------------------------

class TestSlideStructure:
    def test_conference_mode(self, planner):
        result = planner.generate_structure({
            "paper_title": "社会资本对农民工城市融入的影响研究",
            "mode": "conference",
            "key_findings": [
                "强关系社会资本显著促进经济融入",
                "社区参与对文化融入具有正向影响",
                "制度性障碍削弱了社会资本的积极作用",
            ],
        })
        assert result["mode"] == "conference"
        assert result["total_slides"] >= 10
        assert result["total_slides"] <= 20
        assert result["duration_minutes"] == 15
        assert len(result["slides"]) == result["total_slides"]
        assert len(result["tips"]) > 0

        # Check first slide is title
        assert result["slides"][0]["type"] == "title"
        assert "社会资本" in result["slides"][0]["content_bullets"][0]

        # Check final slide
        assert result["slides"][-1]["type"] == "final"

        # Check findings are populated
        finding_slides = [s for s in result["slides"] if "发现" in s["title"]]
        assert len(finding_slides) >= 2
        assert any("强关系" in str(s["content_bullets"]) for s in finding_slides)

    def test_defense_mode(self, planner):
        result = planner.generate_structure({
            "paper_title": "教师职业倦怠的形成机制与干预策略研究",
            "mode": "defense",
            "duration_minutes": 30,
        })
        assert result["mode"] == "defense"
        assert result["total_slides"] >= 20
        assert result["total_slides"] <= 35
        assert result["duration_minutes"] == 30
        assert result["slides"][0]["type"] == "title"
        assert result["slides"][-1]["type"] == "final"

    def test_default_mode_is_conference(self, planner):
        result = planner.generate_structure({"paper_title": "测试"})
        assert result["mode"] == "conference"

    def test_time_per_slide(self, planner):
        result = planner.generate_structure({
            "paper_title": "测试",
            "mode": "conference",
            "duration_minutes": 20,
        })
        assert result["estimated_time_per_slide"] > 0
        # 20 minutes / ~15 slides ≈ 1.3 minutes per slide
        assert result["estimated_time_per_slide"] < 2.0

    def test_defense_has_more_slides(self, planner):
        conf = planner.generate_structure({"paper_title": "测试", "mode": "conference"})
        defense = planner.generate_structure({"paper_title": "测试", "mode": "defense"})
        assert defense["total_slides"] > conf["total_slides"]

    def test_defense_tips_include_defense_specific(self, planner):
        result = planner.generate_structure({"paper_title": "测试", "mode": "defense"})
        tips = result["tips"]
        assert any("答辩" in t or "评委" in t or "创新点" in t for t in tips)

    def test_conference_tips_include_conference_specific(self, planner):
        result = planner.generate_structure({"paper_title": "测试", "mode": "conference"})
        tips = result["tips"]
        assert any("会议" in t or "问答" in t or "核心发现" in t for t in tips)

    def test_slide_has_required_fields(self, planner):
        result = planner.generate_structure({"paper_title": "测试"})
        for slide in result["slides"]:
            assert "slide_number" in slide
            assert "type" in slide
            assert "title" in slide
            assert "content_bullets" in slide
            assert "speaker_notes" in slide
            assert "suggested_visual" in slide
            assert "source_section" in slide


# ---------------------------------------------------------------------------
# K-2: HTML rendering
# ---------------------------------------------------------------------------

class TestHTMLRenderer:
    def test_render_basic(self, planner, renderer):
        structure = planner.generate_structure({
            "paper_title": "测试论文",
            "mode": "conference",
        })
        result = renderer.render({
            "slides": structure["slides"],
            "title": "测试汇报",
        })
        assert "html" in result
        assert result["slide_count"] == structure["total_slides"]
        assert result["title"] == "测试汇报"
        html_content = result["html"]
        assert "<!DOCTYPE html>" in html_content
        assert "测试汇报" in html_content
        assert "方向键翻页" in html_content
        assert 'class="progress"' in html_content

    def test_render_contains_slides(self, planner, renderer):
        structure = planner.generate_structure({
            "paper_title": "测试",
            "mode": "conference",
        })
        result = renderer.render({
            "slides": structure["slides"],
        })
        html_content = result["html"]
        # Should contain slide divs
        assert html_content.count('class="slide') >= structure["total_slides"]

    def test_render_no_slides_error(self, renderer):
        result = renderer.render({"slides": []})
        assert "error" in result

    def test_render_escapes_html(self, planner, renderer):
        structure = planner.generate_structure({
            "paper_title": "<script>alert(1)</script>",
            "mode": "conference",
        })
        result = renderer.render({
            "slides": structure["slides"],
        })
        html_content = result["html"]
        # The title should be escaped (not raw script tag in content)
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_content

    def test_render_notes_visible(self, planner, renderer):
        structure = planner.generate_structure({
            "paper_title": "测试",
            "mode": "conference",
        })
        result = renderer.render({
            "slides": structure["slides"],
        })
        html_content = result["html"]
        assert "演讲备注" in html_content

    def test_render_defense_longer(self, planner, renderer):
        structure = planner.generate_structure({
            "paper_title": "测试",
            "mode": "defense",
        })
        result = renderer.render({
            "slides": structure["slides"],
        })
        assert result["slide_count"] >= 20
