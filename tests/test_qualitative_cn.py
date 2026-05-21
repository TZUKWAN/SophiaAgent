"""Tests for Chinese language support in QualitativeEngine (Task A-7).

Verifies:
1. content() with Chinese text produces reasonable word frequencies
2. sentiment() with Chinese text produces correct sentiment
3. content() with English text still works (backward compat)
4. sentiment() with English text still works (backward compat)
"""

import json
import pytest

from sophia.research.qualitative import QualitativeEngine


# =========================================================================
# Test fixtures
# =========================================================================

@pytest.fixture
def engine():
    """QualitativeEngine without LLM provider (uses fallback methods)."""
    return QualitativeEngine(provider=None, store=None)


CN_TEXTS = [
    "社会资本对城市居民的幸福感有显著的正向影响。研究表明社会网络和信任是社会资本的重要组成部分。",
    "教育公平是社会公平的重要基础。高等教育机会的均衡分配需要政策支持和制度保障。",
    "数字鸿沟是信息时代的重要社会问题。信息技术的发展加剧了不同群体之间的数字不平等。",
    "教师的工作压力主要来自教学任务繁重和科研考核要求。",
    "社区治理是基层治理的重要组成部分，需要多元主体共同参与。",
]

CN_POSITIVE_TEXTS = [
    "这项政策极大改善了农民工的就业环境，取得了显著成效，令人欣慰。",
    "教育改革取得了积极进展，学生的综合素质得到了显著提升。",
]

CN_NEGATIVE_TEXTS = [
    "形式主义严重，基层苦不堪言，问题十分突出，令人担忧。",
    "贫困地区的教育资源严重匮乏，学生面临诸多困难。",
]

EN_TEXTS = [
    "Social capital has a significant positive impact on the well-being of urban residents.",
    "Educational equity is an important foundation of social equity.",
    "The digital divide is a major social problem in the information age.",
    "Teachers face work pressure mainly from heavy teaching loads and research requirements.",
]

EN_POSITIVE_TEXTS = [
    "This policy has greatly improved the employment environment and achieved remarkable results.",
    "Educational reform has made positive progress and student outcomes have improved significantly.",
]

EN_NEGATIVE_TEXTS = [
    "Bureaucracy is severe, grassroots workers are suffering, and problems are very serious.",
    "Educational resources in poor areas are seriously insufficient.",
]


# =========================================================================
# Chinese content analysis
# =========================================================================

class TestChineseContentAnalysis:
    """Tests for content() with Chinese text."""

    def test_chinese_content_produces_word_frequencies(self, engine):
        """Chinese content() should produce non-empty word_frequencies."""
        result = json.loads(engine.content({"texts": CN_TEXTS, "language": "zh"}))
        assert "error" not in result, f"Got error: {result.get('error')}"
        assert "word_frequencies" in result
        wf = result["word_frequencies"]
        assert len(wf) > 0, "Expected non-empty word frequencies"

    def test_chinese_content_detects_key_concepts(self, engine):
        """Chinese content() should identify key concepts like social capital, education."""
        result = json.loads(engine.content({"texts": CN_TEXTS, "language": "zh"}))
        key_concepts = result.get("key_concepts", [])
        assert len(key_concepts) > 0, "Expected non-empty key concepts"

    def test_chinese_content_keyword_extraction(self, engine):
        """Chinese content() without explicit keywords should auto-extract keywords."""
        result = json.loads(engine.content({"texts": CN_TEXTS, "language": "zh"}))
        kw_freq = result.get("keyword_frequencies", {})
        assert len(kw_freq) > 0, "Expected non-empty keyword frequencies"
        # Check that Chinese stopwords are filtered out
        for kw in kw_freq:
            assert kw not in ("的", "了", "在", "是", "和"), f"Stopword '{kw}' should be filtered"

    def test_chinese_content_custom_keywords(self, engine):
        """Chinese content() with custom keywords should find their frequencies."""
        result = json.loads(engine.content({
            "texts": CN_TEXTS,
            "keywords": ["社会资本", "教育", "数字鸿沟"],
            "language": "zh",
        }))
        kw_freq = result.get("keyword_frequencies", {})
        assert "社会资本" in kw_freq, f"Expected '社会资本' in keywords, got {list(kw_freq.keys())}"

    def test_chinese_content_language_field(self, engine):
        """Result should contain language field set to 'zh'."""
        result = json.loads(engine.content({"texts": CN_TEXTS, "language": "zh"}))
        assert result.get("language") == "zh"

    def test_chinese_content_auto_detect(self, engine):
        """Auto-detect should identify Chinese texts."""
        result = json.loads(engine.content({"texts": CN_TEXTS, "language": "auto"}))
        assert result.get("language") == "zh"

    def test_chinese_content_total_tokens(self, engine):
        """Chinese content() should report token counts."""
        result = json.loads(engine.content({"texts": CN_TEXTS, "language": "zh"}))
        assert result["total_tokens"] > 0
        assert result["unique_tokens"] > 0
        assert result["n_documents"] == len(CN_TEXTS)


# =========================================================================
# Chinese sentiment analysis
# =========================================================================

class TestChineseSentimentAnalysis:
    """Tests for sentiment() with Chinese text."""

    def test_chinese_positive_sentiment(self, engine):
        """Chinese positive texts should be detected as positive or at least not negative."""
        result = json.loads(engine.sentiment({"texts": CN_POSITIVE_TEXTS, "language": "zh"}))
        assert "error" not in result
        sentiments = result["sentiments"]
        assert len(sentiments) == len(CN_POSITIVE_TEXTS)
        # At least one should be positive
        labels = [s["label"] for s in sentiments]
        assert "positive" in labels or "neutral" in labels, f"Expected positive/neutral, got {labels}"

    def test_chinese_negative_sentiment(self, engine):
        """Chinese negative texts should be detected as negative or at least not positive."""
        result = json.loads(engine.sentiment({"texts": CN_NEGATIVE_TEXTS, "language": "zh"}))
        assert "error" not in result
        sentiments = result["sentiments"]
        assert len(sentiments) == len(CN_NEGATIVE_TEXTS)
        labels = [s["label"] for s in sentiments]
        assert "negative" in labels or "neutral" in labels, f"Expected negative/neutral, got {labels}"

    def test_chinese_sentiment_output_structure(self, engine):
        """Chinese sentiment() should produce correct output structure."""
        result = json.loads(engine.sentiment({"texts": CN_POSITIVE_TEXTS, "language": "zh"}))
        sentiments = result["sentiments"]
        for s in sentiments:
            assert "text_index" in s
            assert "label" in s
            assert s["label"] in ("positive", "negative", "neutral")
            assert "compound" in s
            assert "key_positive_words" in s
            assert "key_negative_words" in s

    def test_chinese_sentiment_method_field(self, engine):
        """Chinese sentiment() should report 'chinese_lexicon' method."""
        result = json.loads(engine.sentiment({"texts": CN_TEXTS, "language": "zh"}))
        assert result["method"] == "chinese_lexicon"

    def test_chinese_sentiment_auto_detect(self, engine):
        """Auto-detect should identify Chinese texts for sentiment."""
        result = json.loads(engine.sentiment({"texts": CN_POSITIVE_TEXTS, "language": "auto"}))
        assert result.get("language") == "zh"
        assert result["method"] == "chinese_lexicon"

    def test_chinese_sentiment_overall_distribution(self, engine):
        """Chinese sentiment() should report overall distribution."""
        all_texts = CN_POSITIVE_TEXTS + CN_NEGATIVE_TEXTS
        result = json.loads(engine.sentiment({"texts": all_texts, "language": "zh"}))
        dist = result["overall_distribution"]
        assert isinstance(dist, dict)
        assert sum(dist.values()) == len(all_texts)

    def test_chinese_sentiment_key_words(self, engine):
        """Chinese sentiment() should report key positive/negative words."""
        all_texts = CN_POSITIVE_TEXTS + CN_NEGATIVE_TEXTS
        result = json.loads(engine.sentiment({"texts": all_texts, "language": "zh"}))
        # At least some positive or negative words should be detected
        top_pos = result.get("key_positive_words", [])
        top_neg = result.get("key_negative_words", [])
        assert len(top_pos) > 0 or len(top_neg) > 0, "Expected some key sentiment words"


# =========================================================================
# English backward compatibility - content
# =========================================================================

class TestEnglishContentBackwardCompat:
    """Tests that English content() still works after Chinese support added."""

    def test_english_content_word_frequencies(self, engine):
        """English content() should still produce word frequencies."""
        result = json.loads(engine.content({"texts": EN_TEXTS}))
        assert "error" not in result
        wf = result.get("word_frequencies", {})
        assert len(wf) > 0

    def test_english_content_language_field(self, engine):
        """English content() should set language to 'en'."""
        result = json.loads(engine.content({"texts": EN_TEXTS}))
        assert result.get("language") == "en"

    def test_english_content_auto_detect(self, engine):
        """Auto-detect should identify English texts."""
        result = json.loads(engine.content({"texts": EN_TEXTS, "language": "auto"}))
        assert result.get("language") == "en"

    def test_english_content_keyword_extraction(self, engine):
        """English content() should still auto-extract keywords."""
        result = json.loads(engine.content({"texts": EN_TEXTS}))
        kw_freq = result.get("keyword_frequencies", {})
        assert len(kw_freq) > 0
        # Check English stopwords are filtered
        for kw in kw_freq:
            assert kw not in ("the", "and", "is", "of", "in"), f"Stopword '{kw}' should be filtered"

    def test_english_content_no_chinese_contamination(self, engine):
        """English content() should not produce any Chinese-specific behavior."""
        result = json.loads(engine.content({"texts": EN_TEXTS, "language": "en"}))
        assert result.get("language") == "en"
        # content() does not have a 'method' field, but language should be 'en'
        # Verify the stopwords used were English (no Chinese stopwords in keyword_freq)
        kw_freq = result.get("keyword_frequencies", {})
        for kw in kw_freq:
            assert kw not in ("的", "了", "在", "是"), f"Chinese stopword '{kw}' should not appear in English analysis"


# =========================================================================
# English backward compatibility - sentiment
# =========================================================================

class TestEnglishSentimentBackwardCompat:
    """Tests that English sentiment() still works after Chinese support added."""

    def test_english_sentiment_produces_results(self, engine):
        """English sentiment() should still produce sentiment results."""
        result = json.loads(engine.sentiment({"texts": EN_POSITIVE_TEXTS}))
        assert "error" not in result
        sentiments = result["sentiments"]
        assert len(sentiments) == len(EN_POSITIVE_TEXTS)

    def test_english_sentiment_output_structure(self, engine):
        """English sentiment() should still have correct output structure."""
        result = json.loads(engine.sentiment({"texts": EN_TEXTS}))
        sentiments = result["sentiments"]
        for s in sentiments:
            assert "text_index" in s
            assert "label" in s
            assert s["label"] in ("positive", "negative", "neutral")
            assert "compound" in s

    def test_english_sentiment_not_chinese_method(self, engine):
        """English sentiment() should NOT use chinese_lexicon method."""
        result = json.loads(engine.sentiment({"texts": EN_TEXTS}))
        assert result["method"] != "chinese_lexicon"

    def test_english_sentiment_auto_detect(self, engine):
        """Auto-detect should identify English texts and NOT use Chinese path."""
        result = json.loads(engine.sentiment({"texts": EN_POSITIVE_TEXTS, "language": "auto"}))
        assert result.get("language") == "en"
        assert result["method"] != "chinese_lexicon"

    def test_english_sentiment_distribution(self, engine):
        """English sentiment() should still report overall distribution."""
        all_texts = EN_POSITIVE_TEXTS + EN_NEGATIVE_TEXTS
        result = json.loads(engine.sentiment({"texts": all_texts}))
        dist = result["overall_distribution"]
        assert sum(dist.values()) == len(all_texts)


# =========================================================================
# Thematic analysis with Chinese text
# =========================================================================

class TestChineseThematicAnalysis:
    """Tests for thematic() with Chinese text (fallback keyword method)."""

    def test_chinese_thematic_keyword_fallback(self, engine):
        """Chinese thematic() without LLM should use keyword fallback."""
        result = json.loads(engine.thematic({"texts": CN_TEXTS, "language": "zh"}))
        assert "error" not in result
        # Should have themes from keyword co-occurrence
        themes = result.get("themes", [])
        assert len(themes) > 0, "Expected at least one theme"

    def test_chinese_thematic_auto_detect(self, engine):
        """Auto-detect should work for Chinese thematic analysis."""
        result = json.loads(engine.thematic({"texts": CN_TEXTS, "language": "auto"}))
        assert "error" not in result
        themes = result.get("themes", [])
        assert len(themes) > 0

    def test_chinese_thematic_coded_segments(self, engine):
        """Chinese thematic() should produce coded segments."""
        result = json.loads(engine.thematic({"texts": CN_TEXTS, "language": "zh"}))
        assert "coded_segments" in result
        assert "theme_frequencies" in result


# =========================================================================
# Grounded theory with Chinese text
# =========================================================================

class TestChineseGroundedTheory:
    """Tests for grounded_code() with Chinese text (fallback methods)."""

    def test_chinese_grounded_open_coding(self, engine):
        """Chinese grounded_code() open stage should work."""
        result = json.loads(engine.grounded_code({"texts": CN_TEXTS, "stage": "open", "language": "zh"}))
        assert "error" not in result
        codes = result.get("codes", [])
        assert len(codes) > 0, "Expected at least some codes"

    def test_chinese_grounded_auto_detect(self, engine):
        """Auto-detect should work for Chinese grounded theory."""
        result = json.loads(engine.grounded_code({"texts": CN_TEXTS, "stage": "open", "language": "auto"}))
        assert "error" not in result
        assert result.get("stage") == "open"
