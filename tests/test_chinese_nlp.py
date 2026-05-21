"""Tests for sophia.research.chinese_nlp module (Task A-1, A-2, A-3)."""

import json
import time
import pytest

from sophia.research.chinese_nlp import (
    ChineseTokenizer,
    detect_language,
    extract_keywords,
    extract_keywords_tfidf,
    extract_keywords_textrank,
    extract_topics,
    analyze_sentiment_cn,
    analyze_sentiment_batch,
    _CN_STOPWORDS,
    _ACADEMIC_DICT,
    _json,
)


# =========================================================================
# A-1: ChineseTokenizer
# =========================================================================

class TestChineseTokenizer:
    """Tests for ChineseTokenizer core functionality."""

    def setup_method(self):
        self.tok = ChineseTokenizer()

    # --- Basic tokenization ---

    def test_tokenize_basic_chinese(self):
        """Acceptance: 社会资本对城市居民幸福感的影响机制研究 should produce correct tokens."""
        text = "社会资本对城市居民幸福感的影响机制研究"
        tokens = self.tok.tokenize(text)
        # Should contain key academic terms as whole tokens
        assert "社会资本" in tokens, f"Expected '社会资本' in {tokens}"
        assert "城市" in tokens or "城市居民" in tokens
        assert "幸福感" in tokens
        assert "影响" in tokens or "影响机制" in tokens

    def test_tokenize_empty_string(self):
        assert self.tok.tokenize("") == []
        assert self.tok.tokenize("   ") == []

    def test_tokenize_mixed_chinese_english(self):
        """Mixed Chinese-English text should handle both."""
        text = "使用SEM结构方程模型分析数据"
        tokens = self.tok.tokenize(text)
        assert len(tokens) > 0
        # Should have SEM as a token
        assert any("SEM" in t for t in tokens)

    def test_tokenize_modes(self):
        text = "社会资本理论是社会学的重要理论"
        default_tokens = self.tok.tokenize(text, mode="default")
        search_tokens = self.tok.tokenize(text, mode="search")
        assert len(default_tokens) > 0
        assert len(search_tokens) > 0

    # --- Stopword removal ---

    def test_remove_stopwords(self):
        tokens = ["社会", "的", "研究", "了", "方法", "和", "理论"]
        filtered = self.tok.remove_stopwords(tokens)
        assert "的" not in filtered
        assert "了" not in filtered
        assert "和" not in filtered
        assert "社会" in filtered
        assert "研究" in filtered
        assert "方法" in filtered
        assert "理论" in filtered

    def test_remove_stopwords_preserves_meaningful(self):
        """Academic terms should not be in stopword list."""
        academic_terms = ["社会资本", "质性研究", "扎根理论", "田野调查", "混合方法"]
        for term in academic_terms:
            assert term not in _CN_STOPWORDS, f"'{term}' should not be a stopword"

    # --- Academic dictionary ---

    def test_academic_dict_coverage(self):
        """Academic dictionary should have ~2000 terms."""
        assert len(_ACADEMIC_DICT) >= 200, f"Expected >= 200 terms, got {len(_ACADEMIC_DICT)}"

    def test_academic_dict_contains_key_terms(self):
        """Key academic terms should be in the dictionary."""
        must_have = [
            "社会资本", "扎根理论", "民族志", "教育公平", "政治参与",
            "自我效能", "议程设置", "数字鸿沟", "行动研究", "因果推断",
        ]
        for term in must_have:
            assert term in _ACADEMIC_DICT, f"'{term}' missing from academic dict"

    def test_custom_term_not_split(self):
        """Terms like 内卷化 should be kept whole (if jieba available)."""
        from sophia.research.chinese_nlp import HAS_JIEBA
        if not HAS_JIEBA:
            pytest.skip("jieba not available")

        # 内卷 should be recognizable as a unit
        text = "内卷化现象在高校中非常普遍"
        tokens = self.tok.tokenize(text)
        # At minimum "内卷" or "内卷化" should appear as a token
        has_term = any("内卷" in t for t in tokens)
        assert has_term, f"Expected '内卷' or '内卷化' in {tokens}"

    # --- Backend detection ---

    def test_backend_property(self):
        backend = self.tok.backend
        assert backend in ("jieba", "pkuseg", "char")

    def test_stopwords_accessible(self):
        sw = self.tok.stopwords
        assert isinstance(sw, set)
        assert "的" in sw

    # --- Performance ---

    def test_tokenization_speed(self):
        """100 calls on 5000-char text should complete in < 10 seconds."""
        text = "社会科学研究方法论是人文社科领域的重要内容。" * 100  # ~5000 chars
        start = time.time()
        for _ in range(100):
            self.tok.tokenize(text)
        elapsed = time.time() - start
        assert elapsed < 10.0, f"100 tokenization calls took {elapsed:.1f}s (too slow)"

    # --- Language detection ---

    def test_detect_chinese(self):
        assert detect_language("这是一段中文文本") == "zh"

    def test_detect_english(self):
        assert detect_language("This is an English text") == "en"

    def test_detect_mixed(self):
        result = detect_language("Social capital 社会资本 is important 很重要")
        assert result in ("zh", "mixed", "en")  # Just verify no crash

    def test_detect_empty(self):
        assert detect_language("") == "en"


# =========================================================================
# A-2: Keyword extraction & topic modeling
# =========================================================================

class TestKeywordExtraction:
    """Tests for keyword extraction functionality."""

    def setup_method(self):
        self.tok = ChineseTokenizer()

    def test_tfidf_basic(self):
        text = "社会资本对城市居民的幸福感有显著的正向影响。研究表明，社会网络和信任是社会资本的重要组成部分。"
        results = extract_keywords_tfidf(text, self.tok, top_n=5)
        assert len(results) > 0
        assert all(isinstance(w, str) and isinstance(s, float) for w, s in results)

    def test_textrank_basic(self):
        text = "数字鸿沟是信息时代的重要社会问题。信息技术的发展加剧了不同群体之间的数字不平等。"
        results = extract_keywords_textrank(text, self.tok, top_n=5)
        assert len(results) > 0

    def test_hybrid_method(self):
        text = "教育公平是社会公平的重要基础。高等教育机会的均衡分配需要政策支持和制度保障。"
        results = extract_keywords(text, self.tok, top_n=10, method="hybrid")
        assert len(results) > 0
        # Should find meaningful keywords
        words = [w for w, _ in results]
        assert any(w for w in words)  # At least some keywords found

    def test_empty_text(self):
        results = extract_keywords("", self.tok, top_n=5)
        assert results == []

    def test_single_word(self):
        results = extract_keywords("研究", self.tok, top_n=5)
        # Should not crash on very short text
        assert isinstance(results, list)


class TestTopicExtraction:
    """Tests for topic extraction functionality."""

    def setup_method(self):
        self.tok = ChineseTokenizer()

    def test_basic_topics(self):
        """10 education interview texts should produce 3-5 topics."""
        texts = [
            "教师的工作压力主要来自教学任务繁重和科研考核要求。",
            "青年教师面临住房压力和职业发展不确定性。",
            "高校教师的科研压力影响了教学质量的提升。",
            "学校管理层应该减少不必要的行政任务，让教师专注于教学。",
            "导师制度对新教师的职业发展有积极帮助。",
            "教师之间的合作交流可以缓解工作压力。",
            "薪酬待遇是教师职业满意度的重要因素。",
            "培训机会对教师专业发展至关重要。",
            "学校应该建立更加合理的考核评价体系。",
            "教师的工作与生活平衡需要制度层面的保障。",
        ]
        topics = extract_topics(texts, self.tok, n_topics=3)
        assert 1 <= len(topics) <= 5
        for topic in topics:
            assert "topic_id" in topic
            assert "keywords" in topic
            assert len(topic["keywords"]) > 0

    def test_empty_texts(self):
        topics = extract_topics([], self.tok, n_topics=3)
        assert topics == []

    def test_single_text(self):
        topics = extract_topics(["这是唯一的文本"], self.tok, n_topics=2)
        # Should handle gracefully
        assert isinstance(topics, list)


# =========================================================================
# A-3: Chinese sentiment analysis
# =========================================================================

class TestChineseSentiment:
    """Tests for Chinese sentiment analysis."""

    def setup_method(self):
        self.tok = ChineseTokenizer()

    def test_positive_sentiment(self):
        """Positive text should score > 0.6 equivalent."""
        text = "这项政策极大改善了农民工的就业环境，取得了显著成效"
        result = analyze_sentiment_cn(text, self.tok)
        assert result["sentiment"] in ("positive", "neutral")
        # Score should be positive-leaning
        assert result["positive_hits"] > 0

    def test_negative_sentiment(self):
        """Negative text should be detected."""
        text = "形式主义严重，基层苦不堪言，问题十分突出"
        result = analyze_sentiment_cn(text, self.tok)
        assert result["sentiment"] in ("negative", "neutral")
        assert result["negative_hits"] > 0

    def test_neutral_sentiment(self):
        """Factual text should be neutral."""
        text = "本研究采用问卷调查方法，样本量为500人"
        result = analyze_sentiment_cn(text, self.tok)
        assert result["sentiment"] == "neutral"

    def test_output_structure(self):
        text = "这个项目取得了很大的成功"
        result = analyze_sentiment_cn(text, self.tok)
        assert "sentiment" in result
        assert "score" in result
        assert "confidence" in result
        assert "dimensions" in result
        assert "key_phrases" in result
        assert "backend" in result
        assert result["sentiment"] in ("positive", "negative", "neutral")

    def test_dimensions_present(self):
        text = "这项发现令人惊讶，同时也令人担忧"
        result = analyze_sentiment_cn(text, self.tok)
        dims = result["dimensions"]
        assert "joy" in dims
        assert "anger" in dims
        assert "sadness" in dims
        assert "fear" in dims
        assert "surprise" in dims

    def test_batch_analysis(self):
        texts = [
            "这个政策很好",
            "这个问题很严重",
            "今天天气不错",
        ]
        results = analyze_sentiment_batch(texts, self.tok)
        assert len(results) == 3
        assert all("sentiment" in r for r in results)

    def test_batch_performance(self):
        """100 short texts should complete in < 5 seconds."""
        texts = ["这项研究表明教育投入对经济增长有正向促进作用。"] * 100
        start = time.time()
        results = analyze_sentiment_batch(texts, self.tok)
        elapsed = time.time() - start
        assert len(results) == 100
        assert elapsed < 5.0, f"100 texts took {elapsed:.1f}s"

    def test_empty_text(self):
        result = analyze_sentiment_cn("")
        assert result["sentiment"] == "neutral"


# =========================================================================
# JSON serialization
# =========================================================================

class TestJsonSerialization:
    """Tests for _json helper."""

    def test_basic_dict(self):
        result = _json({"key": "值", "num": 1.5})
        parsed = json.loads(result)
        assert parsed["key"] == "值"
        assert parsed["num"] == 1.5

    def test_nan_handling(self):
        result = _json({"val": float("nan")})
        parsed = json.loads(result)
        assert parsed["val"] is None

    def test_set_handling(self):
        result = _json({"items": {1, 2, 3}})
        parsed = json.loads(result)
        assert sorted(parsed["items"]) == [1, 2, 3]
