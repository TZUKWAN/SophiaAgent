"""Tests for QualitativeEngine: thematic, content, grounded theory, sentiment, reliability."""
import json
import math

import numpy as np
import pytest

from sophia.research.qualitative import QualitativeEngine, HAS_VADER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(result: str) -> dict:
    """Parse JSON result string into a dict."""
    return json.loads(result)


@pytest.fixture
def engine():
    return QualitativeEngine()


@pytest.fixture
def sample_texts():
    """Short interview-like text segments."""
    return [
        "The training program was excellent and I learned a great deal about leadership. "
        "The instructor was supportive and provided valuable feedback throughout the course.",
        "I found the communication between team members to be poor and frustrating. "
        "We had many problems with the scheduling and organization of tasks.",
        "The new policy has brought significant improvements to our workflow. "
        "The efficiency of the team has improved dramatically since implementation.",
        "Leadership development is important for organizational success. "
        "Good leaders create positive environments and motivate their teams effectively.",
        "The feedback from management was disappointing and unhelpful. "
        "There were serious issues with the way the project was managed.",
    ]


@pytest.fixture
def long_texts():
    """Longer text segments for content analysis."""
    return [
        "Natural language processing involves computational techniques for analyzing "
        "and generating human language. Machine learning algorithms are commonly used "
        "in NLP tasks such as sentiment analysis, named entity recognition, and text "
        "classification. Deep learning has revolutionized the field of natural language "
        "processing with transformer models achieving state-of-the-art results.",
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Supervised learning, unsupervised "
        "learning, and reinforcement learning are the three main types of machine "
        "learning. Deep learning uses neural networks with multiple layers to learn "
        "complex patterns in data.",
        "Data science combines statistics, programming, and domain knowledge to "
        "extract insights from data. Data scientists use various tools and techniques "
        "including machine learning, data visualization, and statistical analysis. "
        "The field has grown rapidly with the increasing availability of big data.",
    ]


# ===================================================================
# Thematic analysis tests
# ===================================================================

class TestThematic:

    def test_inductive_themes_generated(self, engine, sample_texts):
        res = _parse(engine.thematic({"texts": sample_texts}))
        assert "themes" in res
        assert len(res["themes"]) > 0
        assert "coded_segments" in res
        assert "theme_frequencies" in res
        assert res["approach"] == "inductive"

    def test_inductive_theme_structure(self, engine, sample_texts):
        res = _parse(engine.thematic({"texts": sample_texts}))
        for theme in res["themes"]:
            assert "id" in theme
            assert "label" in theme
            assert "description" in theme
            assert "keywords" in theme

    def test_inductive_with_n_themes(self, engine, sample_texts):
        res = _parse(engine.thematic({"texts": sample_texts, "n_themes": 3}))
        assert len(res["themes"]) <= 3

    def test_deductive_with_existing_themes(self, engine, sample_texts):
        res = _parse(engine.thematic({
            "texts": sample_texts,
            "approach": "deductive",
            "existing_themes": ["leadership", "communication", "performance"],
        }))
        assert "themes" in res
        assert len(res["themes"]) == 3
        labels = [t["label"].lower() for t in res["themes"]]
        assert any("leadership" in l for l in labels)

    def test_deductive_coded_segments(self, engine, sample_texts):
        res = _parse(engine.thematic({
            "texts": sample_texts,
            "approach": "deductive",
            "existing_themes": ["training", "feedback", "leadership"],
        }))
        assert "coded_segments" in res
        assert len(res["coded_segments"]) > 0
        for seg in res["coded_segments"]:
            assert "text_index" in seg
            assert "theme_ids" in seg

    def test_empty_texts_error(self, engine):
        res = _parse(engine.thematic({"texts": []}))
        assert "error" in res

    def test_single_text(self, engine):
        res = _parse(engine.thematic({
            "texts": ["This is a great and wonderful day for learning and education."]
        }))
        assert "themes" in res
        assert len(res["themes"]) >= 1


# ===================================================================
# Content analysis tests
# ===================================================================

class TestContent:

    def test_basic_content_analysis(self, engine, long_texts):
        res = _parse(engine.content({"texts": long_texts}))
        assert "word_frequencies" in res
        assert "keyword_frequencies" in res
        assert "co_occurrence_matrix" in res
        assert "key_concepts" in res
        assert res["n_documents"] == 3

    def test_word_frequencies_present(self, engine, long_texts):
        res = _parse(engine.content({"texts": long_texts}))
        wf = res["word_frequencies"]
        assert isinstance(wf, dict)
        # "learning" should appear in these texts
        assert "learning" in wf

    def test_keyword_frequencies_with_provided_keywords(self, engine, long_texts):
        res = _parse(engine.content({
            "texts": long_texts,
            "keywords": ["machine", "learning", "data", "neural"],
        }))
        kf = res["keyword_frequencies"]
        assert "machine" in kf
        assert "learning" in kf
        assert kf["machine"] > 0
        assert kf["learning"] > 0

    def test_co_occurrence_matrix(self, engine, long_texts):
        res = _parse(engine.content({
            "texts": long_texts,
            "keywords": ["machine", "learning", "data"],
            "window": 5,
        }))
        co = res["co_occurrence_matrix"]
        assert isinstance(co, dict)
        # "machine" and "learning" should co-occur
        if "machine" in co:
            assert "learning" in co["machine"]

    def test_key_concepts(self, engine, long_texts):
        res = _parse(engine.content({"texts": long_texts}))
        kc = res["key_concepts"]
        assert isinstance(kc, list)
        assert len(kc) > 0

    def test_min_freq_filter(self, engine, long_texts):
        res = _parse(engine.content({"texts": long_texts, "min_freq": 5}))
        wf = res["word_frequencies"]
        for word, count in wf.items():
            assert count >= 5

    def test_empty_texts_error(self, engine):
        res = _parse(engine.content({"texts": []}))
        assert "error" in res

    def test_total_tokens_reported(self, engine, long_texts):
        res = _parse(engine.content({"texts": long_texts}))
        assert res["total_tokens"] > 0
        assert res["unique_tokens"] > 0


# ===================================================================
# Grounded theory tests
# ===================================================================

class TestGroundedCode:

    def test_open_coding(self, engine, sample_texts):
        res = _parse(engine.grounded_code({
            "texts": sample_texts,
            "stage": "open",
        }))
        assert res["stage"] == "open"
        assert "codes" in res
        assert len(res["codes"]) > 0
        assert "code_frequencies" in res

    def test_open_code_structure(self, engine, sample_texts):
        res = _parse(engine.grounded_code({
            "texts": sample_texts,
            "stage": "open",
        }))
        for code in res["codes"]:
            assert "code" in code
            assert "frequency" in code
            assert code["frequency"] >= 1

    def test_axial_coding(self, engine, sample_texts):
        res = _parse(engine.grounded_code({
            "texts": sample_texts,
            "stage": "axial",
            "existing_codes": ["leadership", "feedback", "training", "communication", "performance"],
        }))
        assert res["stage"] == "axial"
        assert "categories" in res
        assert "core_category" in res
        assert "relationships" in res

    def test_axial_with_core(self, engine, sample_texts):
        res = _parse(engine.grounded_code({
            "texts": sample_texts,
            "stage": "axial",
            "existing_codes": ["leadership", "feedback", "training"],
            "axial_core": "organizational effectiveness",
        }))
        assert res["core_category"] == "organizational effectiveness"

    def test_selective_coding(self, engine, sample_texts):
        res = _parse(engine.grounded_code({
            "texts": sample_texts,
            "stage": "selective",
            "existing_codes": ["leadership", "feedback", "training", "communication"],
        }))
        assert res["stage"] == "selective"
        assert "core_category" in res
        assert "subcategories" in res

    def test_invalid_stage_error(self, engine, sample_texts):
        res = _parse(engine.grounded_code({
            "texts": sample_texts,
            "stage": "invalid_stage",
        }))
        assert "error" in res

    def test_empty_texts_error(self, engine):
        res = _parse(engine.grounded_code({"texts": [], "stage": "open"}))
        assert "error" in res


# ===================================================================
# Sentiment analysis tests
# ===================================================================

class TestSentiment:

    def test_positive_text(self, engine):
        res = _parse(engine.sentiment({
            "texts": ["This is an excellent and amazing product! I love it so much."]
        }))
        assert "sentiments" in res
        assert res["sentiments"][0]["label"] == "positive"
        assert res["sentiments"][0]["compound"] > 0

    def test_negative_text(self, engine):
        res = _parse(engine.sentiment({
            "texts": ["This is terrible and awful. I hate this horrible product."]
        }))
        assert res["sentiments"][0]["label"] == "negative"
        assert res["sentiments"][0]["compound"] < 0

    def test_neutral_text(self, engine):
        res = _parse(engine.sentiment({
            "texts": ["The meeting is scheduled for Tuesday at three o'clock."]
        }))
        assert res["sentiments"][0]["label"] == "neutral"

    def test_overall_distribution(self, engine):
        texts = [
            "Great job, well done!",
            "This is terrible and disappointing.",
            "The report is due tomorrow.",
            "I am very happy with the results.",
            "The service was poor and slow.",
        ]
        res = _parse(engine.sentiment({"texts": texts}))
        dist = res["overall_distribution"]
        assert "positive" in dist
        assert "negative" in dist
        total = sum(dist.values())
        assert total == 5

    def test_average_compound(self, engine):
        texts = ["Great excellent wonderful!", "Terrible awful bad."]
        res = _parse(engine.sentiment({"texts": texts}))
        assert "average_compound" in res
        assert isinstance(res["average_compound"], float)

    def test_key_words_extracted(self, engine):
        res = _parse(engine.sentiment({
            "texts": [
                "The product is excellent and great.",
                "The service was terrible and disappointing.",
            ]
        }))
        assert "key_positive_words" in res
        assert "key_negative_words" in res

    def test_method_reported(self, engine):
        res = _parse(engine.sentiment({"texts": ["Hello world."]}))
        assert "method" in res
        assert res["method"] in ("vader", "simple_lexicon")

    def test_empty_texts_error(self, engine):
        res = _parse(engine.sentiment({"texts": []}))
        assert "error" in res

    def test_multiple_texts(self, engine):
        texts = [
            "I am happy.",
            "I am sad.",
            "I am okay.",
        ]
        res = _parse(engine.sentiment({"texts": texts}))
        assert len(res["sentiments"]) == 3
        for s in res["sentiments"]:
            assert "label" in s
            assert "compound" in s


# ===================================================================
# Coding reliability tests
# ===================================================================

class TestCodingReliability:

    def test_perfect_agreement(self, engine):
        codes = ["A", "B", "A", "B", "A", "B"]
        res = _parse(engine.coding_reliability({
            "coder1": codes, "coder2": codes,
        }))
        assert res["kappa"] == 1.0
        assert res["agreement_rate"] == 1.0
        assert res["interpretation"] == "almost perfect"

    def test_no_agreement(self, engine):
        res = _parse(engine.coding_reliability({
            "coder1": ["A", "A", "A", "A"],
            "coder2": ["B", "B", "B", "B"],
        }))
        assert res["kappa"] <= 0
        assert res["agreement_rate"] == 0.0

    def test_partial_agreement(self, engine):
        res = _parse(engine.coding_reliability({
            "coder1": ["A", "B", "A", "B", "A", "B"],
            "coder2": ["A", "A", "A", "B", "B", "B"],
        }))
        assert 0.0 < res["agreement_rate"] < 1.0
        assert "kappa" in res

    def test_confusion_matrix_shape(self, engine):
        res = _parse(engine.coding_reliability({
            "coder1": ["A", "B", "C", "A", "B", "C"],
            "coder2": ["A", "B", "C", "A", "B", "C"],
        }))
        matrix = res["confusion_matrix"]
        assert len(matrix) == 3  # 3 categories
        assert len(matrix[0]) == 3
        # Diagonal should have counts
        assert matrix[0][0] == 2
        assert matrix[1][1] == 2
        assert matrix[2][2] == 2

    def test_numeric_codes(self, engine):
        res = _parse(engine.coding_reliability({
            "coder1": [1, 2, 1, 2, 1, 2],
            "coder2": [1, 2, 1, 2, 1, 2],
        }))
        assert res["kappa"] == 1.0
        assert res["categories"] == ["1", "2"]

    def test_length_mismatch_error(self, engine):
        res = _parse(engine.coding_reliability({
            "coder1": ["A", "B", "C"],
            "coder2": ["A", "B"],
        }))
        assert "error" in res

    def test_missing_coder_error(self, engine):
        res = _parse(engine.coding_reliability({"coder1": ["A", "B"]}))
        assert "error" in res

    def test_empty_codes_error(self, engine):
        res = _parse(engine.coding_reliability({
            "coder1": [], "coder2": [],
        }))
        assert "error" in res

    def test_interpretation_scale(self, engine):
        """Test that all interpretation labels are valid."""
        res = _parse(engine.coding_reliability({
            "coder1": ["A", "B", "A", "B"],
            "coder2": ["A", "B", "A", "B"],
        }))
        valid = {"poor", "slight", "fair", "moderate", "substantial", "almost perfect"}
        assert res["interpretation"] in valid

    def test_krippendorff_alpha_present(self, engine):
        res = _parse(engine.coding_reliability({
            "coder1": ["A", "B", "A", "B"],
            "coder2": ["A", "B", "A", "B"],
        }))
        assert "krippendorff_alpha" in res
        assert res["krippendorff_alpha"] is not None

    def test_three_coders_via_list(self, engine):
        res = _parse(engine.coding_reliability({
            "coders": [
                ["A", "B", "A", "B"],
                ["A", "B", "A", "B"],
                ["A", "B", "A", "B"],
            ],
        }))
        assert res["kappa"] == 1.0
        assert res["krippendorff_n_coders"] == 3

    def test_krippendorff_three_coders_partial(self, engine):
        res = _parse(engine.coding_reliability({
            "coders": [
                ["A", "A", "B", "B"],
                ["A", "B", "B", "A"],
                ["A", "A", "B", "B"],
            ],
        }))
        assert "krippendorff_alpha" in res
        assert res["krippendorff_n_coders"] == 3

    def test_krippendorff_ordinal_level(self, engine):
        res = _parse(engine.coding_reliability({
            "coder1": ["1", "2", "3", "1"],
            "coder2": ["1", "2", "3", "1"],
            "level": "ordinal",
        }))
        assert res["krippendorff_level"] == "ordinal"
        assert res["krippendorff_alpha"] == 1.0
