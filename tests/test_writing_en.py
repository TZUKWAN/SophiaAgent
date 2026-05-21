"""Tests for AcademicEnglishEngine and writing_en tools."""

import json
import os

import pytest

from sophia.research.writing_en import (
    AcademicEnglishEngine,
    en_consistency_check,
    en_cover_letter,
    en_diversify_sentences,
    en_glossary_build,
    en_polish,
    en_readability,
    en_review_response,
)


# ---------------------------------------------------------------------------
# AcademicEnglishEngine unit tests
# ---------------------------------------------------------------------------

class TestPolish:
    def test_polish_vocab_upgrade(self):
        engine = AcademicEnglishEngine()
        text = "We need to get more data. It is a big problem."
        result = engine.polish(text, style="social_science", llm=False)
        assert "error" not in result
        assert result["style"] == "social_science"
        revised = result["revised"]
        assert "obtain" in revised.lower() or "get" in revised.lower()
        assert len(result["edits"]) > 0
        assert result["stats"]["total_edits"] > 0

    def test_polish_redundancy_removal(self):
        engine = AcademicEnglishEngine()
        text = "In order to achieve this goal, we must advance planning."
        result = engine.polish(text, style="social_science", llm=False)
        revised = result["revised"]
        # "in order to" should be reduced to "to"
        assert "in order to" not in revised.lower()
        assert any(e["category"] == "redundancy" for e in result["edits"])

    def test_polish_chinglish_detection(self):
        engine = AcademicEnglishEngine()
        text = "With the rapid development of technology, we can see that AI is important."
        result = engine.polish(text, style="social_science", llm=False)
        # Should detect chinglish patterns
        assert any(e["category"] == "chinglish" for e in result["edits"])

    def test_polish_returns_diff_and_clean(self):
        engine = AcademicEnglishEngine()
        text = "This is a simple sentence."
        result = engine.polish(text, style="social_science", llm=False)
        assert "diff" in result
        assert "clean_version" in result
        assert result["llm_available"] is False

    def test_polish_invalid_style_fallback(self):
        engine = AcademicEnglishEngine()
        text = "Hello world."
        result = engine.polish(text, style="invalid_style", llm=False)
        assert result["style"] == "social_science"


class TestReadability:
    def test_analyze_readability_basic(self):
        engine = AcademicEnglishEngine()
        text = (
            "The study examines the relationship between education and income. "
            "Previous research has shown that higher education levels are associated with greater earnings. "
            "However, the causal mechanism remains unclear. "
            "This paper utilizes a difference-in-differences approach to address this question."
        )
        result = engine.analyze_readability(text, style="social_science")
        assert "error" not in result
        assert result["total_sentences"] == 4
        assert result["total_words"] > 0
        metrics = {m["metric"]: m for m in result["metrics"]}
        assert "avg_sentence_length" in metrics
        assert "passive_voice_ratio" in metrics
        assert "flesch_kincaid" in metrics
        assert "type_token_ratio" in metrics
        assert "awl_coverage" in metrics
        assert "avg_paragraph_length" in metrics

    def test_analyze_readability_empty(self):
        engine = AcademicEnglishEngine()
        result = engine.analyze_readability("", style="social_science")
        assert "error" in result

    def test_analyze_readability_sentence_types(self):
        engine = AcademicEnglishEngine()
        text = (
            "Simple sentence here. "
            "This is compound, and that is too. "
            "Because it rained, the event was cancelled. "
            "The sun shone, but the wind blew, although it was warm."
        )
        result = engine.analyze_readability(text, style="social_science")
        dist = result["sentence_type_distribution"]
        assert "simple" in dist
        assert "compound" in dist
        assert "complex" in dist
        assert "compound-complex" in dist


class TestDiversifySentences:
    def test_detect_repeated_structures(self):
        engine = AcademicEnglishEngine()
        text = (
            "The results show that A is significant. "
            "The results show that B is significant. "
            "The results show that C is significant. "
            "Another finding is that D matters."
        )
        result = engine.diversify_sentences(text)
        assert result["total_sentences"] == 4
        assert len(result["repeated_structures"]) > 0
        rep = result["repeated_structures"][0]
        assert rep["count"] >= 3
        assert "sentence_indices" in rep

    def test_no_repetition(self):
        engine = AcademicEnglishEngine()
        text = "Each sentence here is completely different in structure."
        result = engine.diversify_sentences(text)
        assert len(result["repeated_structures"]) == 0


class TestGlossary:
    def test_build_glossary(self):
        engine = AcademicEnglishEngine()
        text = (
            "Social capital theory suggests that network ties facilitate resource exchange. "
            "Network ties are important for social capital. "
            "Social capital builds network ties."
        )
        glossary = engine.build_glossary(text)
        assert len(glossary) > 0
        terms = [g["term"] for g in glossary]
        assert any("social capital" in t for t in terms)

    def test_save_and_load_glossary(self, tmp_path):
        engine = AcademicEnglishEngine()
        ws = str(tmp_path)
        glossary = [{"term": "test term", "frequency": 5}]
        engine.save_glossary(ws, glossary)
        path = os.path.join(ws, ".sophia", "glossary.json")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded[0]["term"] == "test term"


class TestConsistencyCheck:
    def test_spelling_inconsistency(self):
        engine = AcademicEnglishEngine()
        text = "We studied e-commerce and ecommerce platforms."
        issues = engine.check_consistency(text)
        assert any(i["type"] == "spelling_inconsistency" for i in issues)

    def test_abbreviation_undefined(self):
        engine = AcademicEnglishEngine()
        text = "SEM was used to analyze the data."
        issues = engine.check_consistency(text)
        assert any(i["type"] == "abbreviation_undefined" and i["abbreviation"] == "SEM" for i in issues)

    def test_chinese_alignment(self):
        engine = AcademicEnglishEngine()
        text = "This is a test with 中文 mixed in."
        issues = engine.check_consistency(text)
        assert any(i["type"] == "chinese_english_alignment" for i in issues)


class TestCoverLetter:
    def test_generate_cover_letter(self):
        engine = AcademicEnglishEngine()
        paper_meta = {
            "title": "Test Paper",
            "authors": "John Doe, Jane Smith",
            "abstract": "This is the abstract.",
            "keywords": ["test", "paper"],
            "highlights": ["Highlight 1", "Highlight 2"],
        }
        journal = {
            "name": "Journal of Testing",
            "scope": "Testing and validation research.",
            "editor_name": "Dr. Editor",
        }
        letter = engine.generate_cover_letter(paper_meta, journal)
        assert "Test Paper" in letter
        assert "Journal of Testing" in letter
        assert "Dr. Editor" in letter
        assert "Highlight 1" in letter


class TestReviewResponse:
    def test_generate_review_response(self):
        engine = AcademicEnglishEngine()
        comments = [
            {"comment_id": "R1-1", "comment_text": "Clarify the methodology."},
            {"comment_id": "R1-2", "comment_text": "Add more references."},
        ]
        revisions = [
            {"comment_id": "R1-1", "response": "We have clarified the methodology.", "changes": "Updated Section 3."},
            {"comment_id": "R1-2", "response": "We added 5 new references.", "changes": "Updated reference list."},
        ]
        response = engine.generate_review_response(comments, revisions)
        assert "R1-1" in response
        assert "R1-2" in response
        assert "Clarify the methodology" in response
        assert "Updated Section 3" in response


# ---------------------------------------------------------------------------
# Tool wrapper tests (accept dict args, return JSON strings)
# ---------------------------------------------------------------------------

class TestToolWrappers:
    def test_en_polish_wrapper(self):
        result = json.loads(en_polish({"text": "We need to get data.", "style": "social_science", "llm": False}))
        assert "error" not in result
        assert "revised" in result

    def test_en_polish_missing_text(self):
        result = json.loads(en_polish({"style": "social_science"}))
        assert "error" in result

    def test_en_readability_wrapper(self):
        result = json.loads(en_readability({"text": "This is a test sentence. It has two sentences.", "style": "education"}))
        assert "error" not in result
        assert result["total_sentences"] == 2

    def test_en_readability_missing_text(self):
        result = json.loads(en_readability({"style": "social_science"}))
        assert "error" in result

    def test_en_diversify_sentences_wrapper(self):
        result = json.loads(en_diversify_sentences({"text": "A is good. B is good. C is good."}))
        assert "repeated_structures" in result

    def test_en_glossary_build_wrapper(self, tmp_path):
        ws = str(tmp_path)
        text = (
            "Network ties build social capital. "
            "Network ties facilitate social capital. "
            "Network ties strengthen social capital. "
            "Social capital depends on network ties."
        )
        result = json.loads(en_glossary_build({"text": text}, workspace=ws))
        assert result["total_terms"] > 0
        assert os.path.exists(os.path.join(ws, ".sophia", "glossary.json"))

    def test_en_consistency_check_wrapper(self):
        result = json.loads(en_consistency_check({"text": "We use e-commerce and ecommerce."}))
        assert result["issues_found"] > 0

    def test_en_cover_letter_wrapper(self):
        args = {
            "paper_meta": {
                "title": "T",
                "authors": "A",
                "abstract": "Ab",
                "keywords": ["k"],
                "highlights": ["h"],
            },
            "journal": {"name": "J", "scope": "S", "editor_name": "E"},
        }
        result = json.loads(en_cover_letter(args))
        assert "cover_letter" in result
        assert result["word_count"] > 0

    def test_en_cover_letter_missing_args(self):
        result = json.loads(en_cover_letter({"paper_meta": {}}))
        assert "error" in result

    def test_en_review_response_wrapper(self):
        args = {
            "review_comments": [{"comment_id": "1", "comment_text": "Fix this."}],
            "author_revisions": [{"comment_id": "1", "response": "Fixed.", "changes": "Updated."}],
        }
        result = json.loads(en_review_response(args))
        assert "review_response" in result
        assert result["comments_addressed"] == 1

    def test_en_review_response_missing_comments(self):
        result = json.loads(en_review_response({"author_revisions": []}))
        assert "error" in result
