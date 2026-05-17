"""Tests for LLMEngine -- comprehensive pytest suite using mock provider."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sophia.research.llm import (
    LLMEngine,
    _edit_distance_ratio,
    _extract_score_from_text,
    _jaccard_similarity,
    _manual_bleu,
    _rouge_n,
    _tokenize,
    _word_overlap_pct,
)


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class MockProvider:
    """Simulates an LLM provider for testing."""

    def __init__(self, responses=None, delay=0.0):
        self.responses = responses or {}
        self.default_response = "This is a mock response."
        self.call_log = []

    def run(self, prompt=None, **kwargs):
        self.call_log.append(prompt)
        # Return predefined response based on prompt content
        for key, val in self.responses.items():
            if key in (prompt or ""):
                return val
        return self.default_response


class ScoringMockProvider:
    """Returns judge-like responses with numeric scores."""

    def run(self, prompt=None, **kwargs):
        if "rate" in (prompt or "").lower() or "score" in (prompt or "").lower():
            return "Score: 4\nThe response is helpful and well-structured."
        return "This is a helpful response about the topic."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine_no_provider():
    """LLMEngine without provider."""
    return LLMEngine(provider=None)


@pytest.fixture
def engine_with_provider():
    """LLMEngine with mock provider."""
    provider = MockProvider(responses={
        "capital of France": "Paris is the capital of France.",
        "What is AI": "Artificial Intelligence is the simulation of human intelligence by machines.",
    })
    return LLMEngine(provider=provider)


@pytest.fixture
def engine_with_judge():
    """LLMEngine with scoring mock provider."""
    return LLMEngine(provider=ScoringMockProvider())


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestHelperFunctions:

    def test_tokenize_basic(self):
        tokens = _tokenize("Hello World, this is a test!")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_tokenize_empty(self):
        assert _tokenize("") == []

    def test_jaccard_identical(self):
        assert _jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial(self):
        result = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert result == pytest.approx(0.5, abs=0.01)

    def test_jaccard_empty(self):
        assert _jaccard_similarity(set(), set()) == 1.0

    def test_edit_distance_identical(self):
        assert _edit_distance_ratio("hello", "hello") == 1.0

    def test_edit_distance_completely_different(self):
        assert _edit_distance_ratio("abc", "xyz") < 0.5

    def test_edit_distance_empty(self):
        assert _edit_distance_ratio("", "") == 1.0
        assert _edit_distance_ratio("", "abc") == 0.0

    def test_edit_distance_one_char_diff(self):
        ratio = _edit_distance_ratio("hello", "hallo")
        assert ratio == pytest.approx(0.8, abs=0.05)

    def test_word_overlap_full(self):
        assert _word_overlap_pct(
            ["the", "cat", "sat"],
            ["the", "cat", "sat", "on", "the", "mat"]
        ) == 1.0

    def test_word_overlap_partial(self):
        result = _word_overlap_pct(
            ["the", "dog", "ran"],
            ["the", "cat", "sat"]
        )
        assert result == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_word_overlap_empty_response(self):
        assert _word_overlap_pct([], ["some", "context"]) == 1.0

    def test_rouge_n_identical(self):
        tokens = ["the", "cat", "sat"]
        assert _rouge_n(tokens, tokens, 1) == 1.0

    def test_rouge_n_no_overlap(self):
        assert _rouge_n(["a", "b"], ["c", "d"], 1) == 0.0

    def test_rouge_n_partial(self):
        result = _rouge_n(["a", "b", "c"], ["b", "c", "d"], 1)
        assert result == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_manual_bleu_identical(self):
        tokens = ["the", "cat", "sat"]
        assert _manual_bleu(tokens, tokens) > 0.9

    def test_manual_bleu_no_overlap(self):
        assert _manual_bleu(["a", "b"], ["c", "d"]) == 0.0

    def test_extract_score_from_text(self):
        assert _extract_score_from_text("Score: 4\nGood response.", 5) == 4.0
        assert _extract_score_from_text("Rating: 3.5", 5) == 3.5
        assert _extract_score_from_text("I give it 4/5", 5) == 4.0

    def test_extract_score_none_on_invalid(self):
        result = _extract_score_from_text("No score here but number 99", 5)
        # First number found is 99 which exceeds scale, returns None
        assert result is None


# ===========================================================================
# prompt_test
# ===========================================================================

class TestPromptTest:

    def test_prompt_test_basic(self, engine_with_provider):
        result = json.loads(engine_with_provider.prompt_test({
            "variants": [
                {"name": "v1", "template": "Tell me about {input}"},
                {"name": "v2", "template": "Explain {input} in detail"},
            ],
            "test_inputs": [
                {"input": "capital of France"},
                {"input": "What is AI"},
            ],
        }))
        assert "variants" in result
        assert "comparison" in result
        assert len(result["comparison"]) == 2
        assert result["n_variants"] == 2
        assert result["n_test_inputs"] == 2

    def test_prompt_test_no_provider(self, engine_no_provider):
        result = json.loads(engine_no_provider.prompt_test({
            "variants": [{"name": "v1", "template": "{input}"}],
            "test_inputs": [{"input": "test"}],
        }))
        assert "error" in result

    def test_prompt_test_no_variants(self, engine_with_provider):
        result = json.loads(engine_with_provider.prompt_test({
            "test_inputs": [{"input": "test"}],
        }))
        assert "error" in result

    def test_prompt_test_no_inputs(self, engine_with_provider):
        result = json.loads(engine_with_provider.prompt_test({
            "variants": [{"name": "v1", "template": "{input}"}],
        }))
        assert "error" in result

    def test_prompt_test_latency(self, engine_with_provider):
        result = json.loads(engine_with_provider.prompt_test({
            "variants": [{"name": "v1", "template": "{input}"}],
            "test_inputs": [{"input": "capital of France"}],
        }))
        variant_data = result["variants"]["v1"]
        assert "avg_latency_s" in variant_data
        assert isinstance(variant_data["avg_latency_s"], float)


# ===========================================================================
# evaluate
# ===========================================================================

class TestEvaluate:

    def test_evaluate_basic(self, engine_no_provider):
        result = json.loads(engine_no_provider.evaluate({
            "responses": [
                "This is a test response with some words.",
                "Another response here.",
            ],
        }))
        assert result["aggregate"]["n_responses"] == 2
        assert "per_response" in result
        assert "aggregate" in result
        assert result["aggregate"]["avg_word_count"] > 0

    def test_evaluate_with_references(self, engine_no_provider):
        result = json.loads(engine_no_provider.evaluate({
            "responses": ["The cat sat on the mat."],
            "references": ["A feline rested upon the rug."],
        }))
        per = result["per_response"][0]
        assert "jaccard_similarity" in per
        assert "word_overlap" in per

    def test_evaluate_no_responses(self, engine_no_provider):
        result = json.loads(engine_no_provider.evaluate({}))
        assert "error" in result

    def test_evaluate_metrics(self, engine_no_provider):
        result = json.loads(engine_no_provider.evaluate({
            "responses": ["Hello world. This is a test.", "Short."],
        }))
        per0 = result["per_response"][0]
        assert per0["word_count"] == 6
        assert per0["sentence_count"] >= 1
        assert per0["char_count"] > 0


# ===========================================================================
# llm_judge
# ===========================================================================

class TestLLMJudge:

    def test_judge_with_provider(self, engine_with_judge):
        result = json.loads(engine_with_judge.llm_judge({
            "responses": [
                {"prompt": "What is AI?", "response": "AI is artificial intelligence."},
                {"prompt": "What is ML?", "response": "ML is machine learning."},
            ],
            "criteria": "helpfulness",
            "scale": "1-5",
        }))
        assert "per_response" in result
        assert "average_score" in result
        assert result["n_scored"] == 2
        assert result["scale"] == "1-5"

    def test_judge_no_provider_heuristic(self, engine_no_provider):
        result = json.loads(engine_no_provider.llm_judge({
            "responses": [
                {"prompt": "What is AI?", "response": "AI is artificial intelligence."},
            ],
            "criteria": "accuracy",
            "scale": "1-10",
        }))
        assert "per_response" in result
        score = result["per_response"][0]["score"]
        assert score is not None
        assert 1 <= score <= 10

    def test_judge_custom_prompt(self, engine_with_judge):
        result = json.loads(engine_with_judge.llm_judge({
            "responses": [
                {"prompt": "test", "response": "test response"},
            ],
            "judge_prompt": "Rate this: {response}. Use 1-5 scale.",
            "scale": "1-5",
        }))
        assert result["n_scored"] == 1

    def test_judge_no_responses(self, engine_no_provider):
        result = json.loads(engine_no_provider.llm_judge({}))
        assert "error" in result

    def test_judge_distribution(self, engine_with_judge):
        result = json.loads(engine_with_judge.llm_judge({
            "responses": [
                {"prompt": "Q1", "response": "Good answer"},
                {"prompt": "Q2", "response": "Another good answer"},
            ],
        }))
        assert "distribution" in result
        assert isinstance(result["distribution"], dict)


# ===========================================================================
# rag_eval
# ===========================================================================

class TestRagEval:

    def test_rag_eval_basic(self, engine_no_provider):
        result = json.loads(engine_no_provider.rag_eval({
            "queries": ["What is machine learning?"],
            "contexts": ["Machine learning is a branch of artificial intelligence that allows systems to learn."],
            "responses": ["ML is a type of AI that enables learning from data."],
            "references": ["Machine learning lets computers learn automatically."],
        }))
        assert result["aggregate"]["n_queries"] == 1
        per = result["per_query"][0]
        assert "faithfulness" in per
        assert "relevance" in per
        assert "completeness" in per

    def test_rag_eval_faithfulness(self, engine_no_provider):
        result = json.loads(engine_no_provider.rag_eval({
            "queries": ["What is Python?"],
            "contexts": ["Python is a programming language created by Guido van Rossum."],
            "responses": ["Python is a programming language created by Guido van Rossum."],
        }))
        # Response fully contained in context
        assert result["per_query"][0]["faithfulness"] >= 0.5

    def test_rag_eval_no_queries(self, engine_no_provider):
        result = json.loads(engine_no_provider.rag_eval({}))
        assert "error" in result

    def test_rag_eval_aggregate(self, engine_no_provider):
        result = json.loads(engine_no_provider.rag_eval({
            "queries": ["Q1", "Q2"],
            "contexts": ["Context about topic one", "Context about topic two"],
            "responses": ["Response about topic one", "Response about topic two"],
            "references": ["Reference one", "Reference two"],
        }))
        agg = result["aggregate"]
        assert "avg_faithfulness" in agg
        assert "avg_relevance" in agg
        assert "avg_completeness" in agg


# ===========================================================================
# benchmark
# ===========================================================================

class TestBenchmark:

    def test_benchmark_no_provider(self, engine_no_provider):
        result = json.loads(engine_no_provider.benchmark({
            "dataset": [{"input": "test", "expected": "result"}],
        }))
        assert "error" in result

    def test_benchmark_no_dataset(self, engine_with_provider):
        result = json.loads(engine_with_provider.benchmark({}))
        assert "error" in result

    def test_benchmark_basic(self, engine_with_provider):
        result = json.loads(engine_with_provider.benchmark({
            "dataset": [
                {"input": "capital of France", "expected": "Paris"},
                {"input": "What is AI", "expected": "Artificial Intelligence"},
            ],
        }))
        assert result["n_samples"] == 2
        assert "exact_match_accuracy" in result
        assert "contains_match_accuracy" in result
        assert "per_sample" in result

    def test_benchmark_max_samples(self, engine_with_provider):
        result = json.loads(engine_with_provider.benchmark({
            "dataset": [
                {"input": "Q1", "expected": "A1"},
                {"input": "Q2", "expected": "A2"},
                {"input": "Q3", "expected": "A3"},
            ],
            "max_samples": 2,
        }))
        assert result["n_samples"] == 2


# ===========================================================================
# quality_score
# ===========================================================================

class TestQualityScore:

    def test_quality_score_basic(self, engine_no_provider):
        result = json.loads(engine_no_provider.quality_score({
            "generated": ["The cat sat on the mat."],
            "references": ["A feline rested upon the rug."],
        }))
        assert result["aggregate"]["n_samples"] == 1
        per = result["per_sample"][0]
        assert "jaccard" in per
        assert "edit_distance_ratio" in per

    def test_quality_score_rouge(self, engine_no_provider):
        result = json.loads(engine_no_provider.quality_score({
            "generated": ["the cat sat on the mat"],
            "references": ["the cat sat on the mat"],
            "metrics": ["rouge"],
        }))
        per = result["per_sample"][0]
        assert per["rouge_1_recall"] == 1.0

    def test_quality_score_bleu(self, engine_no_provider):
        result = json.loads(engine_no_provider.quality_score({
            "generated": ["the cat sat on the mat"],
            "references": ["the cat sat on the mat"],
            "metrics": ["bleu"],
        }))
        assert result["per_sample"][0]["bleu"] >= 0.9

    def test_quality_score_no_generated(self, engine_no_provider):
        result = json.loads(engine_no_provider.quality_score({
            "references": ["ref text"],
        }))
        assert "error" in result

    def test_quality_score_no_references(self, engine_no_provider):
        result = json.loads(engine_no_provider.quality_score({
            "generated": ["gen text"],
        }))
        assert "error" in result

    def test_quality_score_aggregate(self, engine_no_provider):
        result = json.loads(engine_no_provider.quality_score({
            "generated": ["hello world", "foo bar baz"],
            "references": ["hello world", "foo bar baz"],
        }))
        agg = result["aggregate"]
        assert "avg_jaccard" in agg
        assert "avg_edit_distance_ratio" in agg


# ===========================================================================
# prompt_generate
# ===========================================================================

class TestPromptGenerate:

    def test_generate_rephrase_no_provider(self, engine_no_provider):
        result = json.loads(engine_no_provider.prompt_generate({
            "base_prompt": "What is the meaning of life?",
            "n_variants": 3,
            "variation_type": "rephrase",
        }))
        assert result["method"] == "heuristic"
        assert len(result["variants"]) == 3

    def test_generate_constraints(self, engine_no_provider):
        result = json.loads(engine_no_provider.prompt_generate({
            "base_prompt": "Explain quantum computing.",
            "n_variants": 3,
            "variation_type": "add_constraints",
        }))
        assert len(result["variants"]) == 3
        for v in result["variants"]:
            assert "Explain quantum computing." in v

    def test_generate_role(self, engine_no_provider):
        result = json.loads(engine_no_provider.prompt_generate({
            "base_prompt": "Write about climate change.",
            "n_variants": 2,
            "variation_type": "change_role",
        }))
        assert len(result["variants"]) == 2
        for v in result["variants"]:
            assert "expert" in v.lower() or "professor" in v.lower() or "critical" in v.lower() or "consultant" in v.lower()

    def test_generate_with_provider(self, engine_with_provider):
        result = json.loads(engine_with_provider.prompt_generate({
            "base_prompt": "Explain neural networks.",
            "n_variants": 2,
        }))
        # Either LLM or heuristic
        assert "variants" in result
        assert len(result["variants"]) >= 1

    def test_generate_no_base_prompt(self, engine_no_provider):
        result = json.loads(engine_no_provider.prompt_generate({}))
        assert "error" in result

    def test_generate_unknown_type(self, engine_no_provider):
        result = json.loads(engine_no_provider.prompt_generate({
            "base_prompt": "Test prompt",
            "variation_type": "unknown_type",
        }))
        assert "error" in result
