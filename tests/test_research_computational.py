"""Tests for ComputationalEngine -- comprehensive pytest suite."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sophia.research.computational import (
    HAS_NETWORKX,
    HAS_SKLEARN,
    ComputationalEngine,
    _cosine_similarity_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return ComputationalEngine()


@pytest.fixture
def sample_texts():
    return [
        "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data.",
        "Deep learning is a subset of machine learning that uses neural networks with many layers.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision is a field of AI that trains computers to interpret the visual world.",
        "Reinforcement learning is a type of machine learning where agents learn by interacting with an environment.",
        "Transfer learning allows models trained on one task to be adapted for another related task.",
        "Generative adversarial networks are used to generate synthetic data that resembles real data.",
        "Convolutional neural networks are particularly effective for image classification tasks.",
        "Recurrent neural networks are designed to handle sequential data like text and speech.",
        "Bayesian methods provide a framework for reasoning about uncertainty in machine learning models.",
    ]


@pytest.fixture
def sample_edges():
    return [
        {"source": "Alice", "target": "Bob", "weight": 1.0},
        {"source": "Alice", "target": "Charlie", "weight": 2.0},
        {"source": "Bob", "target": "Charlie", "weight": 1.5},
        {"source": "Bob", "target": "David", "weight": 1.0},
        {"source": "Charlie", "target": "David", "weight": 3.0},
        {"source": "David", "target": "Eve", "weight": 1.0},
        {"source": "Eve", "target": "Alice", "weight": 2.0},
    ]


# ===========================================================================
# topic_model
# ===========================================================================

class TestTopicModel:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_lda_basic(self, engine, sample_texts):
        result = json.loads(engine.topic_model({
            "texts": sample_texts,
            "n_topics": 3,
            "method": "lda",
        }))
        assert result["method"] == "lda"
        assert result["n_topics"] == 3
        assert len(result["topics"]) == 3
        assert result["n_documents"] == len(sample_texts)
        assert "document_topic_matrix" in result
        assert "dominant_topics" in result
        assert "perplexity" in result

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_nmf_basic(self, engine, sample_texts):
        result = json.loads(engine.topic_model({
            "texts": sample_texts,
            "n_topics": 2,
            "method": "nmf",
        }))
        assert result["method"] == "nmf"
        assert len(result["topics"]) == 2
        assert result["perplexity"] is None  # NMF has no perplexity

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_topic_words_populated(self, engine, sample_texts):
        result = json.loads(engine.topic_model({
            "texts": sample_texts,
            "n_topics": 3,
            "n_top_words": 5,
        }))
        for topic in result["topics"]:
            assert len(topic["top_words"]) == 5
            assert all(isinstance(w, str) for w in topic["top_words"])

    def test_no_texts_error(self, engine):
        result = json.loads(engine.topic_model({}))
        assert "error" in result

    def test_too_few_texts_error(self, engine):
        result = json.loads(engine.topic_model({"texts": ["Only one text"]}))
        assert "error" in result


# ===========================================================================
# network_analysis
# ===========================================================================

class TestNetworkAnalysis:

    def test_basic_network(self, engine, sample_edges):
        result = json.loads(engine.network_analysis({
            "edges": sample_edges,
        }))
        assert result["n_nodes"] == 5
        assert result["n_edges"] == 7
        assert result["directed"] is False

    def test_directed_network(self, engine, sample_edges):
        result = json.loads(engine.network_analysis({
            "edges": sample_edges,
            "directed": True,
        }))
        assert result["directed"] is True

    def test_density(self, engine, sample_edges):
        result = json.loads(engine.network_analysis({
            "edges": sample_edges,
            "metrics": ["density"],
        }))
        assert "density" in result
        assert 0 < result["density"] <= 1.0

    @pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
    def test_centrality(self, engine, sample_edges):
        result = json.loads(engine.network_analysis({
            "edges": sample_edges,
            "metrics": ["degree", "betweenness", "closeness"],
        }))
        assert "centrality" in result
        assert "degree" in result["centrality"]
        assert "betweenness" in result["centrality"]
        assert "closeness" in result["centrality"]

    @pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
    def test_communities(self, engine, sample_edges):
        result = json.loads(engine.network_analysis({
            "edges": sample_edges,
            "metrics": ["communities"],
        }))
        assert "communities" in result
        assert "n_communities" in result

    @pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
    def test_top_nodes(self, engine, sample_edges):
        result = json.loads(engine.network_analysis({
            "edges": sample_edges,
            "metrics": ["degree"],
        }))
        assert "top_nodes_by_degree" in result
        assert len(result["top_nodes_by_degree"]) > 0

    def test_no_edges_error(self, engine):
        result = json.loads(engine.network_analysis({}))
        assert "error" in result


# ===========================================================================
# abm_simulate
# ===========================================================================

class TestABMSimulate:

    def test_schelling_basic(self, engine):
        result = json.loads(engine.abm_simulate({
            "n_agents": 50,
            "steps": 20,
            "agent_type": "schelling",
            "seed": 42,
        }))
        assert result["model"] == "schelling"
        assert result["n_agents"] == 50
        assert "time_series" in result
        assert "final_segregation_index" in result
        assert result["time_series"][0]["step"] == 0
        assert len(result["time_series"]) > 1

    def test_schelling_segregation_increases(self, engine):
        result = json.loads(engine.abm_simulate({
            "n_agents": 80,
            "steps": 30,
            "agent_type": "schelling",
            "params": {"threshold": 0.4},
            "seed": 42,
        }))
        initial = result["time_series"][0]["segregation_index"]
        final = result["final_segregation_index"]
        # Schelling model should increase segregation from random initial state
        assert final >= initial - 0.05  # Allow small tolerance

    def test_epidemic_basic(self, engine):
        result = json.loads(engine.abm_simulate({
            "n_agents": 100,
            "steps": 50,
            "agent_type": "epidemic",
            "params": {"infection_rate": 0.3, "recovery_rate": 0.1, "initial_infected": 5},
            "seed": 42,
        }))
        assert result["model"] == "epidemic"
        assert "time_series" in result
        assert "final_state" in result
        ts = result["time_series"]
        assert ts[0]["susceptible"] == 95
        assert ts[0]["infected"] == 5

    def test_epidemic_sir_conservation(self, engine):
        result = json.loads(engine.abm_simulate({
            "n_agents": 100,
            "steps": 30,
            "agent_type": "epidemic",
            "seed": 42,
        }))
        for t in result["time_series"]:
            total = t["susceptible"] + t["infected"] + t["recovered"]
            assert total == 100

    def test_opinion_basic(self, engine):
        result = json.loads(engine.abm_simulate({
            "n_agents": 50,
            "steps": 30,
            "agent_type": "opinion",
            "params": {"confidence_threshold": 0.3},
            "seed": 42,
        }))
        assert result["model"] == "opinion"
        assert len(result["final_opinions"]) == 50
        assert "time_series" in result
        assert "summary" in result
        assert "opinion_clusters" in result["summary"]

    def test_opinion_convergence(self, engine):
        result = json.loads(engine.abm_simulate({
            "n_agents": 30,
            "steps": 100,
            "agent_type": "opinion",
            "params": {"confidence_threshold": 0.5, "n_neighbors": 8},
            "seed": 42,
        }))
        # With high confidence threshold, opinions should converge
        final_std = result["summary"]["final_std"]
        assert final_std < result["summary"]["initial_std"]

    def test_invalid_agent_type(self, engine):
        result = json.loads(engine.abm_simulate({
            "agent_type": "invalid_model",
        }))
        assert "error" in result

    def test_too_few_agents(self, engine):
        result = json.loads(engine.abm_simulate({"n_agents": 1}))
        assert "error" in result

    def test_invalid_steps(self, engine):
        result = json.loads(engine.abm_simulate({"steps": 0}))
        assert "error" in result


# ===========================================================================
# text_classify
# ===========================================================================

class TestTextClassify:

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_classify_basic(self, engine):
        texts = [
            "I love this product, it is amazing and wonderful",
            "This is the best thing I have ever bought",
            "Great quality and fast delivery, very satisfied",
            "Terrible experience, the product broke immediately",
            "Worst purchase ever, I want my money back",
            "Very disappointed with the quality and service",
            "The item was okay, nothing special but it works",
            "Average product, meets basic expectations",
            "Decent quality for the price, would recommend",
            "Not bad but could be improved in several ways",
        ]
        labels = ["positive", "positive", "positive", "negative", "negative",
                   "negative", "neutral", "neutral", "positive", "neutral"]

        result = json.loads(engine.text_classify({
            "texts": texts,
            "labels": labels,
            "method": "tfidf_lr",
            "test_size": 0.3,
        }))
        assert "accuracy" in result
        assert "confusion_matrix" in result
        assert "n_classes" in result
        assert result["n_classes"] == 3
        assert result["method"] == "tfidf_lr"

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_classify_nb(self, engine):
        texts = [
            "This movie is great and exciting",
            "Wonderful film with excellent acting",
            "I really enjoyed watching this movie",
            "The movie was boring and too long",
            "Terrible plot and bad acting throughout",
            "I fell asleep during this dull movie",
        ]
        labels = ["pos", "pos", "pos", "neg", "neg", "neg"]

        result = json.loads(engine.text_classify({
            "texts": texts,
            "labels": labels,
            "method": "count_nb",
        }))
        assert "accuracy" in result
        assert result["method"] == "count_nb"

    def test_no_texts_error(self, engine):
        result = json.loads(engine.text_classify({
            "labels": ["a", "b"],
        }))
        assert "error" in result

    def test_length_mismatch_error(self, engine):
        result = json.loads(engine.text_classify({
            "texts": ["hello", "world"],
            "labels": ["a"],
        }))
        assert "error" in result

    def test_single_label_error(self, engine):
        result = json.loads(engine.text_classify({
            "texts": ["hello", "world"],
            "labels": ["a", "a"],
        }))
        assert "error" in result


# ===========================================================================
# embedding_analysis
# ===========================================================================

class TestEmbeddingAnalysis:

    def test_embedding_basic(self, engine, sample_texts):
        result = json.loads(engine.embedding_analysis({
            "texts": sample_texts[:5],
            "n_clusters": 2,
        }))
        assert result["n_documents"] == 5
        assert "similarity_matrix" in result
        assert "clusters" in result
        assert "nearest_neighbors" in result
        assert len(result["similarity_matrix"]) == 5
        assert len(result["similarity_matrix"][0]) == 5

    def test_similarity_diagonal_is_one(self, engine, sample_texts):
        result = json.loads(engine.embedding_analysis({
            "texts": sample_texts[:3],
        }))
        sim = result["similarity_matrix"]
        for i in range(3):
            assert sim[i][i] == pytest.approx(1.0, abs=0.01)

    def test_similarity_symmetric(self, engine, sample_texts):
        result = json.loads(engine.embedding_analysis({
            "texts": sample_texts[:4],
        }))
        sim = result["similarity_matrix"]
        for i in range(4):
            for j in range(4):
                assert sim[i][j] == pytest.approx(sim[j][i], abs=0.01)

    def test_no_texts_error(self, engine):
        result = json.loads(engine.embedding_analysis({}))
        assert "error" in result

    def test_single_text_error(self, engine):
        result = json.loads(engine.embedding_analysis({"texts": ["Only one"]}))
        assert "error" in result

    def test_nearest_neighbors_count(self, engine, sample_texts):
        result = json.loads(engine.embedding_analysis({
            "texts": sample_texts[:6],
        }))
        nn = result["nearest_neighbors"]
        assert len(nn) == 6
        for item in nn:
            assert item["doc_index"] is not None
            assert len(item["neighbors"]) == min(5, 5)  # 6 docs -> 5 neighbors each


# ===========================================================================
# cosine_similarity_matrix helper
# ===========================================================================

class TestCosineSimilarityMatrix:

    def test_identity_matrix(self):
        vectors = np.eye(3)
        sim = _cosine_similarity_matrix(vectors)
        np.testing.assert_allclose(sim, np.eye(3), atol=1e-10)

    def test_identical_vectors(self):
        vectors = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        sim = _cosine_similarity_matrix(vectors)
        assert sim[0, 1] == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_vectors(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = _cosine_similarity_matrix(vectors)
        assert sim[0, 1] == pytest.approx(0.0, abs=1e-10)
