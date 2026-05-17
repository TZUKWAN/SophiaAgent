"""Computational social science: topic modeling, network analysis, ABM, text classification.

Pure-computation engine using numpy, scikit-learn (optional), and networkx
(optional).  All public methods accept ``args: dict`` and return ``str``
(JSON).
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sophia.research.seed import GlobalSeed

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.naive_bayes import MultinomialNB
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(result: dict) -> str:
    """Serialize *result* to JSON, converting non-serializable types."""
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return _convert(obj.tolist())
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj
    return json.dumps(_convert(result), ensure_ascii=False)


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity from a (n_docs x n_features) matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    return normalized @ normalized.T


# ===========================================================================
# ComputationalEngine
# ===========================================================================

class ComputationalEngine:
    """Computational social science engine.

    Every public method accepts ``args: dict`` (tool-dispatch payload)
    and returns ``str`` (JSON).
    """

    # -----------------------------------------------------------------------
    # topic_model
    # -----------------------------------------------------------------------

    def topic_model(self, args: dict) -> str:
        """Topic modeling (LDA / NMF).

        Args:
            texts: list of str
            n_topics: int (default 5)
            method: str ('lda'|'nmf')
            max_features: int (default 1000)
            n_top_words: int (default 10)

        Returns topics with top words, document-topic matrix, perplexity.
        """
        texts = args.get("texts", [])
        n_topics = int(args.get("n_topics", 5))
        method = args.get("method", "lda").lower().strip()
        max_features = int(args.get("max_features", 1000))
        n_top_words = int(args.get("n_top_words", 10))

        if not texts:
            return _json({"error": "texts is required."})
        if len(texts) < 2:
            return _json({"error": "At least 2 texts are required for topic modeling."})
        if n_topics < 1:
            return _json({"error": "n_topics must be >= 1."})

        if not HAS_SKLEARN:
            return _json({"error": "scikit-learn is required for topic modeling."})

        if method == "lda":
            vectorizer = CountVectorizer(
                max_features=max_features, stop_words="english", token_pattern=r"\b[a-zA-Z]{3,}\b"
            )
            try:
                dt_matrix = vectorizer.fit_transform(texts)
            except ValueError:
                return _json({"error": "Vocabulary is empty after preprocessing. Provide longer texts."})

            feature_names = vectorizer.get_feature_names_out()
            model = LatentDirichletAllocation(
                n_components=n_topics, random_state=GlobalSeed.get_or_default(42), max_iter=20
            )
            doc_topic = model.fit_transform(dt_matrix)
            perplexity = float(model.perplexity(dt_matrix))

        elif method == "nmf":
            vectorizer = TfidfVectorizer(
                max_features=max_features, stop_words="english", token_pattern=r"\b[a-zA-Z]{3,}\b"
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
            except ValueError:
                return _json({"error": "Vocabulary is empty after preprocessing. Provide longer texts."})

            feature_names = vectorizer.get_feature_names_out()
            model = NMF(
                n_components=n_topics, random_state=GlobalSeed.get_or_default(42), max_iter=300, init="nndsvd"
            )
            doc_topic = model.fit_transform(tfidf_matrix)
            perplexity = None  # NMF does not have perplexity
        else:
            return _json({"error": f"Unknown method '{method}'. Use 'lda' or 'nmf'."})

        # Extract top words per topic
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append({
                "topic_id": topic_idx,
                "top_words": top_words,
                "weights": [float(topic[i]) for i in top_indices],
            })

        # Document-topic distribution (normalize rows)
        doc_topic_norm = doc_topic / (doc_topic.sum(axis=1, keepdims=True) + 1e-10)

        # Dominant topic per document
        dominant_topics = doc_topic_norm.argmax(axis=1).tolist()

        return _json({
            "method": method,
            "n_topics": n_topics,
            "topics": topics,
            "document_topic_matrix": doc_topic_norm.tolist(),
            "dominant_topics": dominant_topics,
            "perplexity": perplexity,
            "n_documents": len(texts),
            "vocabulary_size": len(feature_names),
        })

    # -----------------------------------------------------------------------
    # network_analysis
    # -----------------------------------------------------------------------

    def network_analysis(self, args: dict) -> str:
        """Social network analysis.

        Args:
            edges: list of {source, target, weight}
            directed: bool (default False)
            metrics: list of str (e.g. ['degree', 'betweenness', 'closeness',
                     'eigenvector', 'density', 'communities'])

        Returns network-level stats, node centrality rankings, communities.
        """
        edges = args.get("edges", [])
        directed = bool(args.get("directed", False))
        metrics = args.get("metrics", ["degree", "betweenness", "closeness", "density", "communities"])

        if not edges:
            return _json({"error": "edges is required."})

        if HAS_NETWORKX:
            return self._network_analysis_nx(edges, directed, metrics)
        else:
            return self._network_analysis_manual(edges, directed, metrics)

    def _network_analysis_nx(self, edges, directed, metrics):
        """Network analysis using NetworkX."""
        G = nx.DiGraph() if directed else nx.Graph()
        for e in edges:
            source = str(e.get("source", ""))
            target = str(e.get("target", ""))
            weight = float(e.get("weight", 1.0))
            G.add_edge(source, target, weight=weight)

        result: Dict[str, Any] = {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "directed": directed,
        }

        if "density" in metrics:
            result["density"] = float(nx.density(G))

        centrality: Dict[str, Dict[str, float]] = {}

        if "degree" in metrics:
            deg = dict(G.degree())
            centrality["degree"] = {str(k): float(v) for k, v in deg.items()}
            result["avg_degree"] = float(np.mean(list(deg.values())))

        if "betweenness" in metrics:
            try:
                btwn = nx.betweenness_centrality(G)
                centrality["betweenness"] = {str(k): float(v) for k, v in btwn.items()}
            except Exception:
                centrality["betweenness"] = {}

        if "closeness" in metrics:
            try:
                close = nx.closeness_centrality(G)
                centrality["closeness"] = {str(k): float(v) for k, v in close.items()}
            except Exception:
                centrality["closeness"] = {}

        if "eigenvector" in metrics:
            try:
                eigen = nx.eigenvector_centrality(G, max_iter=500)
                centrality["eigenvector"] = {str(k): float(v) for k, v in eigen.items()}
            except Exception:
                centrality["eigenvector"] = {}

        result["centrality"] = centrality

        # Top nodes by degree
        if centrality.get("degree"):
            sorted_nodes = sorted(
                centrality["degree"].items(), key=lambda x: x[1], reverse=True
            )
            result["top_nodes_by_degree"] = sorted_nodes[:10]

        if "communities" in metrics:
            try:
                if not directed:
                    communities = nx.community.greedy_modularity_communities(G)
                    result["communities"] = [
                        list(comm) for comm in communities
                    ]
                    result["n_communities"] = len(communities)
                    result["modularity"] = float(
                        nx.community.modularity(G, communities)
                    )
                else:
                    result["communities"] = "Community detection not supported for directed graphs in this implementation."
            except Exception as exc:
                result["communities_error"] = str(exc)

        return _json(result)

    def _network_analysis_manual(self, edges, directed, metrics):
        """Network analysis without NetworkX (basic degree centrality)."""
        nodes = set()
        adj: Dict[str, Dict[str, float]] = defaultdict(dict)
        degree_count: Dict[str, int] = defaultdict(int)

        for e in edges:
            source = str(e.get("source", ""))
            target = str(e.get("target", ""))
            weight = float(e.get("weight", 1.0))
            nodes.add(source)
            nodes.add(target)
            adj[source][target] = weight
            degree_count[source] += 1
            if not directed:
                adj[target][source] = weight
                degree_count[target] += 1

        n_nodes = len(nodes)
        n_edges = len(edges)

        result: Dict[str, Any] = {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "directed": directed,
            "note": "Basic analysis without NetworkX. Install networkx for full features.",
        }

        if "density" in metrics:
            max_edges = n_nodes * (n_nodes - 1) if directed else n_nodes * (n_nodes - 1) / 2
            result["density"] = n_edges / max_edges if max_edges > 0 else 0.0

        if "degree" in metrics:
            result["avg_degree"] = sum(degree_count.values()) / n_nodes if n_nodes > 0 else 0.0
            sorted_nodes = sorted(degree_count.items(), key=lambda x: x[1], reverse=True)
            result["top_nodes_by_degree"] = sorted_nodes[:10]

        return _json(result)

    # -----------------------------------------------------------------------
    # abm_simulate
    # -----------------------------------------------------------------------

    def abm_simulate(self, args: dict) -> str:
        """Agent-based modeling simulation.

        Args:
            n_agents: int (default 100)
            steps: int (default 50)
            agent_type: str ('schelling'|'epidemic'|'opinion')
            params: dict (model-specific parameters)
            seed: int

        Returns time_series data, final_state, summary_statistics.
        """
        n_agents = int(args.get("n_agents", 100))
        steps = int(args.get("steps", 50))
        agent_type = args.get("agent_type", "schelling").lower().strip()
        params = args.get("params", {})
        seed = args.get("seed", 42)

        if n_agents < 2:
            return _json({"error": "n_agents must be >= 2."})
        if steps < 1:
            return _json({"error": "steps must be >= 1."})

        if agent_type == "schelling":
            return self._schelling(n_agents, steps, params, seed)
        elif agent_type == "epidemic":
            return self._epidemic(n_agents, steps, params, seed)
        elif agent_type == "opinion":
            return self._opinion(n_agents, steps, params, seed)
        else:
            return _json({"error": f"Unknown agent_type '{agent_type}'. Use 'schelling', 'epidemic', or 'opinion'."})

    def _schelling(self, n_agents, steps, params, seed):
        """Schelling segregation model."""
        rng = random.Random(seed)

        grid_size = int(params.get("grid_size", int(math.ceil(math.sqrt(n_agents * 2)))))
        threshold = float(params.get("threshold", 0.3))
        n_types = int(params.get("n_types", 2))

        # Initialize grid
        grid = [[None] * grid_size for _ in range(grid_size)]
        agents = []
        positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        rng.shuffle(positions)

        for i in range(min(n_agents, len(positions))):
            agent_type = i % n_types
            r, c = positions[i]
            grid[r][c] = agent_type
            agents.append({"type": agent_type, "r": r, "c": c})

        time_series = []

        def _segregation_index():
            """Average proportion of same-type neighbors."""
            total = 0.0
            count = 0
            for a in agents:
                r, c = a["r"], a["c"]
                same = 0
                neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            if grid[nr][nc] is not None:
                                neighbors += 1
                                if grid[nr][nc] == a["type"]:
                                    same += 1
                if neighbors > 0:
                    total += same / neighbors
                    count += 1
            return total / count if count > 0 else 0.0

        time_series.append({"step": 0, "segregation_index": round(_segregation_index(), 4)})

        empty_positions = [
            (r, c) for r in range(grid_size)
            for c in range(grid_size) if grid[r][c] is None
        ]

        for step in range(1, steps + 1):
            moved = 0
            order = list(range(len(agents)))
            rng.shuffle(order)
            for idx in order:
                a = agents[idx]
                r, c = a["r"], a["c"]
                same = 0
                neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            if grid[nr][nc] is not None:
                                neighbors += 1
                                if grid[nr][nc] == a["type"]:
                                    same += 1

                ratio = same / neighbors if neighbors > 0 else 1.0
                if ratio < threshold and empty_positions:
                    # Move to random empty position
                    grid[r][c] = None
                    empty_positions.append((r, c))
                    new_pos = rng.choice(empty_positions)
                    empty_positions.remove(new_pos)
                    grid[new_pos[0]][new_pos[1]] = a["type"]
                    a["r"], a["c"] = new_pos
                    moved += 1

            seg = _segregation_index()
            time_series.append({"step": step, "segregation_index": round(seg, 4), "agents_moved": moved})
            if moved == 0:
                break  # Equilibrium reached

        final_seg = time_series[-1]["segregation_index"]
        return _json({
            "model": "schelling",
            "n_agents": len(agents),
            "grid_size": grid_size,
            "threshold": threshold,
            "n_types": n_types,
            "time_series": time_series,
            "final_segregation_index": final_seg,
            "steps_run": len(time_series) - 1,
            "summary": {
                "initial_segregation": time_series[0]["segregation_index"],
                "final_segregation": final_seg,
                "change": round(final_seg - time_series[0]["segregation_index"], 4),
            },
        })

    def _epidemic(self, n_agents, steps, params, seed):
        """SIR epidemic model."""
        rng = random.Random(seed)

        infection_rate = float(params.get("infection_rate", 0.3))
        recovery_rate = float(params.get("recovery_rate", 0.1))
        initial_infected = int(params.get("initial_infected", max(1, n_agents // 10)))
        contact_rate = int(params.get("contact_rate", 3))

        # States: S=0, I=1, R=2
        states = [0] * n_agents
        for i in range(min(initial_infected, n_agents)):
            states[i] = 1
        rng.shuffle(states)

        # Build random network (each agent has some neighbors)
        neighbors: Dict[int, List[int]] = {i: [] for i in range(n_agents)}
        for i in range(n_agents):
            # Random connections
            possible = [j for j in range(n_agents) if j != i]
            n_contacts = min(contact_rate, len(possible))
            contacts = rng.sample(possible, n_contacts)
            neighbors[i].extend(contacts)
            for c in contacts:
                if i not in neighbors[c]:
                    neighbors[c].append(i)

        time_series = []

        def _counts():
            s = sum(1 for x in states if x == 0)
            i = sum(1 for x in states if x == 1)
            r = sum(1 for x in states if x == 2)
            return s, i, r

        s0, i0, r0 = _counts()
        time_series.append({"step": 0, "susceptible": s0, "infected": i0, "recovered": r0})

        for step in range(1, steps + 1):
            new_states = states.copy()
            for i in range(n_agents):
                if states[i] == 0:  # Susceptible
                    # Check contacts
                    infected_contacts = sum(1 for n in neighbors[i] if states[n] == 1)
                    if infected_contacts > 0:
                        prob = 1.0 - (1.0 - infection_rate) ** infected_contacts
                        if rng.random() < prob:
                            new_states[i] = 1  # Infected
                elif states[i] == 1:  # Infected
                    if rng.random() < recovery_rate:
                        new_states[i] = 2  # Recovered

            states = new_states
            s, i_count, r = _counts()
            time_series.append({"step": step, "susceptible": s, "infected": i_count, "recovered": r})
            if i_count == 0:
                break  # Epidemic over

        final = time_series[-1]
        return _json({
            "model": "epidemic",
            "n_agents": n_agents,
            "infection_rate": infection_rate,
            "recovery_rate": recovery_rate,
            "initial_infected": initial_infected,
            "time_series": time_series,
            "final_state": {"susceptible": final["susceptible"], "infected": final["infected"], "recovered": final["recovered"]},
            "summary": {
                "peak_infected": max(t["infected"] for t in time_series),
                "peak_step": max(range(len(time_series)), key=lambda i: time_series[i]["infected"]),
                "total_ever_infected": final["recovered"] + final["infected"],
                "attack_rate": round((final["recovered"] + final["infected"]) / n_agents, 4),
                "steps_run": len(time_series) - 1,
            },
        })

    def _opinion(self, n_agents, steps, params, seed):
        """Opinion dynamics model (bounded confidence / averaging)."""
        rng = random.Random(seed)

        confidence_threshold = float(params.get("confidence_threshold", 0.3))
        convergence_threshold = float(params.get("convergence_threshold", 0.01))
        n_neighbors = int(params.get("n_neighbors", 5))

        # Initialize opinions uniformly in [0, 1]
        opinions = [rng.random() for _ in range(n_agents)]

        # Build random neighbor lists
        neighbor_list: Dict[int, List[int]] = {}
        for i in range(n_agents):
            possible = [j for j in range(n_agents) if j != i]
            n_nbrs = min(n_neighbors, len(possible))
            neighbor_list[i] = rng.sample(possible, n_nbrs)

        time_series = []
        opinion_std = float(np.std(opinions))
        opinion_mean = float(np.mean(opinions))
        time_series.append({
            "step": 0,
            "mean_opinion": round(opinion_mean, 4),
            "std_opinion": round(opinion_std, 4),
            "range": round(max(opinions) - min(opinions), 4),
        })

        for step in range(1, steps + 1):
            new_opinions = opinions.copy()
            order = list(range(n_agents))
            rng.shuffle(order)

            for i in order:
                # Average with neighbors within confidence threshold
                neighbor_vals = []
                for j in neighbor_list[i]:
                    if abs(opinions[i] - opinions[j]) < confidence_threshold:
                        neighbor_vals.append(opinions[j])

                if neighbor_vals:
                    new_opinions[i] = (opinions[i] + sum(neighbor_vals)) / (1 + len(neighbor_vals))

            opinions = new_opinions
            op_std = float(np.std(opinions))
            op_mean = float(np.mean(opinions))
            op_range = max(opinions) - min(opinions)
            time_series.append({
                "step": step,
                "mean_opinion": round(op_mean, 4),
                "std_opinion": round(op_std, 4),
                "range": round(op_range, 4),
            })

            if op_std < convergence_threshold:
                break  # Converged

        final = time_series[-1]
        # Cluster opinions
        op_array = np.array(opinions)
        unique_clusters = 1
        if len(op_array) > 1:
            sorted_ops = sorted(opinions)
            for i in range(1, len(sorted_ops)):
                if sorted_ops[i] - sorted_ops[i - 1] > confidence_threshold:
                    unique_clusters += 1

        return _json({
            "model": "opinion",
            "n_agents": n_agents,
            "confidence_threshold": confidence_threshold,
            "time_series": time_series,
            "final_opinions": [round(o, 4) for o in opinions],
            "summary": {
                "initial_std": time_series[0]["std_opinion"],
                "final_std": final["std_opinion"],
                "convergence_achieved": final["std_opinion"] < convergence_threshold,
                "opinion_clusters": unique_clusters,
                "steps_run": len(time_series) - 1,
            },
        })

    # -----------------------------------------------------------------------
    # text_classify
    # -----------------------------------------------------------------------

    def text_classify(self, args: dict) -> str:
        """Text classification.

        Args:
            texts: list of str
            labels: list of str
            method: str ('tfidf_lr'|'count_nb')
            test_size: float (default 0.2)

        Returns accuracy, classification_report, confusion_matrix.
        """
        texts = args.get("texts", [])
        labels = args.get("labels", [])
        method = args.get("method", "tfidf_lr").lower().strip()
        test_size = float(args.get("test_size", 0.2))

        if not texts or not labels:
            return _json({"error": "texts and labels are required."})
        if len(texts) != len(labels):
            return _json({"error": f"texts ({len(texts)}) and labels ({len(labels)}) must have same length."})
        if len(set(labels)) < 2:
            return _json({"error": "At least 2 unique labels are required for classification."})

        if HAS_SKLEARN:
            return self._text_classify_sklearn(texts, labels, method, test_size)
        else:
            return self._text_classify_manual(texts, labels, test_size)

    def _text_classify_sklearn(self, texts, labels, method, test_size):
        """Classification using scikit-learn."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=GlobalSeed.get_or_default(42), stratify=labels
            )
        except ValueError:
            # Fallback for very small datasets where stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=GlobalSeed.get_or_default(42)
            )

        if method == "tfidf_lr":
            vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"\b[a-zA-Z]{2,}\b")
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            clf = LogisticRegression(max_iter=1000, random_state=GlobalSeed.get_or_default(42))
        elif method == "count_nb":
            vectorizer = CountVectorizer(max_features=5000, token_pattern=r"\b[a-zA-Z]{2,}\b")
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            clf = MultinomialNB()
        else:
            return _json({"error": f"Unknown method '{method}'. Use 'tfidf_lr' or 'count_nb'."})

        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Confusion matrix
        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        cm = [[0] * n_classes for _ in range(n_classes)]
        for true_l, pred_l in zip(y_test, y_pred):
            cm[label_to_idx[true_l]][label_to_idx[pred_l]] += 1

        # Per-class metrics
        per_class = {}
        for label_name, metrics in report.items():
            if isinstance(metrics, dict):
                per_class[label_name] = {
                    "precision": round(metrics.get("precision", 0), 4),
                    "recall": round(metrics.get("recall", 0), 4),
                    "f1_score": round(metrics.get("f1-score", 0), 4),
                    "support": metrics.get("support", 0),
                }

        return _json({
            "method": method,
            "accuracy": round(acc, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": n_classes,
            "labels": unique_labels,
            "confusion_matrix": cm,
            "per_class_metrics": per_class,
            "weighted_avg": {
                "precision": round(report.get("weighted avg", {}).get("precision", 0), 4),
                "recall": round(report.get("weighted avg", {}).get("recall", 0), 4),
                "f1_score": round(report.get("weighted avg", {}).get("f1-score", 0), 4),
            },
        })

    def _text_classify_manual(self, texts, labels, test_size):
        """Simple manual classification using keyword matching."""
        n = len(texts)
        n_test = max(1, int(n * test_size))

        # Simple split
        indices = list(range(n))
        random.Random(42).shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Build keyword profiles per label from training data
        label_keywords: Dict[str, Counter] = defaultdict(Counter)
        for idx in train_indices:
            label = labels[idx]
            words = texts[idx].lower().split()
            label_keywords[label].update(words)

        # Classify test data
        unique_labels = sorted(set(labels))
        n_classes = len(unique_labels)
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        cm = [[0] * n_classes for _ in range(n_classes)]
        correct = 0

        for idx in test_indices:
            words = texts[idx].lower().split()
            true_label = labels[idx]
            best_label = unique_labels[0]
            best_score = -1

            for label in unique_labels:
                score = sum(label_keywords[label].get(w, 0) for w in words)
                if score > best_score:
                    best_score = score
                    best_label = label

            if best_label == true_label:
                correct += 1
            cm[label_to_idx[true_label]][label_to_idx[best_label]] += 1

        acc = correct / len(test_indices) if test_indices else 0.0

        return _json({
            "method": "manual_keyword_matching (sklearn not available)",
            "accuracy": round(acc, 4),
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "n_classes": n_classes,
            "labels": unique_labels,
            "confusion_matrix": cm,
            "note": "Install scikit-learn for proper ML classification.",
        })

    # -----------------------------------------------------------------------
    # embedding_analysis
    # -----------------------------------------------------------------------

    def embedding_analysis(self, args: dict) -> str:
        """Simple embedding/semantic analysis.

        Args:
            texts: list of str
            method: str ('tfidf')
            n_clusters: int (optional, default 3)
            similarity_threshold: float (optional, default 0.7)

        Returns document similarity matrix, clusters, nearest_neighbors.
        """
        texts = args.get("texts", [])
        method = args.get("method", "tfidf").lower().strip()
        n_clusters = int(args.get("n_clusters", 3))
        similarity_threshold = float(args.get("similarity_threshold", 0.7))

        if not texts:
            return _json({"error": "texts is required."})
        if len(texts) < 2:
            return _json({"error": "At least 2 texts are required."})

        # Generate document vectors
        if HAS_SKLEARN:
            vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english",
                token_pattern=r"\b[a-zA-Z]{2,}\b"
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                vectors = tfidf_matrix.toarray()
            except ValueError:
                return _json({"error": "Vocabulary is empty. Provide longer texts."})
        else:
            # Manual bag-of-words vectors
            vocab = set()
            tokenized = []
            for t in texts:
                tokens = t.lower().split()
                tokenized.append(tokens)
                vocab.update(tokens)

            vocab_list = sorted(vocab)
            vocab_idx = {w: i for i, w in enumerate(vocab_list)}
            vectors = np.zeros((len(texts), len(vocab_list)))
            for i, tokens in enumerate(tokenized):
                for t in tokens:
                    if t in vocab_idx:
                        vectors[i, vocab_idx[t]] += 1

        # Cosine similarity matrix
        sim_matrix = _cosine_similarity_matrix(vectors)

        # Clustering
        clusters: Dict[int, List[int]] = {}
        if HAS_SKLEARN and n_clusters <= len(texts):
            actual_k = min(n_clusters, len(texts))
            kmeans = KMeans(n_clusters=actual_k, random_state=GlobalSeed.get_or_default(42), n_init=10)
            cluster_labels = kmeans.fit_predict(vectors)
            for i, cl in enumerate(cluster_labels):
                cl_int = int(cl)
                if cl_int not in clusters:
                    clusters[cl_int] = []
                clusters[cl_int].append(i)
        else:
            # Simple manual clustering by similarity
            cluster_id = 0
            assigned = set()
            for i in range(len(texts)):
                if i in assigned:
                    continue
                cluster_members = [i]
                assigned.add(i)
                for j in range(i + 1, len(texts)):
                    if j not in assigned and sim_matrix[i][j] >= similarity_threshold:
                        cluster_members.append(j)
                        assigned.add(j)
                clusters[cluster_id] = cluster_members
                cluster_id += 1

        # Nearest neighbors for each document
        nearest_neighbors = []
        for i in range(len(texts)):
            sims = [(j, float(sim_matrix[i][j])) for j in range(len(texts)) if j != i]
            sims.sort(key=lambda x: x[1], reverse=True)
            nearest_neighbors.append({
                "doc_index": i,
                "neighbors": [{"index": j, "similarity": round(s, 4)} for j, s in sims[:5]],
            })

        # Convert similarity matrix to list of lists
        sim_list = sim_matrix.tolist()
        # Round for readability
        sim_list = [[round(v, 4) for v in row] for row in sim_list]

        return _json({
            "method": method,
            "n_documents": len(texts),
            "vectors_shape": list(vectors.shape),
            "similarity_matrix": sim_list,
            "clusters": {str(k): v for k, v in clusters.items()},
            "n_clusters": len(clusters),
            "nearest_neighbors": nearest_neighbors,
            "similarity_threshold": similarity_threshold,
        })
