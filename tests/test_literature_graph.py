"""Tests for LiteratureGraph."""

import os

import pytest

from sophia.research.literature_graph import LiteratureGraph, RELATION_TYPES


@pytest.fixture
def graph_mgr(tmp_workspace):
    return LiteratureGraph(tmp_workspace)


@pytest.fixture
def sample_bib(tmp_workspace):
    bib_path = os.path.join(tmp_workspace, ".sophia", "references.bib")
    os.makedirs(os.path.dirname(bib_path), exist_ok=True)
    bib_content = """
@article{smith2024,
  author  = {Smith, John and Doe, Jane},
  title   = {Social Capital and Happiness},
  journal = {Social Science Quarterly},
  year    = {2024},
  volume  = {45},
  pages   = {123--145},
}

@article{zhang2023,
  author  = {Zhang, Wei},
  title   = {Trust and Well-being in China},
  journal = {Chinese Sociology},
  year    = {2023},
  volume  = {12},
  pages   = {67--89},
}

@book{brown2022,
  author  = {Brown, Alice},
  title   = {The Sociology of Networks},
  year    = {2022},
  publisher = {Academic Press},
}
"""
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(bib_content)
    return bib_path


@pytest.fixture
def sample_relations(tmp_workspace):
    relations_path = os.path.join(tmp_workspace, ".sophia", "citation_relations.json")
    relations = [
        {"from": "smith2024", "to": "zhang2023", "type": "cites"},
        {"from": "zhang2023", "to": "brown2022", "type": "extends"},
        {"from": "smith2024", "to": "brown2022", "type": "theory_similar"},
    ]
    with open(relations_path, "w", encoding="utf-8") as f:
        import json
        json.dump(relations, f, ensure_ascii=False, indent=2)
    return relations_path


class TestBuildGraph:
    def test_build_empty_graph(self, graph_mgr):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        assert G.number_of_nodes() == 0

    def test_build_with_data(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3
        assert G.has_edge("smith2024", "zhang2023")
        assert G.edges["smith2024", "zhang2023"]["relation"] == "cites"

    def test_build_filtered(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph(literature_ids=["smith2024", "zhang2023"])
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1

    def test_node_attributes(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        assert G.nodes["smith2024"]["title"] == "Social Capital and Happiness"
        assert G.nodes["zhang2023"]["year"] == "2023"


class TestVisualize:
    def test_mermaid_output(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        mermaid = graph_mgr.visualize(G, format="mermaid")
        assert "graph TD" in mermaid
        assert "smith2024" in mermaid
        assert "zhang2023" in mermaid

    def test_tikz_output(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        tikz = graph_mgr.visualize(G, format="tikz")
        assert "\\begin{tikzpicture}" in tikz
        assert "smith2024" in tikz

    def test_dot_output(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        dot = graph_mgr.visualize(G, format="dot")
        assert "digraph LiteratureGraph" in dot
        assert "smith2024" in dot

    def test_empty_graph_visualization(self, graph_mgr):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        mermaid = graph_mgr.visualize(G, format="mermaid")
        assert "Empty graph" in mermaid

    def test_unknown_format(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        result = graph_mgr.visualize(G, format="svg")
        assert "Unknown format" in result


class TestDetectClusters:
    def test_clusters(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        clusters = graph_mgr.detect_clusters(G)
        assert isinstance(clusters, list)
        assert len(clusters) > 0

    def test_empty_graph_clusters(self, graph_mgr):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        clusters = graph_mgr.detect_clusters(G)
        assert clusters == []


class TestFindKeyPapers:
    def test_key_papers(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        papers = graph_mgr.find_key_papers(G)
        assert isinstance(papers, list)
        assert len(papers) == 3
        # All should have centrality metrics
        for p in papers:
            assert "pagerank" in p
            assert "in_degree" in p
            assert "betweenness" in p

    def test_empty_graph_key_papers(self, graph_mgr):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        papers = graph_mgr.find_key_papers(G)
        assert papers == []


class TestGraphStats:
    def test_stats(self, graph_mgr, sample_bib, sample_relations):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        stats = graph_mgr.graph_stats(G)
        assert stats["nodes"] == 3
        assert stats["edges"] == 3
        assert "density" in stats

    def test_empty_stats(self, graph_mgr):
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("networkx not installed")
        G = graph_mgr.build_graph()
        stats = graph_mgr.graph_stats(G)
        assert stats["nodes"] == 0
        assert stats["edges"] == 0


class TestRelationTypes:
    def test_relation_types_defined(self):
        assert "cites" in RELATION_TYPES
        assert "contradicts" in RELATION_TYPES
        assert "extends" in RELATION_TYPES
        assert "supplements" in RELATION_TYPES
        assert "method_similar" in RELATION_TYPES
        assert "theory_similar" in RELATION_TYPES
