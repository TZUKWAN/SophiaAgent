"""LiteratureGraph: Build, analyze, and visualize citation networks.

Reads from references.bib and citation_relations.json in workspace/.sophia/.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RELATION_TYPES = [
    "cites",
    "contradicts",
    "extends",
    "supplements",
    "method_similar",
    "theory_similar",
]


class LiteratureGraph:
    """Citation network builder and analyzer."""

    def __init__(self, workspace: str):
        self.workspace = workspace
        self.bib_path = os.path.join(workspace, ".sophia", "references.bib")
        self.relations_path = os.path.join(workspace, ".sophia", "citation_relations.json")

    # -- data loading -------------------------------------------------------

    def _load_bib_entries(self) -> List[Dict[str, Any]]:
        """Load BibTeX entries from references.bib."""
        if not os.path.exists(self.bib_path):
            return []
        try:
            import bibtexparser
            from bibtexparser.bparser import BibTexParser

            with open(self.bib_path, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                return []
            parser = BibTexParser(common_strings=True)
            parser.ignore_nonstandard_types = False
            bib_db = bibtexparser.loads(content, parser=parser)
            entries = []
            for entry in bib_db.entries:
                fields = dict(entry)
                fields["_key"] = fields.pop("ID", "")
                fields["_type"] = fields.pop("ENTRYTYPE", "article")
                entries.append(fields)
            return entries
        except Exception as e:
            logger.warning("Failed to parse BibTeX: %s", e)
            return []

    def _load_relations(self) -> List[Dict[str, Any]]:
        """Load citation relations from citation_relations.json."""
        if not os.path.exists(self.relations_path):
            return []
        try:
            with open(self.relations_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load relations: %s", e)
            return []

    # -- graph building -----------------------------------------------------

    def build_graph(
        self,
        literature_ids: Optional[List[str]] = None,
    ) -> Any:
        """Build a NetworkX DiGraph from references and relations.

        Args:
            literature_ids: Optional list of BibTeX keys to filter by.
                            If None, includes all entries.

        Returns:
            networkx.DiGraph
        """
        try:
            import networkx as nx
        except ImportError:
            raise RuntimeError(
                "networkx is required for LiteratureGraph. "
                "Install with: pip install networkx"
            )

        entries = self._load_bib_entries()
        relations = self._load_relations()

        # Filter entries if literature_ids provided
        if literature_ids:
            entry_map = {
                e.get("_key", ""): e
                for e in entries
                if e.get("_key", "") in literature_ids
            }
        else:
            entry_map = {e.get("_key", ""): e for e in entries}

        G = nx.DiGraph()

        # Add nodes
        for key, entry in entry_map.items():
            G.add_node(
                key,
                title=entry.get("title", ""),
                author=entry.get("author", ""),
                year=entry.get("year", ""),
                journal=entry.get("journal", entry.get("booktitle", "")),
                entry_type=entry.get("_type", "article"),
            )

        # Add edges from relations
        for rel in relations:
            src = rel.get("from", "")
            tgt = rel.get("to", "")
            rel_type = rel.get("type", "cites")

            if src in entry_map and tgt in entry_map:
                if rel_type in RELATION_TYPES:
                    G.add_edge(src, tgt, relation=rel_type)
                else:
                    G.add_edge(src, tgt, relation=rel_type)

        # Add implicit "cites" edges from bibtex crossref if available
        for key, entry in entry_map.items():
            crossref = entry.get("crossref", "")
            if crossref and crossref in entry_map and not G.has_edge(key, crossref):
                G.add_edge(key, crossref, relation="cites")

        return G

    # -- visualization ------------------------------------------------------

    def visualize(self, graph: Any, format: str = "mermaid") -> str:
        """Visualize graph in mermaid, tikz, or dot format.

        Args:
            graph: networkx.DiGraph
            format: "mermaid", "tikz", or "dot"

        Returns:
            Visualization string.
        """
        try:
            import networkx as nx
        except ImportError:
            return "Error: networkx is required for visualization"

        if graph.number_of_nodes() == 0:
            return f"% Empty graph ({format})"

        if format == "mermaid":
            return self._to_mermaid(graph)
        elif format == "tikz":
            return self._to_tikz(graph)
        elif format == "dot":
            return self._to_dot(graph)
        else:
            return f"Error: Unknown format '{format}'. Use mermaid, tikz, or dot."

    def _to_mermaid(self, graph: Any) -> str:
        lines = ["graph TD"]

        # Node definitions with labels
        for node, data in graph.nodes(data=True):
            label = data.get("title", node)[:40]
            safe_label = label.replace('"', "'")
            lines.append(f'    {node}["{safe_label}"]')

        # Edges with relation types
        relation_styles = {
            "cites": "-->",
            "contradicts": "-.->|contradicts|",
            "extends": "==>|extends|",
            "supplements": "-->|supplements|",
            "method_similar": "-.->|method|",
            "theory_similar": "-.->|theory|",
        }

        for src, tgt, data in graph.edges(data=True):
            rel = data.get("relation", "cites")
            arrow = relation_styles.get(rel, "-->")
            lines.append(f"    {src} {arrow} {tgt}")

        return "\n".join(lines)

    def _to_tikz(self, graph: Any) -> str:
        lines = [
            "\\begin{tikzpicture}",
            "\\tikzset{node distance=2cm, auto}",
        ]

        # Position nodes in a circle
        import math
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        for i, node in enumerate(nodes):
            angle = 2 * math.pi * i / max(n, 1)
            x = round(3 * math.cos(angle), 2)
            y = round(3 * math.sin(angle), 2)
            label = graph.nodes[node].get("title", node)[:30]
            safe_label = label.replace("&", "\\&").replace("%", "\\%")
            lines.append(f"    \\node ({node}) at ({x},{y}) {{{safe_label}}};")

        # Edges
        relation_arrows = {
            "cites": "->",
            "contradicts": "->[dashed]",
            "extends": "->[thick]",
            "supplements": "->[dotted]",
            "method_similar": "->[blue]",
            "theory_similar": "->[red]",
        }

        for src, tgt, data in graph.edges(data=True):
            rel = data.get("relation", "cites")
            arrow = relation_arrows.get(rel, "->")
            lines.append(f"    \\draw[{arrow}] ({src}) -- ({tgt});")

        lines.append("\\end{tikzpicture}")
        return "\n".join(lines)

    def _to_dot(self, graph: Any) -> str:
        lines = ["digraph LiteratureGraph {"]
        lines.append('    rankdir=LR;')
        lines.append('    node [shape=box];')

        for node, data in graph.nodes(data=True):
            label = data.get("title", node)[:40]
            safe_label = label.replace('"', "'")
            lines.append(f'    "{node}" [label="{safe_label}"];')

        relation_colors = {
            "cites": "black",
            "contradicts": "red",
            "extends": "green",
            "supplements": "blue",
            "method_similar": "orange",
            "theory_similar": "purple",
        }

        for src, tgt, data in graph.edges(data=True):
            rel = data.get("relation", "cites")
            color = relation_colors.get(rel, "black")
            lines.append(f'    "{src}" -> "{tgt}" [label="{rel}", color={color}];')

        lines.append("}")
        return "\n".join(lines)

    # -- cluster detection --------------------------------------------------

    def detect_clusters(self, graph: Any) -> List[Dict[str, Any]]:
        """Detect communities using Louvain algorithm.

        Returns list of community dicts with nodes and size.
        """
        try:
            import networkx as nx
        except ImportError:
            return [{"error": "networkx is required for cluster detection"}]

        if graph.number_of_nodes() == 0:
            return []

        try:
            import community as community_louvain
        except ImportError:
            # Fallback: try python-louvain package name
            try:
                from community import community_louvain
            except ImportError:
                return [{"error": "python-louvain package is required for Louvain clustering"}]

        # Louvain works on undirected graphs
        undirected = graph.to_undirected()
        partition = community_louvain.best_partition(undirected)

        communities: Dict[int, List[str]] = {}
        for node, comm_id in partition.items():
            communities.setdefault(comm_id, []).append(node)

        result = []
        for comm_id, nodes in sorted(communities.items()):
            subgraph = graph.subgraph(nodes)
            result.append({
                "community_id": comm_id,
                "size": len(nodes),
                "nodes": nodes,
                "internal_edges": subgraph.number_of_edges(),
                "density": round(nx.density(subgraph), 4) if len(nodes) > 1 else 0.0,
            })

        return result

    # -- key papers ---------------------------------------------------------

    def find_key_papers(self, graph: Any) -> List[Dict[str, Any]]:
        """Find key papers using PageRank and degree centrality.

        Returns ranked list of papers with centrality scores.
        """
        try:
            import networkx as nx
        except ImportError:
            return [{"error": "networkx is required for key paper detection"}]

        if graph.number_of_nodes() == 0:
            return []

        try:
            pagerank = nx.pagerank(graph)
        except Exception:
            pagerank = {}

        try:
            in_degree = dict(graph.in_degree())
        except Exception:
            in_degree = {}

        try:
            out_degree = dict(graph.out_degree())
        except Exception:
            out_degree = {}

        try:
            betweenness = nx.betweenness_centrality(graph)
        except Exception:
            betweenness = {}

        papers = []
        for node, data in graph.nodes(data=True):
            papers.append({
                "id": node,
                "title": data.get("title", ""),
                "author": data.get("author", ""),
                "year": data.get("year", ""),
                "pagerank": round(pagerank.get(node, 0.0), 6),
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "betweenness": round(betweenness.get(node, 0.0), 6),
            })

        # Sort by PageRank descending
        papers.sort(key=lambda x: x["pagerank"], reverse=True)
        return papers

    # -- statistics ---------------------------------------------------------

    def graph_stats(self, graph: Any) -> Dict[str, Any]:
        """Return basic graph statistics."""
        try:
            import networkx as nx
        except ImportError:
            return {"error": "networkx is required"}

        if graph.number_of_nodes() == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "density": 0.0,
                "is_connected": False,
                "components": 0,
            }

        undirected = graph.to_undirected()
        try:
            components = nx.number_connected_components(undirected)
        except Exception:
            components = 0

        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": round(nx.density(graph), 4),
            "is_connected": nx.is_connected(undirected) if components > 0 else False,
            "components": components,
            "average_clustering": round(nx.average_clustering(undirected), 4) if graph.number_of_nodes() > 1 else 0.0,
        }

    # -- pyvis export (optional) --------------------------------------------

    def to_pyvis(
        self,
        graph: Any,
        output_path: str,
        height: str = "600px",
        width: str = "100%",
    ) -> Dict[str, Any]:
        """Export graph to interactive HTML using pyvis (optional dependency)."""
        try:
            from pyvis.network import Network
        except ImportError:
            return {"error": "pyvis is required for interactive visualization. Install with: pip install pyvis"}

        net = Network(height=height, width=width, directed=True, notebook=False)

        relation_colors = {
            "cites": "#000000",
            "contradicts": "#ef4444",
            "extends": "#22c55e",
            "supplements": "#3b82f6",
            "method_similar": "#f59e0b",
            "theory_similar": "#8b5cf6",
        }

        for node, data in graph.nodes(data=True):
            label = data.get("title", node)[:30]
            net.add_node(
                node,
                label=label,
                title=f"{data.get('title', node)}\n{data.get('author', '')} ({data.get('year', '')})",
            )

        for src, tgt, data in graph.edges(data=True):
            rel = data.get("relation", "cites")
            net.add_edge(
                src,
                tgt,
                label=rel,
                color=relation_colors.get(rel, "#999999"),
            )

        net.show(output_path)
        return {"success": True, "output_path": output_path}
