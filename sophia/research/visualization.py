"""Research visualization engine."""
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sophia.research._input import resolve_parent_ids

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import scipy.stats as _stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import networkx as _nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from sophia.research.workspace_guard import WorkspaceGuard


def _require_mpl():
    if not HAS_MPL:
        raise RuntimeError(
            "matplotlib/seaborn is required for visualization. "
            "Install with: pip install matplotlib seaborn"
        )


class VisualizationEngine:
    """Generates research-quality statistical plots saved to the workspace."""

    def __init__(self, workspace: str, store=None, guard=None):
        self.guard = guard or WorkspaceGuard(workspace)
        self.store = store

    # ------------------------------------------------------------------
    # ResultStore plumbing
    # ------------------------------------------------------------------

    def _sanitize_params(self, args: dict) -> dict:
        """Replace bulky arrays / long strings with summaries."""
        clean: Dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, list):
                if len(v) > 80:
                    clean[k] = f"<list len={len(v)}>"
                elif v and isinstance(v[0], str):
                    total_chars = sum(len(s) for s in v)
                    if total_chars > 4000:
                        clean[k] = f"<list of {len(v)} strings, total_chars={total_chars}>"
                    else:
                        clean[k] = v
                elif v and isinstance(v[0], (list, tuple)):
                    total = sum(len(row) if hasattr(row, "__len__") else 1 for row in v)
                    if total > 200:
                        clean[k] = f"<nested list outer={len(v)} total={total}>"
                    else:
                        clean[k] = v
                else:
                    clean[k] = v
            elif isinstance(v, dict):
                total = sum(len(str(x)) for x in v.values())
                if total > 4000:
                    clean[k] = f"<dict keys={len(v)}>"
                else:
                    clean[k] = v
            elif isinstance(v, str) and len(v) > 2000:
                clean[k] = f"<str len={len(v)}>"
            else:
                clean[k] = v
        return clean

    def _final(self, args: dict, result: dict, tool_name: str) -> str:
        """Persist a successful result to the store and embed result_id."""
        if "error" in result:
            return json.dumps(result)
        if self.store is None:
            return json.dumps(result)
        parents = resolve_parent_ids(args)
        sanitized = self._sanitize_params(args)
        rid = self.store.store(
            result,
            kind="result",
            tool=tool_name,
            params=sanitized,
            parents=parents,
        )
        result = {**result, "result_id": rid}
        return json.dumps(result)


    # ------------------------------------------------------------------
    # plot  (general statistical)
    # ------------------------------------------------------------------
    def plot(self, args: dict) -> str:
        """General statistical plot.

        Args:
            args: {
                data: list or list of lists,
                type: str ('box'|'violin'|'hist'|'bar'|'scatter'|'qq'|'line'),
                labels: list, title: str, x_label: str, y_label: str,
                filename: str, groups: list of str
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        plot_type = args.get("type", "hist")
        data = args.get("data", [])
        labels = args.get("labels", [])
        title = args.get("title", plot_type.capitalize())
        x_label = args.get("x_label", "")
        y_label = args.get("y_label", "")
        filename = args.get("filename", f"{plot_type}_plot.png")
        groups = args.get("groups", [])

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            if plot_type == "box":
                if isinstance(data[0], list):
                    tick_labels = labels if labels else [f"G{i+1}" for i in range(len(data))]
                    ax.boxplot(data, tick_labels=tick_labels)
                else:
                    ax.boxplot(data)
                ax.set_ylabel(y_label or "Value")

            elif plot_type == "violin":
                if isinstance(data[0], list):
                    parts = ax.violinplot(data, showmeans=True, showmedians=True)
                else:
                    parts = ax.violinplot([data], showmeans=True, showmedians=True)
                ax.set_ylabel(y_label or "Value")

            elif plot_type == "hist":
                bins = args.get("bins", "auto")
                if isinstance(data[0], list):
                    for i, d in enumerate(data):
                        label = labels[i] if i < len(labels) else f"G{i+1}"
                        ax.hist(d, bins=bins, alpha=0.6, label=label)
                    ax.legend()
                else:
                    ax.hist(data, bins=bins, alpha=0.7)
                ax.set_xlabel(x_label or "Value")
                ax.set_ylabel(y_label or "Frequency")

            elif plot_type == "bar":
                if labels:
                    x_pos = range(len(labels))
                    ax.bar(x_pos, data, tick_label=labels)
                else:
                    ax.bar(range(len(data)), data)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label or "Value")

            elif plot_type == "scatter":
                if isinstance(data[0], list) and len(data) >= 2:
                    ax.scatter(data[0], data[1])
                    ax.set_xlabel(x_label or "X")
                    ax.set_ylabel(y_label or "Y")
                elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)):
                    xs = [p[0] for p in data]
                    ys = [p[1] for p in data]
                    ax.scatter(xs, ys)
                    ax.set_xlabel(x_label or "X")
                    ax.set_ylabel(y_label or "Y")
                else:
                    ax.scatter(range(len(data)), data)

            elif plot_type == "qq":
                if HAS_SCIPY:
                    flat_data = np.array(data).flatten()
                    _stats.probplot(flat_data, dist="norm", plot=ax)
                else:
                    # Manual QQ plot against normal distribution
                    flat_data = np.sort(np.array(data).flatten())
                    n = len(flat_data)
                    theoretical = _ppf(np.arange(1, n + 1) / (n + 1))
                    ax.scatter(theoretical, flat_data, s=10, alpha=0.6)
                    # Reference line
                    if n > 0:
                        slope = (flat_data[-1] - flat_data[0]) / (theoretical[-1] - theoretical[0]) if (theoretical[-1] - theoretical[0]) != 0 else 1
                        intercept = flat_data[0] - slope * theoretical[0]
                        ax.plot(theoretical, slope * theoretical + intercept, "r--", linewidth=1)
                    ax.set_xlabel("Theoretical Quantiles")
                    ax.set_ylabel("Sample Quantiles")

            elif plot_type == "line":
                if isinstance(data[0], list):
                    for i, d in enumerate(data):
                        label = labels[i] if i < len(labels) else f"G{i+1}"
                        ax.plot(d, label=label)
                    ax.legend()
                else:
                    ax.plot(data)
                ax.set_xlabel(x_label or "Index")
                ax.set_ylabel(y_label or "Value")

            else:
                plt.close(fig)
                return json.dumps({"error": f"Unknown plot type: {plot_type}"})

            ax.set_title(title)
            fig.tight_layout()
            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "type": plot_type,
                "size_bytes": size,
            }, "research_plot")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"Plot failed: {exc}"})

    # ------------------------------------------------------------------
    # forest_plot
    # ------------------------------------------------------------------
    def forest_plot(self, args: dict) -> str:
        """Forest plot for meta-analysis.

        Args:
            args: {
                studies: list of str,
                effects: list of float,
                cis_low: list, cis_high: list,
                x_label: str, title: str, filename: str,
                pooled_effect: float, pooled_ci_low: float, pooled_ci_high: float
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        studies = args.get("studies", [])
        effects = args.get("effects", [])
        cis_low = args.get("cis_low", [])
        cis_high = args.get("cis_high", [])
        x_label = args.get("x_label", "Effect Size")
        title = args.get("title", "Forest Plot")
        filename = args.get("filename", "forest_plot.png")
        pooled_effect = args.get("pooled_effect", None)
        pooled_ci_low = args.get("pooled_ci_low", None)
        pooled_ci_high = args.get("pooled_ci_high", None)

        try:
            n = len(studies)
            total = n + (1 if pooled_effect is not None else 0)
            fig, ax = plt.subplots(figsize=(10, max(3, total * 0.6 + 1.5)))

            y_positions = list(range(n - 1, -1, -1))

            # Error bars for individual studies
            for i in range(n):
                ax.errorbar(
                    effects[i], y_positions[i],
                    xerr=[[effects[i] - cis_low[i]], [cis_high[i] - effects[i]]],
                    fmt="s", color="navy", capsize=4, markersize=6,
                )

            # Pooled estimate
            if pooled_effect is not None:
                pooled_y = -1
                ax.errorbar(
                    pooled_effect, pooled_y,
                    xerr=[[pooled_effect - pooled_ci_low], [pooled_ci_high - pooled_effect]],
                    fmt="D", color="red", capsize=6, markersize=8,
                )
                all_labels = studies + ["Pooled"]
                all_y = y_positions + [pooled_y]
            else:
                all_labels = studies
                all_y = y_positions

            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_yticks(all_y)
            ax.set_yticklabels(all_labels)
            ax.set_xlabel(x_label)
            ax.set_title(title)
            fig.tight_layout()

            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "studies": n,
                "size_bytes": size,
            }, "research_forest_plot")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"Forest plot failed: {exc}"})

    # ------------------------------------------------------------------
    # funnel_plot
    # ------------------------------------------------------------------
    def funnel_plot(self, args: dict) -> str:
        """Funnel plot for publication bias.

        Args:
            args: {
                effects: list of float,
                se: list of float,
                x_label: str, y_label: str,
                title: str, filename: str
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        effects = np.array(args.get("effects", []))
        se = np.array(args.get("se", []))
        x_label = args.get("x_label", "Effect Size")
        y_label = args.get("y_label", "Standard Error")
        title = args.get("title", "Funnel Plot")
        filename = args.get("filename", "funnel_plot.png")

        try:
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.scatter(effects, se, s=30, alpha=0.7, zorder=3)

            # Pseudo confidence interval lines
            if len(effects) > 0:
                mean_effect = float(np.mean(effects))
                max_se = float(np.max(se)) * 1.1
                se_line = np.linspace(0, max_se, 100)
                # 95% CI bounds: effect +/- 1.96*SE
                ax.plot(mean_effect - 1.96 * se_line, se_line, "k--", linewidth=0.8, alpha=0.6)
                ax.plot(mean_effect + 1.96 * se_line, se_line, "k--", linewidth=0.8, alpha=0.6)
                ax.axvline(x=mean_effect, color="gray", linestyle=":", linewidth=0.8)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.invert_yaxis()  # Smaller SE at top
            ax.set_title(title)
            fig.tight_layout()

            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "points": len(effects),
                "size_bytes": size,
            }, "research_funnel_plot")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"Funnel plot failed: {exc}"})

    # ------------------------------------------------------------------
    # network_plot
    # ------------------------------------------------------------------
    def network_plot(self, args: dict) -> str:
        """Network visualization.

        Args:
            args: {
                nodes: list of str,
                edges: list of {source: str, target: str, weight: float},
                layout: str ('spring'|'circular'|'kamada'),
                title: str, filename: str
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        nodes = args.get("nodes", [])
        edges = args.get("edges", [])
        layout = args.get("layout", "spring")
        title = args.get("title", "Network Plot")
        filename = args.get("filename", "network_plot.png")

        try:
            if not nodes:
                return json.dumps({"error": "No nodes provided"})

            fig, ax = plt.subplots(figsize=(10, 10))

            positions: Dict[str, Tuple[float, float]] = {}

            if HAS_NX:
                G = _nx.Graph()
                G.add_nodes_from(nodes)
                for e in edges:
                    G.add_edge(e["source"], e["target"], weight=e.get("weight", 1.0))
                if layout == "circular":
                    positions = _nx.circular_layout(G)
                elif layout == "kamada":
                    try:
                        positions = _nx.kamada_kawai_layout(G)
                    except Exception:
                        positions = _nx.spring_layout(G, seed=42)
                else:
                    positions = _nx.spring_layout(G, seed=42)
                edge_weights = [_nx.degree(G, node) for node in G.nodes()]
                node_sizes = [300 + w * 100 for w in edge_weights]
                _nx.draw_networkx_nodes(G, positions, ax=ax, node_size=node_sizes,
                                         node_color="steelblue", alpha=0.8)
                _nx.draw_networkx_labels(G, positions, ax=ax, font_size=9)
                edge_widths = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
                _nx.draw_networkx_edges(G, positions, ax=ax, width=edge_widths,
                                         alpha=0.5, edge_color="gray")
            else:
                # Fallback: simple spring-layout approximation using pure numpy
                n = len(nodes)
                if n == 0:
                    plt.close(fig)
                    return json.dumps({"error": "No nodes provided"})
                # Initialize random positions
                rng = np.random.RandomState(42)
                pos_arr = rng.rand(n, 2)
                # Simple force-directed layout iterations
                node_idx = {name: i for i, name in enumerate(nodes)}
                for _ in range(100):
                    forces = np.zeros((n, 2))
                    # Repulsion between all nodes
                    for i in range(n):
                        for j in range(i + 1, n):
                            diff = pos_arr[i] - pos_arr[j]
                            dist = np.linalg.norm(diff) + 1e-6
                            force = diff / (dist ** 2)
                            forces[i] += force
                            forces[j] -= force
                    # Attraction along edges
                    for e in edges:
                        si = node_idx.get(e["source"])
                        ti = node_idx.get(e["target"])
                        if si is not None and ti is not None:
                            diff = pos_arr[ti] - pos_arr[si]
                            w = e.get("weight", 1.0)
                            forces[si] += diff * w * 0.1
                            forces[ti] -= diff * w * 0.1
                    pos_arr += forces * 0.05

                # Normalize to [0.1, 0.9]
                pos_arr = (pos_arr - pos_arr.min(axis=0)) / (pos_arr.max(axis=0) - pos_arr.min(axis=0) + 1e-9)
                pos_arr = pos_arr * 0.8 + 0.1
                positions = {nodes[i]: (float(pos_arr[i, 0]), float(pos_arr[i, 1])) for i in range(n)}

                # Draw edges
                for e in edges:
                    si = e["source"]
                    ti = e["target"]
                    if si in positions and ti in positions:
                        xs = [positions[si][0], positions[ti][0]]
                        ys = [positions[si][1], positions[ti][1]]
                        w = e.get("weight", 1.0)
                        ax.plot(xs, ys, "gray", linewidth=w, alpha=0.5)

                # Draw nodes
                for node in nodes:
                    x, y = positions[node]
                    ax.scatter(x, y, s=400, c="steelblue", alpha=0.8, zorder=3)
                    ax.annotate(node, (x, y), textcoords="offset points",
                                xytext=(0, 10), ha="center", fontsize=9)

            ax.set_title(title)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.axis("off")
            fig.tight_layout()

            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "nodes": len(nodes),
                "edges": len(edges),
                "size_bytes": size,
            }, "research_network_plot")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"Network plot failed: {exc}"})

    # ------------------------------------------------------------------
    # heatmap
    # ------------------------------------------------------------------
    def heatmap(self, args: dict) -> str:
        """Correlation/data heatmap.

        Args:
            args: {
                matrix: list of lists,
                x_labels: list, y_labels: list,
                title: str, filename: str,
                annot: bool, cmap: str
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        matrix = np.array(args.get("matrix", [[]]))
        x_labels = args.get("x_labels", [])
        y_labels = args.get("y_labels", [])
        title = args.get("title", "Heatmap")
        filename = args.get("filename", "heatmap.png")
        annot = args.get("annot", True)
        cmap = args.get("cmap", "RdBu_r")

        try:
            fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 1.2 + 2),
                                             max(5, len(y_labels) * 1.2 + 2)))

            sns.heatmap(
                matrix, ax=ax, annot=annot, cmap=cmap,
                xticklabels=x_labels if x_labels else True,
                yticklabels=y_labels if y_labels else True,
                fmt=".2f" if annot else "",
            )
            ax.set_title(title)
            fig.tight_layout()

            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "shape": list(matrix.shape),
                "size_bytes": size,
            }, "research_heatmap")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"Heatmap failed: {exc}"})

    # ------------------------------------------------------------------
    # roc_curve
    # ------------------------------------------------------------------
    def roc_curve(self, args: dict) -> str:
        """ROC curve.

        Args:
            args: {
                y_true: list, y_score: list,
                title: str, filename: str, auc: float
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        y_true = np.array(args.get("y_true", []))
        y_score = np.array(args.get("y_score", []))
        title = args.get("title", "ROC Curve")
        filename = args.get("filename", "roc_curve.png")
        auc_val = args.get("auc", None)

        try:
            fig, ax = plt.subplots(figsize=(8, 8))

            if len(y_true) > 0 and len(y_score) > 0:
                # Compute ROC points
                fpr, tpr = _manual_roc(y_true, y_score)
                ax.plot(fpr, tpr, "b-", linewidth=2,
                        label=f"ROC (AUC={auc_val:.3f})" if auc_val is not None else "ROC")
            else:
                ax.plot([], [], "b-", linewidth=2, label="ROC (no data)")

            # Diagonal reference
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(title)
            ax.legend(loc="lower right")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            fig.tight_layout()

            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "auc": auc_val,
                "size_bytes": size,
            }, "research_roc_curve")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"ROC curve failed: {exc}"})

    # ------------------------------------------------------------------
    # confusion_matrix
    # ------------------------------------------------------------------
    def confusion_matrix(self, args: dict) -> str:
        """Confusion matrix heatmap.

        Args:
            args: {
                matrix: list of lists,
                labels: list of str,
                title: str, filename: str
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        matrix = np.array(args.get("matrix", [[]]))
        labels = args.get("labels", [])
        title = args.get("title", "Confusion Matrix")
        filename = args.get("filename", "confusion_matrix.png")

        try:
            n = matrix.shape[0]
            fig, ax = plt.subplots(figsize=(max(5, n + 2), max(5, n + 2)))

            sns.heatmap(
                matrix, ax=ax, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels if labels else range(n),
                yticklabels=labels if labels else range(n),
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(title)
            fig.tight_layout()

            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "size_bytes": size,
            }, "research_confusion_matrix")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"Confusion matrix failed: {exc}"})

    # ------------------------------------------------------------------
    # did_plot  (Difference-in-Differences)
    # ------------------------------------------------------------------
    def did_plot(self, args: dict) -> str:
        """DiD parallel trends plot.

        Args:
            args: {
                time: list,
                treatment: list, control: list,
                intervention_time: float,
                title: str, filename: str
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        time = args.get("time", [])
        treatment = args.get("treatment", [])
        control = args.get("control", [])
        intervention_time = args.get("intervention_time", 0)
        title = args.get("title", "Difference-in-Differences")
        filename = args.get("filename", "did_plot.png")

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(time, treatment, "o-", color="red", label="Treatment", linewidth=2)
            ax.plot(time, control, "s-", color="blue", label="Control", linewidth=2)
            ax.axvline(x=intervention_time, color="gray", linestyle="--", linewidth=1.5,
                        label="Intervention")

            # Shade post-intervention area
            ax.axvspan(intervention_time, max(time) if len(time) > 0 else intervention_time + 1,
                        alpha=0.1, color="gray")

            ax.set_xlabel("Time")
            ax.set_ylabel("Outcome")
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()

            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "size_bytes": size,
            }, "research_did_plot")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"DiD plot failed: {exc}"})

    # ------------------------------------------------------------------
    # experiment_dashboard
    # ------------------------------------------------------------------
    def experiment_dashboard(self, args: dict) -> str:
        """Multi-run experiment comparison.

        Args:
            args: {
                runs: list of {name: str, metrics: dict},
                title: str, filename: str
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        runs = args.get("runs", [])
        title = args.get("title", "Experiment Dashboard")
        filename = args.get("filename", "experiment_dashboard.png")

        try:
            if not runs:
                return json.dumps({"error": "No runs provided"})

            # Collect all metric names
            all_metrics: List[str] = []
            for run in runs:
                for m in run.get("metrics", {}):
                    if m not in all_metrics:
                        all_metrics.append(m)

            n_metrics = len(all_metrics)
            n_runs = len(runs)
            fig, axes = plt.subplots(1, max(1, n_metrics), figsize=(max(6, n_metrics * 5), 6),
                                      squeeze=False)

            for i, metric in enumerate(all_metrics):
                ax = axes[0, i]
                names = [r.get("name", f"Run {j+1}") for j, r in enumerate(runs)]
                values = [r.get("metrics", {}).get(metric, 0) for r in runs]
                bars = ax.bar(range(n_runs), values, tick_label=names, color="steelblue", alpha=0.8)
                ax.set_title(metric)
                ax.set_ylabel(metric)
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
                ax.tick_params(axis="x", rotation=45)

            fig.suptitle(title, fontsize=14)
            fig.tight_layout()

            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "runs": n_runs,
                "metrics": n_metrics,
                "size_bytes": size,
            }, "research_experiment_dashboard")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"Dashboard failed: {exc}"})

    # ------------------------------------------------------------------
    # effect_size_plot
    # ------------------------------------------------------------------
    def effect_size_plot(self, args: dict) -> str:
        """Effect size visualization.

        Args:
            args: {
                effects: list of {name: str, value: float, ci_low: float, ci_high: float},
                title: str, filename: str
            }
        """
        try:
            _require_mpl()
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        effects = args.get("effects", [])
        title = args.get("title", "Effect Sizes")
        filename = args.get("filename", "effect_size_plot.png")

        try:
            n = len(effects)
            if n == 0:
                return json.dumps({"error": "No effects provided"})

            fig, ax = plt.subplots(figsize=(10, max(3, n * 0.6 + 1.5)))

            y_positions = list(range(n - 1, -1, -1))
            names = [e.get("name", f"Effect {i+1}") for i, e in enumerate(effects)]
            values = [e.get("value", 0) for e in effects]
            ci_lows = [e.get("ci_low", e.get("value", 0)) for e in effects]
            ci_highs = [e.get("ci_high", e.get("value", 0)) for e in effects]

            ax.errorbar(
                values, y_positions,
                xerr=[[v - lo for v, lo in zip(values, ci_lows)],
                       [hi - v for v, hi in zip(values, ci_highs)]],
                fmt="o", color="navy", capsize=5, markersize=8,
            )

            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(names)
            ax.set_xlabel("Effect Size")
            ax.set_title(title)
            fig.tight_layout()

            resolved = self.guard.resolve_write(filename, subdir="figures")
            fig.savefig(resolved, dpi=150, bbox_inches="tight")
            plt.close(fig)
            size = os.path.getsize(resolved)

            return self._final(args, {
                "path": resolved,
                "filename": filename,
                "effects": n,
                "size_bytes": size,
            }, "research_effect_size_plot")

        except Exception as exc:
            plt.close("all")
            return json.dumps({"error": f"Effect size plot failed: {exc}"})


# ======================================================================
# Helper functions (module-level, no class dependency)
# ======================================================================

def _ppf(probabilities):
    """Approximate normal inverse CDF (ppf) using rational approximation.

    Used as a fallback when scipy is not available.
    """
    result = np.zeros_like(probabilities, dtype=float)
    for i, p in enumerate(probabilities):
        if p <= 0:
            result[i] = -3.5
        elif p >= 1:
            result[i] = 3.5
        else:
            # Abramowitz and Stegun approximation 26.2.23
            t = math.sqrt(-2.0 * math.log(p)) if p < 0.5 else math.sqrt(-2.0 * math.log(1 - p))
            c0 = 2.515517
            c1 = 0.802853
            c2 = 0.010328
            d1 = 1.432788
            d2 = 0.189269
            d3 = 0.001308
            z = t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
            result[i] = -z if p < 0.5 else z
    return result


def _manual_roc(y_true, y_score):
    """Compute ROC curve points without scipy."""
    # Sort by decreasing score
    desc_indices = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_indices]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / total_pos if total_pos > 0 else 0)
        fpr_list.append(fp / total_neg if total_neg > 0 else 0)

    return np.array(fpr_list), np.array(tpr_list)
