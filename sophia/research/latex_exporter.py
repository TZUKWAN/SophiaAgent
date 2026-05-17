"""LaTeX report exporter: generate journal-ready .tex from ResultStore results."""
from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from sophia.research.apa import APAFormatter
from sophia.research.result_store import ResultStore


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


class LaTeXReporter:
    """Generate a compilable .tex paper skeleton from research results."""

    def __init__(self, result_store: ResultStore):
        self.store = result_store

    def export(self, args: dict) -> str:
        """
        Args:
            result_ids: List[str] — results to include
            title: str
            authors: List[str]
            sections: List[str] — e.g. ["abstract", "methods", "results", "discussion"]
            template: "apa7" | "elsevier" | "ieee"
            include_figures: bool (default True)
            include_tables: bool (default True)
            output_name: str (default "report")

        Returns JSON with: path, compiled, n_results, warnings.
        """
        result_ids = args.get("result_ids", [])
        title = args.get("title", "Research Report")
        authors = args.get("authors", ["Anonymous"])
        sections = args.get("sections", ["abstract", "methods", "results", "discussion"])
        template = args.get("template", "apa7")
        include_tables = bool(args.get("include_tables", True))
        output_name = args.get("output_name", "report")

        # Collect results from store
        results: List[Dict[str, Any]] = []
        warnings: List[str] = []
        for rid in result_ids:
            try:
                data = self.store.get(rid)
                meta = self.store.get_metadata(rid)
                if isinstance(data, dict):
                    results.append({"result_id": rid, "data": data, "meta": meta})
                else:
                    warnings.append(f"result_id {rid}: payload is not a dict, skipping.")
            except Exception as exc:
                warnings.append(f"result_id {rid}: {exc}")

        # Build sections
        abstract_text = self._build_abstract(results, title)
        methods_text = self._build_methods(results)
        results_text = self._build_results(results, include_tables)
        discussion_text = self._build_discussion(results)

        body_parts = []
        for sec in sections:
            if sec == "abstract":
                continue  # handled in template frontmatter
            if sec == "methods":
                body_parts.append(f"\\section{{{self._section_title(sec)}}}\n{methods_text}")
            elif sec == "results":
                body_parts.append(f"\\section{{{self._section_title(sec)}}}\n{results_text}")
            elif sec == "discussion":
                body_parts.append(f"\\section{{{self._section_title(sec)}}}\n{discussion_text}")
            else:
                body_parts.append(f"\\section{{{self._section_title(sec)}}}\n")

        body = "\n\n".join(body_parts)

        # Authors block
        authors_block = self._build_authors(authors, template)

        # Bibliography placeholder
        bib_text = (
            "\\section*{References}\n"
            "\\begin{enumerate}\n"
            "\\item StatsModels: Seabold, S., \& Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. \textit{Proceedings of the 9th Python in Science Conference}.\n"
            "\\item Pingouin: Vallat, R. (2018). Pingouin: statistics in Python. \textit{Journal of Open Source Software}, 3(31), 1026.\n"
            "\\end{enumerate}"
        )

        # Load template
        tpl_path = os.path.join(TEMPLATE_DIR, f"{template}.tex")
        if not os.path.exists(tpl_path):
            tpl_path = os.path.join(TEMPLATE_DIR, "apa7.tex")
        with open(tpl_path, "r", encoding="utf-8") as f:
            tpl = f.read()

        # Escape LaTeX special chars in title / abstract
        safe_title = self._escape_latex(title)
        safe_abstract = self._escape_latex(abstract_text)
        safe_body = body
        safe_bib = bib_text

        tex = (
            tpl.replace("{{title}}", safe_title)
            .replace("{{authors_block}}", authors_block)
            .replace("{{abstract}}", safe_abstract)
            .replace("{{body}}", safe_body)
            .replace("{{bibliography}}", safe_bib)
        )

        # Write output
        reports_dir = os.path.join(self.store.workspace, ".research", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        out_path = os.path.join(reports_dir, f"{output_name}.tex")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tex)

        return json.dumps({
            "path": out_path,
            "template": template,
            "n_results": len(results),
            "warnings": warnings,
        }, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_abstract(results: List[dict], title: str) -> str:
        lines = [
            f"This report presents the findings of \\textit{{{LaTeXReporter._escape_latex(title)}}}. "
            f"A total of {len(results)} analyses were conducted."
        ]
        key_findings = []
        for r in results:
            data = r["data"]
            apa = data.get("apa", "")
            if apa and len(apa) < 300:
                key_findings.append(apa)
        if key_findings:
            lines.append(" Key findings include: " + " ".join(key_findings[:3]))
        return "".join(lines)

    @staticmethod
    def _build_methods(results: List[dict]) -> str:
        paragraphs = []
        seen_tools = set()
        for r in results:
            meta = r["meta"]
            tool = meta.get("tool", "unknown")
            if tool in seen_tools or tool in ("research_load_data", "research_validate_data"):
                continue
            seen_tools.add(tool)
            raw_params = meta.get("params", "{}")
            params = json.loads(raw_params) if isinstance(raw_params, str) else (raw_params or {})
            # Sanitize bulky params
            params_clean = {k: v for k, v in params.items()
                            if k not in ("data", "texts", "X", "y", "groups")
                            and not isinstance(v, (list, dict)) or k in ("test", "method", "type")}
            n_obs = r["data"].get("n", r["data"].get("N", "N/A"))
            para = APAFormatter.methods_section(tool, params_clean, r["data"])
            # Escape for LaTeX
            para = LaTeXReporter._escape_latex(para)
            paragraphs.append(para)
        return "\n\n".join(paragraphs)

    @staticmethod
    def _build_results(results: List[dict], include_tables: bool) -> str:
        paragraphs = []
        tables = []
        for i, r in enumerate(results):
            data = r["data"]
            meta = r["meta"]
            tool = meta.get("tool", "unknown")
            apa = data.get("apa", "")
            if apa:
                paragraphs.append(LaTeXReporter._escape_latex(apa))
            # Generate table for structured results
            if include_tables and tool.startswith("research_"):
                tbl = LaTeXReporter._result_to_table(data, tool, i)
                if tbl:
                    tables.append(tbl)
        text = "\n\n".join(paragraphs)
        if tables:
            text += "\n\n" + "\n\n".join(tables)
        return text

    @staticmethod
    def _build_discussion(results: List[dict]) -> str:
        return (
            "The results reported above should be interpreted in light of their underlying assumptions. "
            "Readers are encouraged to examine diagnostic outputs (e.g., parallel-trends tests, "
            "goodness-of-fit statistics, heterogeneity indices) before drawing causal or generalizable conclusions."
        )

    # ------------------------------------------------------------------
    # Table generators
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_table(data: dict, tool: str, idx: int) -> Optional[str]:
        """Generate a LaTeX table snippet for common result types."""
        # Regression coefficients
        if "coefficients" in data and "std_errors" in data and "p_values" in data:
            coeffs = data["coefficients"]
            ses = data["std_errors"]
            ps = data["p_values"]
            ts = data.get("t_stats", {})
            rows = []
            for name in coeffs:
                b = coeffs[name]
                se = ses.get(name, "")
                t = ts.get(name, "")
                p = ps.get(name, "")
                rows.append(f"{name} & {b:.3f} & {se:.3f} & {t:.2f} & {p:.3f} \\\\")
            if not rows:
                return None
            caption = f"Regression coefficients ({tool.replace('research_', '')})."
            return (
                f"\\begin{{table}}[htbp]\n"
                f"\\centering\n"
                f"\\caption{{{LaTeXReporter._escape_latex(caption)}}}\n"
                f"\\begin{{tabular}}{{lcccc}}\n"
                f"\\toprule\n"
                f"Predictor & $\\beta$ & SE & $t$ & $p$ \\\\\n"
                f"\\midrule\n"
                + "\n".join(rows) +
                f"\n\\bottomrule\n"
                f"\\end{{tabular}}\n"
                f"\\label{{tab:result{idx}}}\n"
                f"\\end{{table}}"
            )

        # Meta-analysis summary
        if "pooled_effect" in data:
            pe = data["pooled_effect"]
            se = data.get("se", "")
            ci_low = data.get("ci_low", "")
            ci_high = data.get("ci_high", "")
            p = data.get("p", "")
            model = data.get("model", "")
            return (
                f"\\begin{{table}}[htbp]\n"
                f"\\centering\n"
                f"\\caption{{{LaTeXReporter._escape_latex(model)} meta-analysis summary.}}\n"
                f"\\begin{{tabular}}{{lcc}}\n"
                f"\\toprule\n"
                f"Statistic & Value \\\\\n"
                f"\\midrule\n"
                f"Pooled effect & {pe:.3f} \\\\\n"
                f"SE & {se:.3f} \\\\\n"
                f"95\\% CI & [{ci_low:.3f}, {ci_high:.3f}] \\\\\n"
                f"$p$-value & {p:.3f} \\\\\n"
                f"\\bottomrule\n"
                f"\\end{{tabular}}\n"
                f"\\label{{tab:meta{idx}}}\n"
                f"\\end{{table}}"
            )

        # Cronbach / reliability
        if "alpha" in data:
            alpha = data["alpha"]
            n_items = data.get("n_items", "")
            n_resp = data.get("n_responses", "")
            return (
                f"\\begin{{table}}[htbp]\n"
                f"\\centering\n"
                f"\\caption{{Reliability statistics.}}\n"
                f"\\begin{{tabular}}{{lc}}\n"
                f"\\toprule\n"
                f"Statistic & Value \\\\\n"
                f"\\midrule\n"
                f"Cronbach's $\\alpha$ & {alpha:.3f} \\\\\n"
                f"Number of items & {n_items} \\\\\n"
                f"Number of responses & {n_resp} \\\\\n"
                f"\\bottomrule\n"
                f"\\end{{tabular}}\n"
                f"\\label{{tab:reliability{idx}}}\n"
                f"\\end{{table}}"
            )

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_title(sec: str) -> str:
        return sec.replace("_", " ").title()

    @staticmethod
    def _build_authors(authors: List[str], template: str) -> str:
        if template == "apa7":
            lines = [f"\\author{{{a}}}" for a in authors]
            return "\n".join(lines)
        else:
            lines = [f"\\author{{{a}}}" for a in authors]
            return "\n".join(lines)

    @staticmethod
    def _escape_latex(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        replacements = [
            ("\\", "\\textbackslash{}"),
            ("&", "\\&"),
            ("%", "\\%"),
            ("$", "\\$"),
            ("#", "\\#"),
            ("_", "\\_"),
            ("{", "\\{"),
            ("}", "\\}"),
            ("~", "\\textasciitilde{}"),
            ("^", "\\textasciicircum{}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text
