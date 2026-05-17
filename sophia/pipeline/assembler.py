"""PaperAssembler: build Methods / Results / Tables from ResultStore.

Given a document with embedded result_ids, auto-generate paper content.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PaperAssembler:
    """Assemble paper sections from research results in a ResultStore."""

    def __init__(self, result_store=None):
        self.store = result_store

    def assemble(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Methods and Results sections from result_ids in doc.

        Returns updated doc with auto-populated sections.
        """
        result_ids = self._extract_result_ids(doc)
        if not result_ids:
            logger.info("No result_ids found in document.")
            return doc

        methods_lines = ["## Data and Methods", ""]
        results_lines = ["## Results", ""]
        tables = []

        for rid in result_ids:
            if not self.store:
                continue
            meta = self.store.get_metadata(rid)
            payload = self.store.get(rid)
            if not meta or not payload:
                continue

            tool = meta.get("tool", "")
            params = meta.get("params", {}) or {}

            method_para = self._build_methods_paragraph(tool, params, payload)
            if method_para:
                methods_lines.append(method_para)
                methods_lines.append("")

            result_para = self._build_results_paragraph(tool, payload)
            if result_para:
                results_lines.append(result_para)
                results_lines.append("")

            table = self._build_table(tool, payload)
            if table:
                tables.append(table)

        # Write into doc sections if they exist
        for key, sec in doc.get("sections", {}).items():
            title_lower = sec.get("title", "").lower()
            if any(k in title_lower for k in ("方法", "methods", "methodology", "data")):
                existing = sec.get("content", "")
                if not existing.strip():
                    sec["content"] = "\n".join(methods_lines)
                    sec["status"] = "completed"
            if any(k in title_lower for k in ("结果", "results", "findings", "发现")):
                existing = sec.get("content", "")
                if not existing.strip():
                    sec["content"] = "\n".join(results_lines)
                    sec["status"] = "completed"

        doc["_assembled_tables"] = tables
        return doc

    def _extract_result_ids(self, doc: Dict[str, Any]) -> List[str]:
        """Scan doc content for result_id references."""
        ids = []
        text = ""
        for sec in doc.get("sections", {}).values():
            text += " " + sec.get("content", "")
        # Match result_id patterns: res_xxxxxx, result_id: xxx, etc.
        ids = list(set(re.findall(r'res_[a-f0-9]{6,}', text)))
        return ids

    def _build_methods_paragraph(self, tool: str, params: Dict, payload: Dict) -> Optional[str]:
        """Generate a Methods paragraph describing how the analysis was done."""
        if tool.startswith("research_did"):
            treat = params.get("treatment_col", "treatment")
            post = params.get("post_col", "post")
            entity = params.get("entity_col", "entity")
            return (
                f"We employed a difference-in-differences (DiD) design to estimate the causal effect. "
                f"Panel data were structured with unit identifier '{entity}', time indicator '{post}', "
                f"and treatment status '{treat}'. Two-way fixed effects (TWFE) estimation with clustered standard errors was used."
            )
        if tool.startswith("research_psm"):
            return (
                "Propensity score matching (PSM) was used to construct a comparable control group. "
                "Nearest-neighbor matching on the logit of the propensity score was performed, "
                "and balance was assessed via standardized mean differences."
            )
        if tool.startswith("research_ttest"):
            return "An independent-samples t-test was conducted to compare group means."
        if tool.startswith("research_anova"):
            return "One-way analysis of variance (ANOVA) was used to compare means across groups."
        if tool.startswith("research_regression"):
            return "Ordinary least squares (OLS) regression was estimated."
        if tool.startswith("research_scm") or tool.startswith("research_synthetic_control"):
            return (
                "The synthetic control method (Abadie et al., 2010) was applied. "
                "Donor weights were chosen to minimize the pre-treatment root mean squared prediction error (RMSPE)."
            )
        if tool.startswith("research_thematic"):
            return "Thematic analysis was conducted following Braun and Clarke (2006)."
        return None

    def _build_results_paragraph(self, tool: str, payload: Dict) -> Optional[str]:
        """Generate a Results paragraph from the payload."""
        apa = payload.get("apa", "")
        if apa:
            return apa

        if tool.startswith("research_did"):
            beta = payload.get("did_estimate")
            se = payload.get("se")
            p = payload.get("p_value")
            if beta is not None and se is not None and p is not None:
                sig = "significant" if p < 0.05 else "not significant"
                return (
                    f"The DiD estimate indicates a treatment effect of β = {beta:.3f} (SE = {se:.3f}, p = {p:.3f}). "
                    f"The effect is statistically {sig}."
                )
        if tool.startswith("research_ttest"):
            t = payload.get("t_statistic")
            p = payload.get("p_value")
            d = payload.get("cohens_d")
            if t is not None and p is not None:
                return (
                    f"An independent-samples t-test revealed t = {t:.2f}, p = {p:.3f}, "
                    f"Cohen's d = {d:.2f}."
                )
        if tool.startswith("research_anova"):
            f = payload.get("f_statistic")
            p = payload.get("p_value")
            if f is not None and p is not None:
                return f"ANOVA showed F = {f:.2f}, p = {p:.3f}."
        if tool.startswith("research_regression"):
            r2 = payload.get("r_squared")
            if r2 is not None:
                return f"The regression model yielded R² = {r2:.3f}."
        return None

    def _build_table(self, tool: str, payload: Dict) -> Optional[Dict]:
        """Build an APA-style table dict from payload."""
        if tool.startswith("research_regression"):
            coeffs = payload.get("coefficients", [])
            if coeffs:
                return {
                    "title": "Regression Coefficients",
                    "headers": ["Predictor", "B", "SE", "t", "p", "95% CI"],
                    "rows": [
                        [
                            c.get("name", ""),
                            f"{c.get('estimate', 0):.3f}",
                            f"{c.get('std_error', 0):.3f}",
                            f"{c.get('t_statistic', 0):.2f}",
                            f"{c.get('p_value', 0):.3f}",
                            f"[{c.get('ci_lower', 0):.2f}, {c.get('ci_upper', 0):.2f}]",
                        ]
                        for c in coeffs
                    ],
                }
        if tool.startswith("research_did"):
            se_comp = payload.get("se_comparison", {})
            if se_comp:
                return {
                    "title": "DiD Robustness Check — Standard Error Comparison",
                    "headers": ["Specification", "Estimate", "SE", "p-value"],
                    "rows": [
                        [spec, f"{payload.get('did_estimate', 0):.3f}", f"{val:.3f}", "—"]
                        for spec, val in se_comp.items()
                    ],
                }
        return None

