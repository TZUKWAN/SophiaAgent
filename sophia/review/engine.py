"""ReviewEngine: orchestrates all six dimension checkers.

Usage:
    engine = ReviewEngine(result_store=store)
    report = engine.review(doc, citation_style="apa7")
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sophia.review.authenticity import AuthenticityChecker
from sophia.review.logic import LogicChecker
from sophia.review.citations import CitationChecker
from sophia.review.language import LanguageChecker
from sophia.review.statistics import StatisticsChecker
from sophia.review.ethics import EthicsChecker

logger = logging.getLogger(__name__)


class ReviewEngine:
    """Main review engine that coordinates all dimension checkers."""

    DIMENSIONS = ["authenticity", "logic", "citations", "language", "statistics", "ethics"]
    DIMENSION_WEIGHTS = {
        "authenticity": 0.20,
        "logic": 0.20,
        "citations": 0.15,
        "language": 0.10,
        "statistics": 0.20,
        "ethics": 0.15,
    }

    SEVERITY_ORDER = {"fatal": 0, "major": 1, "minor": 2, "suggestion": 3}

    def __init__(self, result_store=None):
        self.store = result_store
        self.checkers = {
            "authenticity": AuthenticityChecker(result_store=result_store),
            "logic": LogicChecker(),
            "citations": CitationChecker(),
            "language": LanguageChecker(),
            "statistics": StatisticsChecker(),
            "ethics": EthicsChecker(),
        }

    def review(
        self,
        doc: Dict[str, Any],
        citation_style: str = "apa7",
        dimensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run full review on a document.

        Args:
            doc: Document dict (title, authors, abstract, sections, references)
            citation_style: "apa7" or "gb-t-7714-2015"
            dimensions: Subset of dimensions to check (default: all)

        Returns:
            Structured review report.
        """
        dims = dimensions or self.DIMENSIONS
        results = {}
        all_findings = []

        for dim in dims:
            checker = self.checkers.get(dim)
            if not checker:
                continue
            try:
                if dim == "citations":
                    result = checker.check(doc, citation_style=citation_style)
                else:
                    result = checker.check(doc)
                results[dim] = result
                all_findings.extend(result.get("findings", []))
            except Exception as exc:
                logger.warning("Review dimension '%s' failed: %s", dim, exc)
                results[dim] = {
                    "dimension": dim,
                    "score": 0.0,
                    "pass": False,
                    "findings": [{
                        "type": "checker_error",
                        "severity": "minor",
                        "detail": f"Review checker failed: {exc}",
                    }],
                    "summary": f"Checker error: {exc}",
                }

        # Compute weighted overall score
        overall_score = 0.0
        total_weight = 0.0
        for dim, result in results.items():
            weight = self.DIMENSION_WEIGHTS.get(dim, 1.0 / len(dims))
            overall_score += result["score"] * weight
            total_weight += weight

        if total_weight > 0:
            overall_score /= total_weight

        # Determine recommendation
        fatal_count = sum(1 for f in all_findings if f.get("severity") == "fatal")
        major_count = sum(1 for f in all_findings if f.get("severity") == "major")

        if fatal_count > 0:
            recommendation = "reject"
        elif major_count > 3:
            recommendation = "major_revision"
        elif major_count > 0:
            recommendation = "minor_revision"
        else:
            recommendation = "accept"

        # Sort findings by severity
        all_findings.sort(key=lambda f: self.SEVERITY_ORDER.get(f.get("severity", "suggestion"), 99))

        return {
            "document_id": doc.get("id", ""),
            "document_title": doc.get("title", ""),
            "overall_score": round(overall_score, 1),
            "recommendation": recommendation,
            "dimensions": results,
            "all_findings": all_findings,
            "critical_issues": [f for f in all_findings if f.get("severity") in ("fatal", "major")],
            "stats": {
                "total_findings": len(all_findings),
                "fatal": fatal_count,
                "major": major_count,
                "minor": sum(1 for f in all_findings if f.get("severity") == "minor"),
                "suggestion": sum(1 for f in all_findings if f.get("severity") == "suggestion"),
            },
        }

    def review_dimension(self, doc: Dict[str, Any], dimension: str, citation_style: str = "apa7") -> Dict[str, Any]:
        """Run a single dimension check."""
        checker = self.checkers.get(dimension)
        if not checker:
            return {"error": f"Unknown dimension: {dimension}"}
        if dimension == "citations":
            return checker.check(doc, citation_style=citation_style)
        return checker.check(doc)
