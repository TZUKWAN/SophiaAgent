"""Authenticity checker for academic papers.

Verifies:
- Citation existence (via CrossRef API)
- Data consistency (text descriptions vs actual result values)
- Fact plausibility (basic sanity checks)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class AuthenticityChecker:
    """Check the authenticity of claims, citations, and data in a paper."""

    def __init__(self, result_store=None):
        self.store = result_store

    def check(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Run all authenticity checks and return findings."""
        findings = []
        score = 100.0

        # 1. Citation verification
        citation_issues = self._verify_citations(doc)
        findings.extend(citation_issues)
        score -= len(citation_issues) * 5

        # 2. Data consistency
        data_issues = self._check_data_consistency(doc)
        findings.extend(data_issues)
        score -= len(data_issues) * 8

        # 3. Fact plausibility
        fact_issues = self._check_fact_plausibility(doc)
        findings.extend(fact_issues)
        score -= len(fact_issues) * 3

        score = max(0.0, min(100.0, score))

        return {
            "dimension": "authenticity",
            "score": round(score, 1),
            "pass": score >= 70,
            "findings": findings,
            "summary": self._summary(findings),
        }

    # ------------------------------------------------------------------
    # Citation verification
    # ------------------------------------------------------------------

    def _verify_citations(self, doc: Dict) -> List[Dict]:
        """Verify that cited works exist (best-effort via CrossRef)."""
        issues = []
        refs = doc.get("references", [])

        for i, ref in enumerate(refs):
            # Extract DOI if present
            doi_match = re.search(r'10\.\d{4,}/[^\s]+', ref)
            if doi_match and HAS_REQUESTS:
                doi = doi_match.group(0)
                verified = self._check_crossref(doi)
                if not verified:
                    issues.append({
                        "type": "unverified_citation",
                        "severity": "minor",
                        "location": f"Reference [{i + 1}]",
                        "detail": f"DOI {doi} could not be verified via CrossRef.",
                        "suggestion": "Verify the DOI or provide an alternative citation.",
                    })

            # Heuristic: check for obviously fabricated patterns
            if self._looks_fabricated(ref):
                issues.append({
                    "type": "suspicious_citation",
                    "severity": "major",
                    "location": f"Reference [{i + 1}]",
                    "detail": f"Citation appears potentially fabricated: {ref[:100]}",
                    "suggestion": "Verify this reference against the original source.",
                })

        return issues

    def _check_crossref(self, doi: str) -> bool:
        """Check if a DOI exists in CrossRef (best-effort, no timeout blocking)."""
        try:
            url = f"https://api.crossref.org/works/{doi}"
            resp = requests.get(url, timeout=3)
            return resp.status_code == 200
        except Exception:
            return True  # Assume valid if API fails

    def _looks_fabricated(self, ref: str) -> bool:
        """Heuristic detection of potentially fabricated citations."""
        red_flags = [
            r'\b19\d{2}\b',  # Suspiciously old year (pre-1950)
            r'Vol\.?\s*\d+\s*\(\s*\d+\s*\)',  # Inconsistent volume format
        ]
        # More importantly: check if reference has all required components
        has_author = bool(re.search(r'[A-Z][a-z]+', ref))
        has_year = bool(re.search(r'\b(19|20)\d{2}\b', ref))
        has_title = len(ref) > 30
        has_journal = bool(re.search(r'[A-Z][a-z]+.*\d+', ref))

        missing = []
        if not has_author:
            missing.append("author")
        if not has_year:
            missing.append("year")
        if not has_title:
            missing.append("title")
        if not has_journal:
            missing.append("journal/volume")

        return len(missing) >= 2

    # ------------------------------------------------------------------
    # Data consistency
    # ------------------------------------------------------------------

    def _check_data_consistency(self, doc: Dict) -> List[Dict]:
        """Check that numerical claims in text match stored results."""
        issues = []
        if not self.store:
            return issues

        # Extract all numbers from the document text
        full_text = self._extract_full_text(doc)
        numbers_in_text = self._extract_numbers(full_text)

        # Get all results from ResultStore linked to this document
        result_ids = self._extract_result_ids(doc)
        for rid in result_ids:
            try:
                data = self.store.get(rid)
                if not isinstance(data, dict):
                    continue
                # Check key numerical fields
                for field in ("n", "N", "mean", "sd", "t", "f", "p", "beta", "r_squared", "alpha"):
                    val = data.get(field)
                    if val is not None:
                        found = self._find_approximate(numbers_in_text, float(val))
                        if not found:
                            issues.append({
                                "type": "data_mismatch",
                                "severity": "major",
                                "location": f"result_id {rid}",
                                "detail": f"Result shows {field}={val}, but this value was not found in the paper text.",
                                "suggestion": f"Ensure the result ({field}={val}) is reported in the text.",
                            })
            except Exception as exc:
                logger.debug("Data consistency check failed for %s: %s", rid, exc)

        return issues

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text."""
        nums = []
        for m in re.finditer(r'[-+]?\d+\.?\d*', text):
            try:
                nums.append(float(m.group(0)))
            except ValueError:
                pass
        return nums

    def _find_approximate(self, numbers: List[float], target: float, tolerance: float = 0.01) -> bool:
        """Check if target (or a rounded version) exists in numbers."""
        for n in numbers:
            if abs(n - target) < tolerance:
                return True
            # Also check rounded versions
            if abs(round(n, 2) - round(target, 2)) < tolerance:
                return True
            if abs(round(n, 3) - round(target, 3)) < tolerance:
                return True
        return False

    # ------------------------------------------------------------------
    # Fact plausibility
    # ------------------------------------------------------------------

    def _check_fact_plausibility(self, doc: Dict) -> List[Dict]:
        """Basic sanity checks on claims."""
        issues = []
        full_text = self._extract_full_text(doc)

        # Check for impossible p-values
        p_values = re.findall(r'p\s*[=<>]\s*([\d.]+)', full_text)
        for pv_str in p_values:
            try:
                pv = float(pv_str)
                if pv < 0 or pv > 1:
                    issues.append({
                        "type": "impossible_p_value",
                        "severity": "fatal",
                        "location": "text",
                        "detail": f"p-value {pv} is outside valid range [0, 1].",
                        "suggestion": "Correct the p-value. p-values must be between 0 and 1.",
                    })
            except ValueError:
                pass

        # Check for impossible effect sizes (Cohen's d > 5 is extremely rare)
        d_values = re.findall(r"[Cc]ohen'?s?\s*d\s*=\s*([\d.]+)", full_text)
        for d_str in d_values:
            try:
                d = float(d_str)
                if d > 5.0:
                    issues.append({
                        "type": "implausible_effect_size",
                        "severity": "major",
                        "location": "text",
                        "detail": f"Cohen's d = {d} is extremely large and may indicate an error.",
                        "suggestion": "Double-check the effect size calculation.",
                    })
            except ValueError:
                pass

        # Check sample sizes
        n_matches = re.findall(r'[Nn]\s*=\s*(\d+)', full_text)
        for n_str in n_matches:
            n = int(n_str)
            if n < 10:
                issues.append({
                    "type": "very_small_sample",
                    "severity": "minor",
                    "location": "text",
                    "detail": f"Sample size N={n} is very small.",
                    "suggestion": "Discuss the limitation of small sample size or consider increasing it.",
                })

        return issues

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_full_text(doc: Dict) -> str:
        """Extract all text content from a document."""
        parts = []
        if doc.get("abstract"):
            parts.append(doc["abstract"])
        for key in sorted(doc.get("sections", {}).keys(), key=lambda x: int(x)):
            s = doc["sections"][key]
            if s.get("content"):
                parts.append(s["content"])
        return "\n".join(parts)

    @staticmethod
    def _extract_result_ids(doc: Dict) -> List[str]:
        """Extract result_id references from document."""
        text = json.dumps(doc, ensure_ascii=False)
        return list(set(re.findall(r'res_[a-zA-Z0-9]+', text)))

    @staticmethod
    def _summary(findings: List[Dict]) -> str:
        if not findings:
            return "No authenticity issues detected."
        fatal = sum(1 for f in findings if f.get("severity") == "fatal")
        major = sum(1 for f in findings if f.get("severity") == "major")
        minor = sum(1 for f in findings if f.get("severity") == "minor")
        return f"Found {len(findings)} issues: {fatal} fatal, {major} major, {minor} minor."
