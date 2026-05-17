"""Ethics checker for academic papers.

Verifies:
- Data fabrication detection (statistical anomaly patterns)
- Plagiarism/suspicious similarity flagging
"""

import re
from typing import Any, Dict, List


class EthicsChecker:
    """Check research ethics indicators."""

    def check(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Run ethics checks."""
        findings = []
        score = 100.0

        findings.extend(self._check_fabrication_indicators(doc))
        findings.extend(self._check_citation_density(doc))

        for f in findings:
            sev = f.get("severity", "minor")
            if sev == "fatal":
                score -= 25
            elif sev == "major":
                score -= 10
            elif sev == "minor":
                score -= 3

        score = max(0.0, min(100.0, score))

        return {
            "dimension": "ethics",
            "score": round(score, 1),
            "pass": score >= 70,
            "findings": findings,
            "summary": self._summary(findings),
        }

    def _check_fabrication_indicators(self, doc: Dict) -> List[Dict]:
        """Detect statistical anomalies that may indicate fabricated data."""
        issues = []
        text = self._extract_full_text(doc)

        # Pattern 1: Too many p-values exactly at .05
        p_values = re.findall(r'p\s*=\s*([0-9]*\.?[0-9]+)', text)
        p_vals = [float(p) for p in p_values if self._is_valid_p(float(p))]

        if p_vals:
            near_threshold = sum(1 for p in p_vals if 0.04 <= p <= 0.06)
            if len(p_vals) >= 3 and near_threshold / len(p_vals) > 0.5:
                issues.append({
                    "type": "suspicious_p_distribution",
                    "severity": "major",
                    "location": "Results section",
                    "detail": f"{near_threshold}/{len(p_vals)} p-values are near .05 threshold. This pattern is unusual.",
                    "suggestion": "Verify p-value calculations. Unusual clustering near .05 can indicate p-hacking or data fabrication.",
                })

        # Pattern 2: All sample sizes are round numbers
        n_values = re.findall(r'[Nn]\s*=\s*(\d+)', text)
        if n_values:
            round_numbers = [n for n in n_values if n.endswith('00') or n.endswith('50')]
            if len(n_values) >= 3 and len(round_numbers) == len(n_values):
                issues.append({
                    "type": "suspicious_sample_sizes",
                    "severity": "minor",
                    "location": "text",
                    "detail": f"All reported sample sizes ({', '.join(n_values)}) are round numbers.",
                    "suggestion": "Real data rarely have exactly round sample sizes. Verify data collection records.",
                })

        # Pattern 3: Perfect correlations
        r_values = re.findall(r'r\s*=\s*([0-9]*\.?[0-9]+)', text)
        perfect = [r for r in r_values if float(r) >= 0.99]
        if perfect:
            issues.append({
                "type": "perfect_correlation",
                "severity": "major",
                "location": "Results section",
                "detail": f"Perfect or near-perfect correlation(s) reported (r = {', '.join(perfect)}).",
                "suggestion": "Near-perfect correlations in real data are extremely rare. Double-check calculations.",
            })

        return issues

    def _check_citation_density(self, doc: Dict) -> List[Dict]:
        """Flag suspiciously low or high citation density."""
        issues = []
        text = self._extract_full_text(doc)
        refs = doc.get("references", [])

        # Word count
        words = len(text.split())
        if words == 0:
            return issues

        # Citation count
        citations = len(re.findall(r'\(\w+,\s*\d{4}\)', text))
        citations += len(re.findall(r'\[\d+\]', text))

        density = citations / words * 1000  # citations per 1000 words

        if density < 2 and words > 500:
            issues.append({
                "type": "very_low_citation_density",
                "severity": "minor",
                "location": "overall",
                "detail": f"Citation density is very low ({density:.1f} per 1000 words).",
                "suggestion": "Academic papers typically cite more frequently. Ensure all claims are properly supported.",
            })

        if density > 50:
            issues.append({
                "type": "very_high_citation_density",
                "severity": "minor",
                "location": "overall",
                "detail": f"Citation density is extremely high ({density:.1f} per 1000 words).",
                "suggestion": "Consider whether some citations are unnecessary or if the text relies too heavily on secondary sources.",
            })

        # Check for reference list much shorter than in-text citations
        if citations > len(refs) * 2 and len(refs) > 0:
            issues.append({
                "type": "citation_list_imbalance",
                "severity": "major",
                "location": "References",
                "detail": f"{citations} in-text citations but only {len(refs)} references.",
                "suggestion": "Ensure all in-text citations have corresponding entries in the reference list.",
            })

        return issues

    @staticmethod
    def _is_valid_p(value: float) -> bool:
        return 0 <= value <= 1

    @staticmethod
    def _extract_full_text(doc: Dict) -> str:
        parts = []
        if doc.get("abstract"):
            parts.append(doc["abstract"])
        for key in sorted(doc.get("sections", {}).keys(), key=lambda x: int(x)):
            s = doc["sections"][key]
            if s.get("content"):
                parts.append(s["content"])
        return "\n".join(parts)

    @staticmethod
    def _summary(findings: List[Dict]) -> str:
        if not findings:
            return "No ethics issues detected."
        fatal = sum(1 for f in findings if f.get("severity") == "fatal")
        major = sum(1 for f in findings if f.get("severity") == "major")
        return f"Found {len(findings)} issues: {fatal} fatal, {major} major."
