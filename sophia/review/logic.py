"""Logic consistency checker for academic papers.

Verifies:
- Methodology matches research question
- Evidence supports conclusions
- Hypothesis-method-conclusion chain is intact
"""

import re
from typing import Any, Dict, List


class LogicChecker:
    """Check logical consistency of paper arguments."""

    def check(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Run logic checks."""
        findings = []
        score = 100.0

        findings.extend(self._check_methodology_match(doc))
        findings.extend(self._check_evidence_support(doc))
        findings.extend(self._check_conclusion_chain(doc))

        # Deduct score based on severity
        for f in findings:
            sev = f.get("severity", "minor")
            if sev == "fatal":
                score -= 20
            elif sev == "major":
                score -= 10
            elif sev == "minor":
                score -= 3

        score = max(0.0, min(100.0, score))

        return {
            "dimension": "logic",
            "score": round(score, 1),
            "pass": score >= 70,
            "findings": findings,
            "summary": self._summary(findings),
        }

    def _check_methodology_match(self, doc: Dict) -> List[Dict]:
        """Check if methodology matches research question."""
        issues = []
        text = self._extract_full_text(doc)
        lower_text = text.lower()

        # Detect research question type
        is_causal = any(k in lower_text for k in ("effect", "impact", "因果", "影响", "效应"))
        is_descriptive = any(k in lower_text for k in ("describe", "distribution", "描述", "分布"))
        is_relational = any(k in lower_text for k in ("correlation", "relationship", "相关", "关系"))

        # Detect methods mentioned
        has_experiment = any(k in lower_text for k in ("experiment", "randomized", "实验", "随机"))
        has_did = any(k in lower_text for k in ("difference-in-differences", "双重差分", "did", "双重差分法"))
        has_iv = any(k in lower_text for k in ("instrumental variable", "工具变量", "iv"))
        has_regression = any(k in lower_text for k in ("regression", "回归", "ols"))
        has_correlation = any(k in lower_text for k in ("correlation", "pearson", "spearman", "相关分析"))

        # Causal question without causal method
        if is_causal and not (has_experiment or has_did or has_iv or has_regression):
            issues.append({
                "type": "methodology_mismatch",
                "severity": "major",
                "location": "Methods section",
                "detail": "Research question implies causal inference, but no causal method (experiment, DiD, IV, regression) is mentioned.",
                "suggestion": "Consider using a causal inference method appropriate for your data and design.",
            })

        # Correlation question without correlation method
        if is_relational and not (has_correlation or has_regression):
            issues.append({
                "type": "methodology_mismatch",
                "severity": "minor",
                "location": "Methods section",
                "detail": "Research question asks about relationships, but no correlation or regression analysis is mentioned.",
                "suggestion": "Consider adding correlation analysis or regression to examine relationships.",
            })

        return issues

    def _check_evidence_support(self, doc: Dict) -> List[Dict]:
        """Check if conclusions are supported by evidence in Results."""
        issues = []
        sections = doc.get("sections", {})

        results_text = ""
        discussion_text = ""
        for key, sec in sections.items():
            title_lower = sec.get("title", "").lower()
            content = sec.get("content", "")
            if "result" in title_lower or "结果" in title_lower:
                results_text += "\n" + content
            if "discussion" in title_lower or "讨论" in title_lower:
                discussion_text += "\n" + content

        if not results_text and discussion_text:
            issues.append({
                "type": "missing_evidence",
                "severity": "fatal",
                "location": "Results section",
                "detail": "Discussion section exists but Results section is empty or missing.",
                "suggestion": "Add a Results section with empirical findings before the Discussion.",
            })

        # Check if discussion claims go beyond results
        if results_text and discussion_text:
            # Heuristic: if discussion mentions something not in results
            strong_claims = re.findall(r'\b(prove|demonstrate|confirm|establish|表明|证明|证实)\b', discussion_text, re.I)
            if len(strong_claims) > 3:
                issues.append({
                    "type": "overstated_claims",
                    "severity": "major",
                    "location": "Discussion section",
                    "detail": f"Discussion uses strong causal language ({len(strong_claims)} instances) that may exceed what the results support.",
                    "suggestion": "Use more cautious language (e.g., 'suggests', 'is consistent with') unless strong causal evidence exists.",
                })

        return issues

    def _check_conclusion_chain(self, doc: Dict) -> List[Dict]:
        """Check hypothesis-method-conclusion chain."""
        issues = []
        text = self._extract_full_text(doc)
        lower_text = text.lower()

        has_hypothesis = any(k in lower_text for k in ("hypothesis", "假设", "h1", "h0", "null hypothesis"))
        has_method = any(k in lower_text for k in ("method", "方法", "analysis", "分析"))
        has_result = any(k in lower_text for k in ("result", "结果", "finding", "发现"))
        has_conclusion = any(k in lower_text for k in ("conclusion", "结论", "讨论"))

        if has_hypothesis and not has_result:
            issues.append({
                "type": "broken_chain",
                "severity": "fatal",
                "location": "overall",
                "detail": "Hypotheses are stated but no results are reported to test them.",
                "suggestion": "Add a Results section that tests each hypothesis.",
            })

        if has_result and not has_conclusion:
            issues.append({
                "type": "broken_chain",
                "severity": "major",
                "location": "overall",
                "detail": "Results are reported but there is no Conclusion or Discussion section.",
                "suggestion": "Add a Conclusion or Discussion section to interpret the results.",
            })

        if has_method and not has_result:
            issues.append({
                "type": "broken_chain",
                "severity": "fatal",
                "location": "overall",
                "detail": "Methods are described but no results are presented.",
                "suggestion": "Add a Results section.",
            })

        return issues

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
            return "No logical consistency issues detected."
        fatal = sum(1 for f in findings if f.get("severity") == "fatal")
        major = sum(1 for f in findings if f.get("severity") == "major")
        return f"Found {len(findings)} issues: {fatal} fatal, {major} major."
