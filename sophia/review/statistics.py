"""Statistical reporting checker for academic papers.

Verifies:
- p-value consistency (reported values match "significant"/"non-significant" language)
- Effect size completeness (Cohen's d, eta², etc. reported alongside p-values)
- Assumption reporting (normality, homoscedasticity, etc. mentioned where required)
"""

import re
from typing import Any, Dict, List


class StatisticsChecker:
    """Check statistical reporting quality."""

    # Effect size labels and their common symbols
    EFFECT_SIZES = {
        "cohen's d": "d",
        "cohen d": "d",
        "hedges' g": "g",
        "eta squared": "η²",
        "partial eta squared": "ηp²",
        "omega squared": "ω²",
        "r squared": "R²",
        "adjusted r squared": "adjusted R²",
        "cramer's v": "V",
        "phi coefficient": "φ",
        "odds ratio": "OR",
        "risk ratio": "RR",
        "mean difference": "MD",
        "standardized mean difference": "SMD",
    }

    # Assumptions commonly required by methods
    ASSUMPTION_MAP = {
        "t-test": ["normality", "homogeneity of variance", "independence"],
        "anova": ["normality", "homogeneity of variance", "independence"],
        "manova": ["multivariate normality", "homogeneity of variance-covariance"],
        "regression": ["linearity", "independence", "homoscedasticity", "normality of residuals"],
        "logistic regression": ["independence", "linearity in logit"],
        "chi-square": ["expected frequency", "independence"],
        "pearson correlation": ["linearity", "normality", "homoscedasticity"],
        "spearman correlation": ["monotonicity"],
        "mann-whitney": ["independence", "ordinal scale"],
        "kruskal-wallis": ["independence", "ordinal scale"],
        "wilcoxon": ["independence", "symmetry"],
        "factor analysis": ["sample size", "multivariate normality"],
        "structural equation modeling": ["multivariate normality", "linearity", "sample size"],
    }

    # Keywords that imply a method was used
    METHOD_KEYWORDS = {
        "t-test": ["t-test", "ttest", "t test", "独立样本", "配对样本"],
        "anova": ["anova", "方差分析", "f-test"],
        "manova": ["manova", "多元方差分析"],
        "regression": ["regression", "回归"],
        "logistic regression": ["logistic regression", "逻辑回归", "logit"],
        "chi-square": ["chi-square", "卡方", "χ²"],
        "pearson correlation": ["pearson correlation", "pearson r"],
        "spearman correlation": ["spearman correlation", "spearman rho"],
        "mann-whitney": ["mann-whitney", "mann whitney", "u test"],
        "kruskal-wallis": ["kruskal-wallis", "kruskal wallis"],
        "wilcoxon": ["wilcoxon", "wilcoxon signed"],
        "factor analysis": ["factor analysis", "因子分析", "exploratory factor analysis", "efa"],
        "structural equation modeling": ["structural equation", "sem", "结构方程"],
    }

    def check(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Run statistical reporting checks."""
        findings = []
        score = 100.0

        text = self._extract_full_text(doc)
        sections = doc.get("sections", {})
        results_text = self._extract_section_text(sections, ["results", "结果", "findings", "发现"])
        methods_text = self._extract_section_text(sections, ["methods", "methodology", "方法", "method"])

        findings.extend(self._check_p_value_consistency(text))
        findings.extend(self._check_effect_size_completeness(text))
        findings.extend(self._check_assumption_reporting(methods_text))
        findings.extend(self._check_multiple_comparison(text))

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
            "dimension": "statistics",
            "score": round(score, 1),
            "pass": score >= 70,
            "findings": findings,
            "summary": self._summary(findings),
        }

    def _check_p_value_consistency(self, text: str) -> List[Dict]:
        """Check that p-value reporting is consistent with significance claims."""
        issues = []

        # Pattern: "significant" but p >= .05
        significant_p = re.findall(
            r'(significant|显著|有意义).*?p\s*[<>=]\s*([0-9]*\.?[0-9]+)',
            text, re.IGNORECASE | re.DOTALL
        )
        for _, p_str in significant_p:
            try:
                p_val = float(p_str)
                if p_val >= 0.05:
                    issues.append({
                        "type": "p_value_inconsistency",
                        "severity": "major",
                        "location": "Results",
                        "detail": f"Claims significance but reports p = {p_val} (>= .05).",
                        "suggestion": "Either change significance claim or verify p-value calculation.",
                    })
            except ValueError:
                continue

        # Pattern: "not significant" but p < .05
        nonsig_p = re.findall(
            r'(not significant|不显著|无意义|non-significant).*?p\s*[<>=]\s*([0-9]*\.?[0-9]+)',
            text, re.IGNORECASE | re.DOTALL
        )
        for _, p_str in nonsig_p:
            try:
                p_val = float(p_str)
                if p_val < 0.05:
                    issues.append({
                        "type": "p_value_inconsistency",
                        "severity": "major",
                        "location": "Results",
                        "detail": f"Claims non-significance but reports p = {p_val} (< .05).",
                        "suggestion": "Either change non-significance claim or verify p-value calculation.",
                    })
            except ValueError:
                continue

        # Pattern: p-values reported without degrees of freedom for t/F tests
        t_f_tests = re.findall(
            r'[tF]\s*\(\s*(\d+)\s*\)\s*=\s*[\d.]+.*?p\s*[<>=]\s*[0-9]*\.?[0-9]+',
            text, re.IGNORECASE
        )
        all_p = re.findall(r'p\s*[<>=]\s*[0-9]*\.?[0-9]+', text, re.IGNORECASE)
        # Exclude p-values that are preceded by a t/F statistic in the same sentence
        p_without_df = []
        for p_str in all_p:
            idx = text.lower().find(p_str.lower())
            snippet = text[max(0, idx - 120):idx]
            if not re.search(r'[tF]\s*\(\s*\d+\s*\)\s*=\s*[\d.]+', snippet, re.IGNORECASE):
                p_without_df.append(p_str)
        if len(p_without_df) > 2 and len(t_f_tests) == 0:
            issues.append({
                "type": "missing_degrees_of_freedom",
                "severity": "minor",
                "location": "Results",
                "detail": f"{len(p_without_df)} p-values reported without test statistics or degrees of freedom.",
                "suggestion": "Report test statistics (t, F, χ²) with degrees of freedom alongside p-values.",
            })

        # Pattern: p = .000 (should be p < .001)
        zero_p = re.findall(r'p\s*=\s*0\.000+', text, re.IGNORECASE)
        if zero_p:
            issues.append({
                "type": "p_value_rounding",
                "severity": "minor",
                "location": "Results",
                "detail": "p = .000 reported. Should use p < .001.",
                "suggestion": "Use p < .001 instead of p = .000 to indicate value below reporting threshold.",
            })

        return issues

    def _check_effect_size_completeness(self, text: str) -> List[Dict]:
        """Check that effect sizes are reported alongside significance tests."""
        issues = []

        # Find significance tests
        sig_tests = re.findall(
            r'\b(t\s*\(\s*\d+\s*\)\s*=|F\s*\(\s*\d+\s*,\s*\d+\s*\)\s*=|χ²\s*\(\s*\d+\s*\)\s*=|chi-square\s*\(\s*\d+\s*\)\s*=)',
            text, re.IGNORECASE
        )

        if not sig_tests:
            return issues

        # Check if any effect size is mentioned
        has_effect_size = any(
            re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)
            for keyword in self.EFFECT_SIZES.keys()
        )

        # Check for common effect size symbols
        has_effect_symbol = bool(re.search(r'\b(d|g|η²|ηp²|ω²|R²|V|φ|OR|RR)\s*[=＝]\s*\d', text))

        if sig_tests and not (has_effect_size or has_effect_symbol):
            issues.append({
                "type": "missing_effect_sizes",
                "severity": "major",
                "location": "Results",
                "detail": f"{len(sig_tests)} significance test(s) reported without any effect size measure.",
                "suggestion": "Report effect sizes (Cohen's d, eta², etc.) alongside p-values. APA 7th edition requires effect sizes.",
            })

        # Check confidence intervals
        has_ci = bool(re.search(r'95%\s*CI|confidence interval|置信区间', text, re.IGNORECASE))
        if sig_tests and not has_ci:
            issues.append({
                "type": "missing_confidence_intervals",
                "severity": "major",
                "location": "Results",
                "detail": "Significance tests reported without confidence intervals.",
                "suggestion": "Report 95% confidence intervals for key estimates. Required by APA 7th edition.",
            })

        return issues

    def _check_assumption_reporting(self, methods_text: str) -> List[Dict]:
        """Check that statistical assumptions are reported for methods used."""
        issues = []
        if not methods_text:
            return issues

        lower_methods = methods_text.lower()

        for method, keywords in self.METHOD_KEYWORDS.items():
            method_used = any(kw in lower_methods for kw in keywords)
            if not method_used:
                continue

            assumptions = self.ASSUMPTION_MAP.get(method, [])
            if not assumptions:
                continue

            reported = []
            missing = []
            for assumption in assumptions:
                if assumption in lower_methods:
                    reported.append(assumption)
                else:
                    missing.append(assumption)

            if missing and not reported:
                issues.append({
                    "type": "missing_assumptions",
                    "severity": "major",
                    "location": "Methods",
                    "detail": f"{method} used but no assumptions reported ({', '.join(missing)}).",
                    "suggestion": f"Report and verify assumptions for {method}: {', '.join(assumptions)}.",
                })
            elif missing:
                issues.append({
                    "type": "incomplete_assumptions",
                    "severity": "minor",
                    "location": "Methods",
                    "detail": f"{method} assumptions partially reported. Missing: {', '.join(missing)}.",
                    "suggestion": f"Complete assumption reporting for {method}.",
                })

        return issues

    def _check_multiple_comparison(self, text: str) -> List[Dict]:
        """Check if multiple comparisons are corrected."""
        issues = []

        # Look for multiple tests
        p_count = len(re.findall(r'p\s*[<>=]\s*[\d.]+', text, re.IGNORECASE))
        if p_count < 3:
            return issues

        # Check for correction methods
        corrections = ["bonferroni", "fdr", "false discovery rate", "holm",
                       "benjamini", "hochberg", "sidak", "scheffe", "tukey",
                       "多重比较校正", "多重比较"]
        has_correction = any(c in text.lower() for c in corrections)

        if p_count >= 5 and not has_correction:
            issues.append({
                "type": "multiple_comparison_uncorrected",
                "severity": "major",
                "location": "Results",
                "detail": f"{p_count} p-values reported but no multiple comparison correction applied.",
                "suggestion": "Apply Bonferroni, FDR (Benjamini-Hochberg), or Tukey correction for multiple comparisons.",
            })
        elif p_count >= 3 and not has_correction:
            issues.append({
                "type": "multiple_comparison_uncorrected",
                "severity": "minor",
                "location": "Results",
                "detail": f"{p_count} p-values reported. Consider multiple comparison correction.",
                "suggestion": "Consider Bonferroni or FDR correction when conducting multiple tests.",
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
    def _extract_section_text(sections: Dict, keywords: List[str]) -> str:
        """Extract text from sections matching keywords."""
        parts = []
        for key, sec in sections.items():
            title_lower = sec.get("title", "").lower()
            if any(kw in title_lower for kw in keywords):
                content = sec.get("content", "")
                if content:
                    parts.append(content)
        return "\n".join(parts)

    @staticmethod
    def _summary(findings: List[Dict]) -> str:
        if not findings:
            return "No statistical reporting issues detected."
        fatal = sum(1 for f in findings if f.get("severity") == "fatal")
        major = sum(1 for f in findings if f.get("severity") == "major")
        return f"Found {len(findings)} issues: {fatal} fatal, {major} major."
