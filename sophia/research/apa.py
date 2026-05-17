"""APA 7th edition style formatter for research results.

Provides human-readable, journal-ready text summaries for statistical
and causal inference outputs.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _p_str(p: Optional[float]) -> str:
    if p is None:
        return "p = .---"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".lstrip("0")


def _sig_word(p: Optional[float]) -> str:
    if p is None:
        return ""
    return "significant" if p < 0.05 else "not significant"


def _ci_str(ci: List[float]) -> str:
    if not ci or len(ci) != 2 or any(math.isnan(x) for x in ci):
        return ""
    return f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"


def _d_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible effect size"
    if ad < 0.5:
        return "small effect size"
    if ad < 0.8:
        return "medium effect size"
    return "large effect size"


def _eta_label(eta: float) -> str:
    aeta = abs(eta)
    if aeta < 0.01:
        return "negligible"
    if aeta < 0.06:
        return "small"
    if aeta < 0.14:
        return "medium"
    return "large"


def _r_label(r: float) -> str:
    ar = abs(r)
    if ar < 0.10:
        return "negligible"
    if ar < 0.30:
        return "small"
    if ar < 0.50:
        return "medium"
    return "large"


class APAFormatter:
    """Generate APA 7th-edition style prose from analysis results."""

    @staticmethod
    def t_test(t: float, df: float, p: Optional[float], d: float,
               mean_diff: float, ci: List[float]) -> str:
        sig = _sig_word(p)
        ci_text = _ci_str(ci)
        return (
            f"An independent-samples t-test revealed a {sig} difference between groups, "
            f"t({df:.0f}) = {t:.2f}, {_p_str(p)}, d = {d:.2f} ({_d_label(d)}). "
            f"The mean difference was {mean_diff:.2f} ({ci_text})."
        )

    @staticmethod
    def anova(f: float, df1: float, df2: float, p: Optional[float],
              eta_sq: float) -> str:
        sig = _sig_word(p)
        return (
            f"A one-way ANOVA showed a {sig} effect, "
            f"F({df1:.0f}, {df2:.0f}) = {f:.2f}, {_p_str(p)}, "
            f"eta squared = {eta_sq:.3f} ({_eta_label(eta_sq)} effect size)."
        )

    @staticmethod
    def correlation(r: float, p: Optional[float], n: int) -> str:
        sig = _sig_word(p)
        direction = "positive" if r > 0 else "negative"
        return (
            f"A Pearson correlation revealed a {sig} {direction} relationship, "
            f"r({n-2}) = {r:.2f}, {_p_str(p)} ({_r_label(r)} effect size)."
        )

    @staticmethod
    def regression_coefficient(b: float, se: float, t: float,
                               p: Optional[float], ci: List[float],
                               predictor_name: str = "predictor") -> str:
        sig = _sig_word(p)
        ci_text = _ci_str(ci)
        return (
            f"For {predictor_name}, the regression coefficient was {b:.3f} (SE = {se:.3f}), "
            f"t = {t:.2f}, {_p_str(p)}. {ci_text}"
        )

    @staticmethod
    def chi_square(chi2: float, df: float, p: Optional[float],
                   n: int, cramers_v: Optional[float] = None) -> str:
        sig = _sig_word(p)
        extra = ""
        if cramers_v is not None:
            extra = f" Cramer's V = {cramers_v:.3f}."
        return (
            f"A chi-square test of independence showed a {sig} association, "
            f"chi-square({df:.0f}, N = {n}) = {chi2:.2f}, {_p_str(p)}.{extra}"
        )

    @staticmethod
    def mann_whitney(u: float, p: Optional[float], n1: int, n2: int) -> str:
        sig = _sig_word(p)
        return (
            f"A Mann-Whitney U test indicated a {sig} difference between groups, "
            f"U = {u:.1f}, {_p_str(p)} (n1 = {n1}, n2 = {n2})."
        )

    @staticmethod
    def wilcoxon(z: float, p: Optional[float], n: int) -> str:
        sig = _sig_word(p)
        return (
            f"A Wilcoxon signed-rank test showed a {sig} difference, "
            f"z = {z:.2f}, {_p_str(p)} (N = {n})."
        )

    @staticmethod
    def did(beta: float, se: float, p: Optional[float], ci: List[float],
            parallel_trends_p: Optional[float] = None,
            method: str = "TWFE") -> str:
        sig = _sig_word(p)
        ci_text = _ci_str(ci)
        pt_text = ""
        if parallel_trends_p is not None:
            pt_pass = "passed" if parallel_trends_p > 0.10 else "failed"
            pt_text = f" The parallel-trends assumption {pt_pass} (F-test p = {parallel_trends_p:.3f})."
        return (
            f"A difference-in-differences analysis ({method}) revealed a {sig} "
            f"treatment effect of {beta:.3f} (SE = {se:.3f}), {_p_str(p)}. "
            f"{ci_text}.{pt_text}"
        )

    @staticmethod
    def rdd(tau: float, se: float, p: Optional[float], bandwidth: float,
            n_within_bw: int) -> str:
        sig = _sig_word(p)
        return (
            f"A regression discontinuity design estimated a {sig} treatment effect of "
            f"{tau:.3f} (SE = {se:.3f}), {_p_str(p)}, using a bandwidth of {bandwidth:.3f} "
            f"(n = {n_within_bw} observations within the bandwidth)."
        )

    @staticmethod
    def iv(beta: float, se: float, p: Optional[float], f_first: float,
           sargan_p: Optional[float] = None) -> str:
        sig = _sig_word(p)
        sargan_text = ""
        if sargan_p is not None:
            sargan_sig = "not rejected" if sargan_p > 0.05 else "rejected"
            sargan_text = (
                f" The overidentification test (Sargan/Hansen J) was {sargan_sig} "
                f"(p = {sargan_p:.3f}), suggesting the instruments are {('valid' if sargan_p > 0.05 else 'potentially invalid')}."
            )
        return (
            f"A two-stage least squares (2SLS) instrumental variable estimation yielded a {sig} "
            f"effect of {beta:.3f} (SE = {se:.3f}), {_p_str(p)}. "
            f"The first-stage F-statistic was {f_first:.2f}.{sargan_text}"
        )

    @staticmethod
    def psm(att: float, se: float, p: Optional[float],
            n_treated: int, n_matched: Optional[int] = None,
            method: str = "nearest neighbor") -> str:
        sig = _sig_word(p)
        match_text = f" (n = {n_matched} matched pairs)" if n_matched else ""
        return (
            f"Propensity score matching ({method}) estimated a {sig} average treatment "
            f"effect on the treated (ATT) of {att:.3f} (SE = {se:.3f}), {_p_str(p)}. "
            f"The sample included {n_treated} treated units{match_text}."
        )

    @staticmethod
    def scm(ate: float, rmspe_ratio: float, perm_p: Optional[float]) -> str:
        sig = _sig_word(perm_p)
        return (
            f"A synthetic control method analysis estimated an average treatment effect of "
            f"{ate:.3f}. The pre-to-post RMSPE ratio was {rmspe_ratio:.3f}, "
            f"with a permutation-based p-value of {_p_str(perm_p)} ({sig})."
        )

    @staticmethod
    def meta_analysis(es: float, ci: List[float], q: float, i2: float,
                      n_studies: int) -> str:
        ci_text = _ci_str(ci)
        het = "low" if i2 < 25 else "moderate" if i2 < 75 else "high"
        return (
            f"A random-effects meta-analysis of {n_studies} studies yielded a pooled effect "
            f"size of {es:.3f} ({ci_text}). Heterogeneity was {het} (I-squared = {i2:.1f}%, "
            f"Q = {q:.2f})."
        )

    @staticmethod
    def krippendorff(alpha: float, n_coders: int, n_units: int,
                     level: str = "nominal") -> str:
        if alpha is None:
            return "Krippendorff's alpha could not be computed."
        interp = ""
        if alpha < 0:
            interp = "indicating unreliable agreement"
        elif alpha < 0.67:
            interp = "indicating questionable agreement"
        elif alpha < 0.80:
            interp = "indicating tentative agreement"
        else:
            interp = "indicating substantial agreement"
        return (
            f"Krippendorff's alpha ({level}) for {n_coders} coders and {n_units} units was "
            f"alpha = {alpha:.3f}, {interp}."
        )

    @staticmethod
    def its(level_change: float, level_se: float, level_p: Optional[float],
            trend_change: float, trend_se: float, trend_p: Optional[float],
            n: int, n_pre: int, n_post: int, method_label: str = "ITS") -> str:
        sig_level = _sig_word(level_p)
        sig_trend = _sig_word(trend_p)
        return (
            f"An interrupted time series analysis ({method_label}) with {n} observations "
            f"({n_pre} pre-intervention, {n_post} post-intervention) revealed a "
            f"{sig_level} level change of {level_change:.3f} (SE = {level_se:.3f}, {_p_str(level_p)}) "
            f"and a {sig_trend} trend change of {trend_change:.3f} (SE = {trend_se:.3f}, {_p_str(trend_p)})."
        )

    @staticmethod
    def kruskal_wallis(h: float, p: Optional[float], df: int,
                       epsilon_squared: float, n_total: int) -> str:
        sig = _sig_word(p)
        return (
            f"A Kruskal-Wallis H test showed a {sig} difference among groups, "
            f"H({df}) = {h:.2f}, {_p_str(p)} (N = {n_total}). "
            f"Effect size epsilon-squared = {epsilon_squared:.3f}."
        )

    @staticmethod
    def friedman(chi2: float, p: Optional[float], n: int, k: int) -> str:
        sig = _sig_word(p)
        return (
            f"A Friedman test showed a {sig} difference among the {k} related groups, "
            f"chi-square({k - 1}, N = {n}) = {chi2:.2f}, {_p_str(p)}."
        )

    @staticmethod
    def bayesian_ttest(BF10: float, t: float, p: Optional[float],
                       df: float, cohen_d: float, interpretation: str) -> str:
        return (
            f"A Bayesian t-test yielded BF10 = {BF10:.2f} ({interpretation}), "
            f"t({df:.0f}) = {t:.2f}, {_p_str(p)}, d = {cohen_d:.2f}."
        )

    @staticmethod
    def methods_section(tool_name: str, params: Dict[str, Any],
                        result: Dict[str, Any]) -> str:
        """Generate a Methods paragraph describing the analysis."""
        n = result.get("n", result.get("n_items", "N/A"))
        return (
            f"We conducted a {tool_name.replace('_', ' ')} analysis on a sample of "
            f"{n} observations using the Sophia research engine."
        )
