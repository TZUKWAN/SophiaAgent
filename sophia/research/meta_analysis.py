"""Meta-analysis engine.

Pure-computation engine for meta-analytic methods.
All public methods accept ``args: dict`` (from tool dispatch) and return
``str`` (JSON).  Optional dependencies are handled gracefully.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import numpy as np

from sophia.research._input import resolve_parent_ids

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(result: dict) -> str:
    """Serialize *result* to a JSON string, converting non-serializable types."""

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return _convert(obj.tolist())
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    return json.dumps(_convert(result), ensure_ascii=False)


def _coerce_list(data: Any) -> Optional[np.ndarray]:
    """Coerce *data* to a 1-D numpy float64 array, dropping NaNs."""
    if data is None:
        return None
    try:
        arr = np.asarray(data, dtype=np.float64).ravel()
        arr = arr[~np.isnan(arr)]
        return arr
    except (TypeError, ValueError):
        return None


# ===========================================================================
# MetaAnalysisEngine
# ===========================================================================

class MetaAnalysisEngine:
    """Meta-analysis methodology engine.

    Every public method:

    1. Accepts ``args: dict`` (tool-dispatch payload).
    2. Validates inputs.
    3. Runs the real computation.
    4. Returns a JSON string with full results.
    """

    def __init__(self, store=None, guard=None):
        self.store = store
        self.guard = guard

    # -----------------------------------------------------------------------
    # ResultStore plumbing
    # -----------------------------------------------------------------------

    def _sanitize_params(self, args: dict) -> dict:
        """Replace bulky arrays (effects, variances, study lists) with summaries."""
        clean: Dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, list):
                if len(v) > 80:
                    clean[k] = f"<list len={len(v)}>"
                elif v and isinstance(v[0], (list, tuple)):
                    total = sum(len(row) if hasattr(row, "__len__") else 1 for row in v)
                    if total > 200:
                        clean[k] = f"<nested list outer={len(v)} total={total}>"
                    else:
                        clean[k] = v
                else:
                    clean[k] = v
            elif isinstance(v, dict):
                total = sum(len(x) if hasattr(x, "__len__") else 1 for x in v.values())
                if total > 200 or len(v) > 50:
                    clean[k] = f"<dict keys={len(v)} total={total}>"
                else:
                    clean[k] = v
            else:
                clean[k] = v
        return clean

    def _final(self, args: dict, result: dict, tool_name: str) -> str:
        """Persist a successful result to the store and embed result_id."""
        if "error" in result:
            return _json(result)
        if self.store is None:
            return _json(result)
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
        return _json(result)

    # -----------------------------------------------------------------------
    # Fixed-effect meta-analysis
    # -----------------------------------------------------------------------

    def fixed_effect(self, args: dict) -> str:
        """Fixed-effect meta-analysis.

        Args:
            effects: list of float (effect sizes).
            variances: list of float (variances of effect sizes).
            study_names: list of str (optional).
            effect_label: str (default 'Effect Size').

        Returns JSON with: pooled_effect, se, ci_low, ci_high, z, p, Q, weights.
        """
        effects_raw = args.get("effects")
        variances_raw = args.get("variances")
        study_names = args.get("study_names")
        effect_label = args.get("effect_label", "Effect Size")

        effects = _coerce_list(effects_raw)
        variances = _coerce_list(variances_raw)

        if effects is None or len(effects) < 1:
            return _json({"error": "effects must be a non-empty list of numbers."})
        if variances is None or len(variances) < 1:
            return _json({"error": "variances must be a non-empty list of numbers."})
        if len(effects) != len(variances):
            return _json({
                "error": f"effects ({len(effects)}) and variances ({len(variances)}) "
                         f"must have the same length."
            })
        if np.any(variances <= 0):
            return _json({"error": "All variances must be positive."})

        k = len(effects)
        if study_names is None:
            study_names = [f"study_{i + 1}" for i in range(k)]
        elif len(study_names) != k:
            return _json({"error": "study_names count does not match effects count."})

        # Fixed-effect weights: w_i = 1 / variance_i
        weights = 1.0 / variances

        # Pooled effect: sum(w_i * y_i) / sum(w_i)
        w_sum = np.sum(weights)
        pooled = np.sum(weights * effects) / w_sum

        # Standard error
        se = 1.0 / math.sqrt(w_sum)

        # Confidence interval (95%)
        z_crit = 1.96
        if HAS_SCIPY:
            z_crit = float(sp_stats.norm.ppf(0.975))
        ci_low = pooled - z_crit * se
        ci_high = pooled + z_crit * se

        # Z-test
        z_val = pooled / se if se > 0 else 0.0
        if HAS_SCIPY:
            p_val = float(2 * (1 - sp_stats.norm.cdf(abs(z_val))))
        else:
            # Approximate p-value
            p_val = float(2 * math.exp(-0.717 * z_val - 0.416 * z_val ** 2)) if abs(z_val) < 10 else 0.0

        # Cochran's Q
        Q = float(np.sum(weights * (effects - pooled) ** 2))

        # Forest plot data
        study_results = []
        for i in range(k):
            se_i = math.sqrt(variances[i])
            ci_i_low = effects[i] - z_crit * se_i
            ci_i_high = effects[i] + z_crit * se_i
            study_results.append({
                "study": study_names[i],
                "effect": float(effects[i]),
                "se": float(se_i),
                "ci_low": float(ci_i_low),
                "ci_high": float(ci_i_high),
                "weight_pct": float(weights[i] / w_sum * 100),
            })

        result = {
            "model": "fixed-effect",
            "effect_label": effect_label,
            "k": k,
            "pooled_effect": float(pooled),
            "se": float(se),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "z": float(z_val),
            "p": float(p_val),
            "Q": Q,
            "weights": {
                study_names[i]: float(weights[i]) for i in range(k)
            },
            "studies": study_results,
        }
        try:
            from sophia.research.apa import APAFormatter
            result["apa"] = APAFormatter.meta_analysis(
                es=result["pooled_effect"], ci=[result["ci_low"], result["ci_high"]],
                q=result["Q"], i2=0.0, n_studies=result["k"]
            )
        except Exception:
            pass
        return self._final(args, result, "research_fixed_effect")

    # -----------------------------------------------------------------------
    # Random-effects meta-analysis (DerSimonian-Laird)
    # -----------------------------------------------------------------------

    def random_effect(self, args: dict) -> str:
        """Random-effects meta-analysis (DerSimonian-Laird).

        Args:
            effects: list of float.
            variances: list of float.
            study_names: list of str (optional).
            effect_label: str (default 'Effect Size').

        Returns JSON with: pooled_effect, se, ci_low, ci_high, z, p,
        tau2, Q, I2, weights.
        """
        effects_raw = args.get("effects")
        variances_raw = args.get("variances")
        study_names = args.get("study_names")
        effect_label = args.get("effect_label", "Effect Size")

        effects = _coerce_list(effects_raw)
        variances = _coerce_list(variances_raw)

        if effects is None or len(effects) < 1:
            return _json({"error": "effects must be a non-empty list of numbers."})
        if variances is None or len(variances) < 1:
            return _json({"error": "variances must be a non-empty list of numbers."})
        if len(effects) != len(variances):
            return _json({
                "error": f"effects ({len(effects)}) and variances ({len(variances)}) "
                         f"must have the same length."
            })
        if np.any(variances <= 0):
            return _json({"error": "All variances must be positive."})

        k = len(effects)
        if study_names is None:
            study_names = [f"study_{i + 1}" for i in range(k)]
        elif len(study_names) != k:
            return _json({"error": "study_names count does not match effects count."})

        # Step 1: Fixed-effect weights to compute Q
        w_fe = 1.0 / variances
        w_sum = np.sum(w_fe)
        pooled_fe = np.sum(w_fe * effects) / w_sum

        # Cochran's Q
        Q = float(np.sum(w_fe * (effects - pooled_fe) ** 2))
        df = k - 1

        # DerSimonian-Laird tau^2
        C = w_sum - np.sum(w_fe ** 2) / w_sum
        if C > 0 and Q > df:
            tau2 = float((Q - df) / C)
        else:
            tau2 = 0.0

        # Step 2: Random-effects weights
        w_re = 1.0 / (variances + tau2)
        w_re_sum = np.sum(w_re)
        pooled_re = np.sum(w_re * effects) / w_re_sum

        # Standard error
        se_re = 1.0 / math.sqrt(w_re_sum)

        # Confidence interval (95%)
        z_crit = 1.96
        if HAS_SCIPY:
            z_crit = float(sp_stats.norm.ppf(0.975))
        ci_low = pooled_re - z_crit * se_re
        ci_high = pooled_re + z_crit * se_re

        # Z-test
        z_val = pooled_re / se_re if se_re > 0 else 0.0
        if HAS_SCIPY:
            p_val = float(2 * (1 - sp_stats.norm.cdf(abs(z_val))))
        else:
            p_val = None

        # I^2
        I2 = float(max(0, (Q - df) / Q * 100)) if Q > 0 else 0.0

        # H statistic
        H = float(math.sqrt(Q / df)) if df > 0 and Q > 0 else 1.0

        # Forest plot data
        study_results = []
        for i in range(k):
            se_i = math.sqrt(variances[i])
            ci_i_low = effects[i] - z_crit * se_i
            ci_i_high = effects[i] + z_crit * se_i
            study_results.append({
                "study": study_names[i],
                "effect": float(effects[i]),
                "se": float(se_i),
                "ci_low": float(ci_i_low),
                "ci_high": float(ci_i_high),
                "weight_pct": float(w_re[i] / w_re_sum * 100),
            })

        result = {
            "model": "random-effects (DerSimonian-Laird)",
            "effect_label": effect_label,
            "k": k,
            "pooled_effect": float(pooled_re),
            "se": float(se_re),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "z": float(z_val),
            "p": float(p_val),
            "tau2": tau2,
            "Q": Q,
            "df": df,
            "I2": I2,
            "H": H,
            "weights": {
                study_names[i]: float(w_re[i]) for i in range(k)
            },
            "studies": study_results,
        }
        try:
            from sophia.research.apa import APAFormatter
            result["apa"] = APAFormatter.meta_analysis(
                es=result["pooled_effect"], ci=[result["ci_low"], result["ci_high"]],
                q=result["Q"], i2=result["I2"], n_studies=result["k"]
            )
        except Exception:
            pass
        return self._final(args, result, "research_random_effect")

    # -----------------------------------------------------------------------
    # Heterogeneity statistics
    # -----------------------------------------------------------------------

    def heterogeneity(self, args: dict) -> str:
        """Heterogeneity statistics.

        Args:
            effects: list of float.
            variances: list of float.

        Returns JSON with: Q, df, p_Q, I2, tau2, H, interpretation.
        """
        effects_raw = args.get("effects")
        variances_raw = args.get("variances")

        effects = _coerce_list(effects_raw)
        variances = _coerce_list(variances_raw)

        if effects is None or len(effects) < 2:
            return _json({"error": "At least 2 effects are required."})
        if variances is None or len(variances) < 2:
            return _json({"error": "At least 2 variances are required."})
        if len(effects) != len(variances):
            return _json({"error": "effects and variances must have the same length."})
        if np.any(variances <= 0):
            return _json({"error": "All variances must be positive."})

        k = len(effects)
        w = 1.0 / variances
        w_sum = np.sum(w)
        pooled = np.sum(w * effects) / w_sum

        # Q statistic
        Q = float(np.sum(w * (effects - pooled) ** 2))
        df = k - 1

        # p-value for Q
        if HAS_SCIPY:
            p_Q = float(1 - sp_stats.chi2.cdf(Q, df)) if df > 0 else 1.0
        else:
            # Approximate p-value using normal approximation of chi-squared
            if df > 0:
                x = (Q / df) ** (1.0 / 3.0)
                mu = 1 - 2.0 / (9 * df)
                sigma = math.sqrt(2.0 / (9 * df))
                z_q = (x - mu) / sigma if sigma > 0 else 0.0
                # Approximate upper tail
                p_Q = float(max(0, min(1, 1 - 0.5 * (1 + math.erf(z_q / math.sqrt(2))))))
            else:
                p_Q = 1.0

        # I^2
        I2 = float(max(0, (Q - df) / Q * 100)) if Q > 0 else 0.0

        # tau^2 (DerSimonian-Laird)
        C = w_sum - np.sum(w ** 2) / w_sum
        if C > 0 and Q > df:
            tau2 = float((Q - df) / C)
        else:
            tau2 = 0.0

        # H statistic
        H = float(math.sqrt(Q / df)) if df > 0 and Q > 0 else 1.0

        # Interpretation
        if I2 < 25:
            interpretation = "Low heterogeneity"
        elif I2 < 75:
            interpretation = "Moderate heterogeneity"
        else:
            interpretation = "High heterogeneity"

        result = {
            "Q": Q,
            "df": df,
            "p_Q": p_Q,
            "I2": I2,
            "tau2": tau2,
            "H": H,
            "interpretation": interpretation,
            "k": k,
        }
        try:
            result["apa"] = (
                f"Heterogeneity statistics: Q = {Q:.2f} (df = {df}, p = {p_Q:.3f}), "
                f"I-squared = {I2:.1f}%, tau-squared = {tau2:.3f} ({interpretation})."
            )
        except Exception:
            pass
        return self._final(args, result, "research_heterogeneity")

    # -----------------------------------------------------------------------
    # Publication bias tests
    # -----------------------------------------------------------------------

    def bias_test(self, args: dict) -> str:
        """Publication bias tests.

        Args:
            effects: list of float.
            variances: list of float.
            test: str ('egger'|'begg'|'fail_safe', default 'egger').

        Returns JSON with: test statistic, p-value, interpretation.
        """
        effects_raw = args.get("effects")
        variances_raw = args.get("variances")
        test_type = args.get("test", "egger")

        effects = _coerce_list(effects_raw)
        variances = _coerce_list(variances_raw)

        if effects is None or len(effects) < 3:
            return _json({"error": "At least 3 effects are required for bias testing."})
        if variances is None or len(variances) < 3:
            return _json({"error": "At least 3 variances are required for bias testing."})
        if len(effects) != len(variances):
            return _json({"error": "effects and variances must have the same length."})
        if np.any(variances <= 0):
            return _json({"error": "All variances must be positive."})

        k = len(effects)

        if test_type == "egger":
            inner = self._egger_test(effects, variances, k)
        elif test_type == "begg":
            inner = self._begg_test(effects, variances, k)
        elif test_type == "fail_safe":
            inner = self._fail_safe_n(effects, variances, k)
        else:
            return _json({"error": f"Unknown test '{test_type}'. Use egger, begg, or fail_safe."})
        parsed = json.loads(inner)
        return self._final(args, parsed, "research_bias_test")

    def _egger_test(self, effects: np.ndarray, variances: np.ndarray, k: int) -> str:
        """Egger's test: regression of standardized effect on precision.

        Standardized effect = effect / se
        Precision = 1 / se
        If intercept != 0, suggests publication bias.
        """
        se = np.sqrt(variances)
        precision = 1.0 / se  # 1/SE (predictor)
        standardized = effects / se  # effect/SE (response)

        # Linear regression: standardized = intercept + slope * precision
        # Using OLS: y = a + b*x
        n = k
        x = precision
        y = standardized

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))
        ss_xx = np.sum((x - x_mean) ** 2)

        if ss_xx == 0:
            return _json({"error": "Cannot compute Egger's test: zero variance in precision."})

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean

        # Residuals and standard error of intercept
        y_pred = intercept + slope * x
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        mse = ss_res / (n - 2) if n > 2 else 0.0

        # Standard error of intercept
        se_intercept = math.sqrt(mse * (1.0 / n + x_mean ** 2 / ss_xx)) if mse > 0 else 0.0

        # t-test for intercept
        if se_intercept > 0:
            t_stat = intercept / se_intercept
            df = n - 2
            if HAS_SCIPY:
                p_val = float(2 * sp_stats.t.sf(abs(t_stat), df))
            else:
                # Approximate
                p_val = None
        else:
            t_stat = 0.0
            p_val = 1.0

        # Interpretation
        if p_val < 0.05:
            interpretation = "Significant asymmetry detected (p < 0.05), suggesting possible publication bias."
        else:
            interpretation = "No significant asymmetry detected (p >= 0.05), no strong evidence of publication bias."

        return _json({
            "test": "Egger's test",
            "intercept": float(intercept),
            "se_intercept": float(se_intercept),
            "t_statistic": float(t_stat),
            "df": int(n - 2),
            "p_value": float(p_val),
            "slope": float(slope),
            "interpretation": interpretation,
            "k": k,
        })

    def _begg_test(self, effects: np.ndarray, variances: np.ndarray, k: int) -> str:
        """Begg's rank correlation test.

        Rank correlation between standardized effect and variance (or precision).
        """
        se = np.sqrt(variances)

        # Standardized effects (centered by pooled effect)
        w = 1.0 / variances
        pooled = np.sum(w * effects) / np.sum(w)
        centered = effects - pooled

        # Standardized by SE
        standardized = centered / se

        # Precision
        precision = 1.0 / se

        # Rank correlation (Kendall's tau)
        if HAS_SCIPY:
            tau, p_val = sp_stats.kendalltau(standardized, precision)
        else:
            # Manual Kendall's tau
            n = k
            concordant = 0
            discordant = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = standardized[i] - standardized[j]
                    dy = precision[i] - precision[j]
                    prod = dx * dy
                    if prod > 0:
                        concordant += 1
                    elif prod < 0:
                        discordant += 1
            total = concordant + discordant
            tau = (concordant - discordant) / total if total > 0 else 0.0
            # Approximate p-value
            z = tau * math.sqrt(9 * n * (n - 1) / (2 * (2 * n + 5))) if n > 2 else 0.0
            p_val = 2 * math.exp(-0.717 * abs(z) - 0.416 * z ** 2) if abs(z) < 10 else 0.0

        tau = float(tau) if not math.isnan(float(tau)) else 0.0
        p_val = float(p_val) if not math.isnan(float(p_val)) else 1.0

        if p_val < 0.05:
            interpretation = "Significant rank correlation (p < 0.05), suggesting possible publication bias."
        else:
            interpretation = "No significant rank correlation (p >= 0.05), no strong evidence of publication bias."

        return _json({
            "test": "Begg's rank correlation test",
            "tau": tau,
            "p_value": p_val,
            "interpretation": interpretation,
            "k": k,
        })

    def _fail_safe_n(self, effects: np.ndarray, variances: np.ndarray, k: int) -> str:
        """Fail-safe N (Rosenthal's method).

        Number of null studies needed to bring p > 0.05.
        """
        # Compute sum of z-scores
        w = 1.0 / variances
        pooled = np.sum(w * effects) / np.sum(w)
        se_pooled = 1.0 / math.sqrt(np.sum(w))
        z_sum = float(pooled / se_pooled) if se_pooled > 0 else 0.0

        # Rosenthal's fail-safe N
        alpha = 0.05
        z_alpha = 1.96
        if HAS_SCIPY:
            z_alpha = float(sp_stats.norm.ppf(1 - alpha / 2))

        # N_fs = (sum(z)^2 / z_alpha^2) - k
        N_fs = (z_sum ** 2 / z_alpha ** 2) - k

        # Orwin's fail-safe N (based on effect sizes)
        mean_effect = float(np.mean(np.abs(effects)))
        trivial_effect = mean_effect * 0.5  # trivial = half of observed mean
        if mean_effect > 0 and trivial_effect > 0:
            N_orwin = int(k * (mean_effect - trivial_effect) / trivial_effect)
        else:
            N_orwin = 0

        # Interpretation
        N_fs_int = max(0, int(N_fs))
        if N_fs_int > 5 * k + 10:
            interpretation = f"Fail-safe N ({N_fs_int}) exceeds tolerance (5k+10={5 * k + 10}). " \
                             "Results appear robust to publication bias."
        else:
            interpretation = f"Fail-safe N ({N_fs_int}) does not exceed tolerance (5k+10={5 * k + 10}). " \
                             "Results may be vulnerable to publication bias."

        return _json({
            "test": "Fail-safe N (Rosenthal)",
            "z_sum": float(z_sum),
            "N_fail_safe": N_fs_int,
            "N_orwin": N_orwin,
            "tolerance_threshold": 5 * k + 10,
            "interpretation": interpretation,
            "k": k,
        })

    # -----------------------------------------------------------------------
    # Subgroup analysis
    # -----------------------------------------------------------------------

    def subgroup(self, args: dict) -> str:
        """Subgroup analysis.

        Args:
            effects: list of float.
            variances: list of float.
            subgroups: list of str (group label for each study).
            study_names: list of str (optional).

        Returns JSON with: subgroup results (pooled effect per group),
        Q_between, Q_within, test for subgroup differences.
        """
        effects_raw = args.get("effects")
        variances_raw = args.get("variances")
        subgroups_raw = args.get("subgroups")
        study_names = args.get("study_names")

        effects = _coerce_list(effects_raw)
        variances = _coerce_list(variances_raw)

        if effects is None or len(effects) < 2:
            return _json({"error": "At least 2 effects are required."})
        if variances is None or len(variances) < 2:
            return _json({"error": "At least 2 variances are required."})
        if len(effects) != len(variances):
            return _json({"error": "effects and variances must have the same length."})
        if np.any(variances <= 0):
            return _json({"error": "All variances must be positive."})

        k = len(effects)

        if subgroups_raw is None or len(subgroups_raw) != k:
            return _json({"error": "subgroups must be a list with the same length as effects."})

        subgroups = [str(s) for s in subgroups_raw]

        if study_names is None:
            study_names = [f"study_{i + 1}" for i in range(k)]
        elif len(study_names) != k:
            return _json({"error": "study_names count does not match effects count."})

        # Group studies by subgroup
        unique_groups = sorted(set(subgroups))
        if len(unique_groups) < 2:
            return _json({"error": "At least 2 subgroups are required."})

        group_indices: Dict[str, List[int]] = {g: [] for g in unique_groups}
        for i, g in enumerate(subgroups):
            group_indices[g].append(i)

        # Run fixed-effect meta-analysis within each subgroup
        subgroup_results = []
        Q_within_total = 0.0

        for group in unique_groups:
            idx = group_indices[group]
            g_effects = effects[idx]
            g_vars = variances[idx]

            if len(idx) == 0:
                continue

            w = 1.0 / g_vars
            w_sum = float(np.sum(w))
            pooled = float(np.sum(w * g_effects) / w_sum)
            se = 1.0 / math.sqrt(w_sum)

            z_crit = 1.96
            if HAS_SCIPY:
                z_crit = float(sp_stats.norm.ppf(0.975))

            ci_low = pooled - z_crit * se
            ci_high = pooled + z_crit * se
            z_val = pooled / se if se > 0 else 0.0

            if HAS_SCIPY:
                p_val = float(2 * (1 - sp_stats.norm.cdf(abs(z_val))))
            else:
                p_val = None

            Q_w = float(np.sum(w * (g_effects - pooled) ** 2))
            Q_within_total += Q_w

            subgroup_results.append({
                "subgroup": group,
                "k": len(idx),
                "pooled_effect": pooled,
                "se": float(se),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "z": float(z_val),
                "p": p_val,
                "Q_within": Q_w,
                "studies": [study_names[i] for i in idx],
            })

        # Total Q (overall)
        w_all = 1.0 / variances
        w_all_sum = np.sum(w_all)
        pooled_all = np.sum(w_all * effects) / w_all_sum
        Q_total = float(np.sum(w_all * (effects - pooled_all) ** 2))

        # Q_between = Q_total - Q_within
        Q_between = Q_total - Q_within_total
        df_between = len(unique_groups) - 1
        df_within = k - len(unique_groups)

        # Test for subgroup differences
        if HAS_SCIPY and df_between > 0:
            p_between = float(1 - sp_stats.chi2.cdf(max(0, Q_between), df_between))
        else:
            p_between = None

        # I^2 for subgroup differences
        I2_between = float(max(0, (Q_between - df_between) / Q_between * 100)) if Q_between > 0 and df_between > 0 else 0.0

        # Interpretation
        if p_between is not None and p_between < 0.05:
            interpretation = "Significant difference between subgroups (p < 0.05)."
        elif p_between is not None:
            interpretation = "No significant difference between subgroups (p >= 0.05)."
        else:
            interpretation = "Could not compute test for subgroup differences."

        result = {
            "model": "fixed-effect subgroup analysis",
            "k": k,
            "n_subgroups": len(unique_groups),
            "subgroups": subgroup_results,
            "Q_total": Q_total,
            "Q_within": Q_within_total,
            "Q_between": Q_between,
            "df_between": df_between,
            "df_within": df_within,
            "p_between": p_between,
            "I2_between": I2_between,
            "interpretation": interpretation,
        }
        try:
            result["apa"] = (
                f"Subgroup analysis of {k} studies across {len(unique_groups)} subgroups: "
                f"Q-between = {Q_between:.2f} (df = {df_between}, p = {p_between:.3f}). {interpretation}"
            )
        except Exception:
            pass
        return self._final(args, result, "research_subgroup")
