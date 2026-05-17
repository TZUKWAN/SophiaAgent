"""Statistical testing engine for empirical research.

Pure-computation wrapper around scipy, pingouin, numpy, and statsmodels.
All public methods accept ``args: dict`` (from tool dispatch) and return
``str`` (JSON).  Optional dependencies are handled gracefully.

When constructed with a ``ResultStore`` (P1.4), each method also persists
its result and adds ``result_id`` to the response JSON, enabling tool-to-
tool data flow via lightweight references instead of inlined arrays.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sophia.research._input import (
    InputResolutionError,
    resolve_dataframe,
    resolve_parent_ids,
)
from sophia.research.apa import APAFormatter

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import researchpy as rp
    HAS_RESEARCHPY = True
except ImportError:
    HAS_RESEARCHPY = False

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


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


def _safe_float(value: Any) -> Optional[float]:
    """Convert to float, returning None on failure."""
    if value is None:
        return None
    try:
        # pingouin may return strings like "2.154e+55" or "nan" for BF10
        v = float(str(value))
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _coerce_numeric_list(data: Any) -> Optional[np.ndarray]:
    """Coerce *data* to a 1-D numpy float64 array, dropping NaNs."""
    if data is None:
        return None
    try:
        arr = np.asarray(data, dtype=np.float64).ravel()
        arr = arr[~np.isnan(arr)]
        return arr
    except (TypeError, ValueError):
        return None


def _cohens_d(g1: np.ndarray, g2: np.ndarray, paired: bool = False) -> float:
    """Compute Cohen's *d* effect size."""
    if paired:
        diff = g1 - g2
        return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) != 0 else 0.0
    n1, n2 = len(g1), len(g2)
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled)


def _hedges_g(g1: np.ndarray, g2: np.ndarray, paired: bool = False) -> float:
    """Compute Hedges' *g* effect size."""
    d = _cohens_d(g1, g2, paired=paired)
    n1, n2 = len(g1), len(g2)
    df = n1 + n2 - 2
    if df < 1:
        return d
    correction = 1 - (3 / (4 * df - 1))
    return d * correction


def _cramers_v(table: np.ndarray) -> float:
    """Compute Cramer's V from a contingency table."""
    chi2 = sp_stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, c = table.shape
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((r - 1) * (c - 1)) / (n - 1)) if n > 1 else 0.0
    r_corr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    c_corr = c - ((c - 1) ** 2) / (n - 1) if n > 1 else c
    if min(r_corr - 1, c_corr - 1) <= 0:
        return 0.0
    return float(np.sqrt(phi2_corrected / min(r_corr - 1, c_corr - 1)))


def _eta_squared(f_val: float, df1: int, df2: int) -> float:
    """Compute eta-squared from F and degrees of freedom."""
    return (f_val * df1) / (f_val * df1 + df2) if (f_val * df1 + df2) != 0 else 0.0


# ===========================================================================
# StatisticalEngine
# ===========================================================================

class StatisticalEngine:
    """Wraps scipy / pingouin / statsmodels for statistical hypothesis testing.

    Every public method:

    1. Accepts ``args: dict`` (tool-dispatch payload).
    2. Validates inputs.
    3. Runs the real computation.
    4. Returns a JSON string with full results.

    When a ``store`` (ResultStore) is configured, each call also:

    - Resolves DataFrame-shaped inputs (``result_id`` / ``path`` / dict-of-cols).
    - Supports column-name selectors (``*_col`` args) on top of legacy lists.
    - Persists the result and embeds ``result_id`` in the response, recording
      lineage back to any upstream result_ids referenced in args.
    """

    def __init__(self, store: Optional[Any] = None, guard: Optional[Any] = None):
        self.store = store
        self.guard = guard

    # -----------------------------------------------------------------------
    # ResultStore / input resolution helpers
    # -----------------------------------------------------------------------

    def _resolve_input_df(self, args: dict) -> Optional[pd.DataFrame]:
        """Try to resolve a DataFrame from args.

        Returns None when args provides no DataFrame-shaped source. We
        deliberately do NOT treat plain ``data=[1,2,3]`` (a flat numeric list)
        as a DataFrame to avoid coercing legacy scalar inputs into single-
        column frames.
        """
        if args.get("result_id") or args.get("path"):
            try:
                return resolve_dataframe(args, store=self.store, guard=self.guard)
            except InputResolutionError:
                return None
        data = args.get("data")
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return pd.DataFrame(data)
        if isinstance(data, dict) and data:
            try:
                first_val = next(iter(data.values()))
                if isinstance(first_val, (list, tuple, np.ndarray, pd.Series)):
                    return pd.DataFrame(data)
            except StopIteration:
                pass
        return None

    def _column(self, df: pd.DataFrame, name: str) -> np.ndarray:
        if name not in df.columns:
            raise InputResolutionError(
                f"Column '{name}' not in DataFrame; available: {list(df.columns)}"
            )
        return df[name].dropna().to_numpy(dtype=np.float64)

    def _normalize_args(self, args: dict) -> dict:
        """Apply column-name resolution to common arg shapes.

        Supported substitutions (when a DataFrame can be resolved):

        - ``data_col``     -> ``data``
        - ``group1_col``   -> ``group1``
        - ``group2_col``   -> ``group2``
        - ``x_col``        -> ``x``
        - ``y_col``        -> ``y``
        - ``x_cols``       -> ``X``  (also populates ``x_names`` if absent)
        - ``value_col`` + ``group_col`` -> ``groups`` (long-format expansion)

        For regression, all columns share a single NA-drop mask so y and X
        rows stay aligned. For other methods we drop NAs per column.
        """
        df = self._resolve_input_df(args)
        if df is None:
            return args
        new_args = dict(args)

        # Long-format expansion: value_col + group_col -> groups (list of lists)
        value_col = args.get("value_col")
        group_col = args.get("group_col")
        if value_col and group_col and value_col in df.columns and group_col in df.columns:
            sub = df[[value_col, group_col]].dropna()
            groups: List[List[float]] = []
            labels: List[str] = []
            for label, grp_df in sub.groupby(group_col, sort=False):
                vals = pd.to_numeric(grp_df[value_col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                groups.append(vals.tolist())
                labels.append(str(label))
            if groups:
                new_args["groups"] = groups
                new_args.setdefault("data", groups)
                new_args.setdefault("groups_labels", labels)

        # Regression-specific: shared NA mask for y + X
        x_cols = args.get("x_cols")
        y_col = args.get("y_col")
        if x_cols and isinstance(x_cols, list) and y_col and y_col in df.columns:
            valid_x_cols = [c for c in x_cols if c in df.columns]
            if valid_x_cols:
                subset = df[[y_col] + valid_x_cols].dropna()
                new_args["y"] = pd.to_numeric(subset[y_col], errors="coerce").dropna().tolist()
                new_args["X"] = [
                    pd.to_numeric(subset[c], errors="coerce").dropna().tolist()
                    for c in valid_x_cols
                ]
                new_args.setdefault("x_names", valid_x_cols)
                new_args.setdefault("y_name", y_col)
        else:
            # Independent column substitutions (drops NAs per-column)
            simple_map = {
                "data_col": "data",
                "group1_col": "group1",
                "group2_col": "group2",
                "x_col": "x",
                "y_col": "y",
            }
            for col_key, target in simple_map.items():
                col_name = args.get(col_key)
                if col_name and col_name in df.columns and target not in args:
                    new_args[target] = self._column(df, col_name).tolist()

        return new_args

    def _sanitize_params(self, args: dict) -> dict:
        out: Dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, np.ndarray):
                out[k] = f"<ndarray shape={list(v.shape)}>"
            elif isinstance(v, pd.DataFrame):
                out[k] = f"<DataFrame shape={list(v.shape)}>"
            elif isinstance(v, pd.Series):
                out[k] = f"<Series len={len(v)}>"
            elif isinstance(v, list):
                if len(v) > 80:
                    out[k] = f"<list len={len(v)}>"
                elif v and isinstance(v[0], (list, tuple, np.ndarray)):
                    total = 0
                    try:
                        total = sum(len(sub) for sub in v)
                    except TypeError:
                        total = len(v)
                    if total > 80:
                        out[k] = f"<nested groups={len(v)} total={total}>"
                    else:
                        out[k] = v
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    def _final(self, args: dict, result: dict, tool_name: str) -> str:
        """Strip numpy types from *result*, optionally persist it, return JSON.

        Stores only successful results (no ``error`` key). Records lineage
        from any ``res_*`` references in args via ``resolve_parent_ids``.
        Failures in the store path are swallowed — they must not block the
        scientific output.
        """
        try:
            clean = json.loads(_json(result))
        except (TypeError, ValueError):
            clean = result
        if (
            self.store is not None
            and isinstance(clean, dict)
            and "error" not in clean
        ):
            try:
                parents = resolve_parent_ids(args)
                params = self._sanitize_params(args)
                rid = self.store.store(
                    clean,
                    kind="result",
                    tool=tool_name,
                    params=params,
                    parents=parents,
                )
                clean["result_id"] = rid
            except Exception:
                pass
        return json.dumps(clean, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Descriptive statistics
    # -----------------------------------------------------------------------

    def describe(self, args: dict) -> str:
        """Descriptive statistics.

        Args:
            data: list of numbers, or pass ``result_id``/``path`` + ``data_col``.

        Returns JSON with: n, mean, std, min, q1, median, q3, max,
        skew, kurtosis, se, ci_95.
        """
        args = self._normalize_args(args)
        raw = args.get("data")
        arr = _coerce_numeric_list(raw)
        if arr is None or len(arr) == 0:
            return _json({"error": "No valid numeric data provided."})

        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        se = std / math.sqrt(n) if n > 1 else 0.0
        q1 = float(np.percentile(arr, 25))
        median = float(np.percentile(arr, 50))
        q3 = float(np.percentile(arr, 75))
        ci_95 = None
        if n > 1 and HAS_SCIPY:
            ci = sp_stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
            ci_95 = [float(ci[0]), float(ci[1])]

        # skew & kurtosis
        if HAS_SCIPY:
            skew = float(sp_stats.skew(arr, bias=False))
            kurtosis = float(sp_stats.kurtosis(arr, bias=False))
        else:
            # Manual fallback
            if std == 0:
                skew = 0.0
                kurtosis = 0.0
            else:
                z = (arr - mean) / std
                skew = float(np.mean(z ** 3)) if n > 2 else 0.0
                kurtosis = float(np.mean(z ** 4) - 3) if n > 3 else 0.0

        return self._final(args, {
            "n": n,
            "mean": mean,
            "std": std,
            "min": float(np.min(arr)),
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": float(np.max(arr)),
            "skew": skew,
            "kurtosis": kurtosis,
            "se": se,
            "ci_95": ci_95,
        }, "research_describe")

    # -----------------------------------------------------------------------
    # T-tests
    # -----------------------------------------------------------------------

    def ttest(self, args: dict) -> str:
        """T-test.

        Args:
            group1: list of numbers (or ``group1_col`` against a DataFrame).
            group2: list of numbers (or ``group2_col``).
            paired: bool  (default False)
            welch:  bool  (default False – only relevant for independent)
            popmean: float (one-sample test target, optional)

        Returns JSON with: t, p, df, cohen_d, ci, bf10 (if pingouin).
        """
        args = self._normalize_args(args)
        g1_raw = args.get("group1")
        g2_raw = args.get("group2")
        paired = bool(args.get("paired", False))
        welch = bool(args.get("welch", False))
        popmean = args.get("popmean")

        g1 = _coerce_numeric_list(g1_raw)
        if g1 is None or len(g1) == 0:
            return _json({"error": "group1 must be a non-empty list of numbers."})

        # ---- one-sample (no group2, popmean given) ----
        if g2_raw is None and popmean is not None:
            pm = float(popmean)
            if HAS_PINGOUIN:
                df_pg = pd.DataFrame({"vals": g1})
                res = pg.ttest(df_pg["vals"], pm).round(6)
                row = res.iloc[0]
                result = {
                    "test": "one-sample t-test",
                    "t": _safe_float(row.get("T")),
                    "p": _safe_float(row.get("p_val")),
                    "df": _safe_float(row.get("dof")),
                    "cohen_d": _safe_float(row.get("cohen_d")),
                    "ci": None,
                    "bf10": _safe_float(row.get("BF10")),
                }
                try:
                    result["apa"] = APAFormatter.t_test(
                        t=result["t"], df=result["df"], p=result["p"],
                        d=result["cohen_d"], mean_diff=0.0, ci=[]
                    )
                except Exception:
                    pass
                return self._final(args, result, "research_ttest")
            if HAS_SCIPY:
                t_val, p_val = sp_stats.ttest_1samp(g1, pm)
                cd = _cohens_d(g1, np.full_like(g1, pm), paired=False)
                return self._final(args, {
                    "test": "one-sample t-test",
                    "t": float(t_val),
                    "p": float(p_val),
                    "df": int(len(g1) - 1),
                    "cohen_d": cd,
                }, "research_ttest")
            return _json({"error": "Neither pingouin nor scipy is available."})

        # ---- two-sample ----
        g2 = _coerce_numeric_list(g2_raw)
        if g2 is None or len(g2) == 0:
            return _json({"error": "group2 must be a non-empty list of numbers."})

        if paired and len(g1) != len(g2):
            return _json({"error": "For paired t-test, group1 and group2 must have equal length."})

        # Use pingouin when available
        if HAS_PINGOUIN:
            if paired:
                res = pg.ttest(g1, g2, paired=True).round(6)
                test_name = "paired t-test"
            elif welch:
                res = pg.ttest(g1, g2, correction=True).round(6)
                test_name = "Welch t-test"
            else:
                res = pg.ttest(g1, g2, correction=False).round(6)
                test_name = "independent t-test"
            row = res.iloc[0]
            result = {
                "test": test_name,
                "t": _safe_float(row.get("T")),
                "p": _safe_float(row.get("p_val")),
                "df": _safe_float(row.get("dof")),
                "cohen_d": _safe_float(row.get("cohen_d")),
                "ci": None,
                "bf10": _safe_float(row.get("BF10")),
            }
            try:
                result["apa"] = APAFormatter.t_test(
                    t=result["t"], df=result["df"], p=result["p"],
                    d=result["cohen_d"], mean_diff=0.0, ci=[]
                )
            except Exception:
                pass
            return self._final(args, result, "research_ttest")

        # Fallback to scipy
        if not HAS_SCIPY:
            return _json({"error": "Neither pingouin nor scipy is available."})

        if paired:
            t_val, p_val = sp_stats.ttest_rel(g1, g2)
            test_name = "paired t-test"
        elif welch:
            t_val, p_val = sp_stats.ttest_ind(g1, g2, equal_var=False)
            test_name = "Welch t-test"
        else:
            t_val, p_val = sp_stats.ttest_ind(g1, g2, equal_var=True)
            test_name = "independent t-test"

        cd = _cohens_d(g1, g2, paired=paired)
        # degrees of freedom
        if paired:
            df_val = len(g1) - 1
        elif welch:
            n1, n2 = len(g1), len(g2)
            s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
            denom = (s1 / n1 + s2 / n2) ** 2
            num = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
            df_val = denom / num if num != 0 else 0
        else:
            df_val = len(g1) + len(g2) - 2

        mean_diff = float(np.mean(g1) - np.mean(g2))
        result = {
            "test": test_name,
            "t": float(t_val),
            "p": float(p_val),
            "df": _safe_float(df_val),
            "cohen_d": cd,
        }
        try:
            result["apa"] = APAFormatter.t_test(
                t=float(t_val), df=float(df_val), p=float(p_val),
                d=cd, mean_diff=mean_diff, ci=[]
            )
        except Exception:
            pass
        return self._final(args, result, "research_ttest")

    # -----------------------------------------------------------------------
    # ANOVA
    # -----------------------------------------------------------------------

    def anova(self, args: dict) -> str:
        """ANOVA.

        Args:
            data: list of lists (each sub-list = one group's values), OR
                  provide ``value_col`` + ``group_col`` with a DataFrame source.
            groups: list of str labels (optional).
            repeated: bool (default False – repeated-measures).
            type: str ('one-way' | 'rm' | 'welch' | default 'one-way').

        Returns JSON with: F, p, np2 (eta-squared), source, df1, df2.
        """
        args = self._normalize_args(args)
        raw = args.get("data")
        repeated = bool(args.get("repeated", False))
        anova_type = args.get("type", "one-way")
        # ``groups`` may legitimately be (a) a list of str labels (legacy
        # caller intent) or (b) a list of lists / arrays when the engine
        # was normalized from long-format args. Only treat list-of-str as
        # labels — otherwise fall back to ``groups_labels``.
        groups_arg = args.get("groups")
        labels_arg = args.get("groups_labels")
        if isinstance(groups_arg, list) and groups_arg and all(
            isinstance(x, str) for x in groups_arg
        ):
            group_labels = groups_arg
        elif isinstance(labels_arg, list) and labels_arg and all(
            isinstance(x, str) for x in labels_arg
        ):
            group_labels = labels_arg
        else:
            group_labels = None

        if not isinstance(raw, list) or len(raw) < 2:
            return _json({"error": "data must be a list of at least 2 groups."})

        groups_data = []
        for i, g in enumerate(raw):
            arr = _coerce_numeric_list(g)
            if arr is None or len(arr) < 2:
                return _json({"error": f"Group {i} has fewer than 2 valid values."})
            groups_data.append(arr)

        k = len(groups_data)
        if group_labels is None:
            group_labels = [f"group_{i}" for i in range(k)]
        elif len(group_labels) != k:
            return _json({"error": "groups label count does not match data group count."})

        # ---- pingouin path ----
        if HAS_PINGOUIN:
            if repeated:
                # repeated-measures ANOVA: all groups must have same length
                lengths = [len(g) for g in groups_data]
                if len(set(lengths)) != 1:
                    return _json({"error": "Repeated-measures ANOVA requires equal group sizes."})
                n_subjects = lengths[0]
                rows = []
                for subj_idx in range(n_subjects):
                    for grp_idx, grp in enumerate(groups_data):
                        rows.append({
                            "subject": f"s{subj_idx}",
                            "group": group_labels[grp_idx],
                            "value": float(grp[subj_idx]),
                        })
                df_long = pd.DataFrame(rows)
                res = pg.rm_anova(data=df_long, dv="value", within="group",
                                  subject="subject").round(6)
                row = res.iloc[0]
                f_val = _safe_float(row.get("F"))
                p_val = _safe_float(row.get("p_unc"))
                np2 = _safe_float(row.get("np2") or row.get("ng2"))
                df1 = _safe_float(row.get("ddof1"))
                df2 = _safe_float(row.get("ddof2"))
                result = {
                    "test": "repeated-measures ANOVA",
                    "F": f_val,
                    "p": p_val,
                    "np2": np2,
                    "df1": df1,
                    "df2": df2,
                    "source": "group",
                }
                try:
                    result["apa"] = APAFormatter.anova(
                        f=result["F"], df1=float(result["df1"] or 0),
                        df2=float(result["df2"] or 0), p=result["p"],
                        eta_sq=result["np2"] or 0
                    )
                except Exception:
                    pass
                return self._final(args, result, "research_anova")

            if anova_type == "welch":
                # Welch ANOVA via pingouin
                rows = []
                for grp_idx, grp in enumerate(groups_data):
                    for val in grp:
                        rows.append({"group": group_labels[grp_idx], "value": float(val)})
                df_long = pd.DataFrame(rows)
                res = pg.welch_anova(data=df_long, dv="value", between="group").round(6)
                row = res.iloc[0]
                result = {
                    "test": "Welch ANOVA",
                    "F": _safe_float(row.get("F")),
                    "p": _safe_float(row.get("p_unc")),
                    "np2": None,
                    "df1": _safe_float(row.get("ddof1")),
                    "df2": _safe_float(row.get("ddof2")),
                    "source": "group",
                }
                try:
                    result["apa"] = APAFormatter.anova(
                        f=result["F"], df1=float(result["df1"] or 0),
                        df2=float(result["df2"] or 0), p=result["p"],
                        eta_sq=0
                    )
                except Exception:
                    pass
                return self._final(args, result, "research_anova")

            # Standard one-way ANOVA via pingouin
            rows = []
            for grp_idx, grp in enumerate(groups_data):
                for val in grp:
                    rows.append({"group": group_labels[grp_idx], "value": float(val)})
            df_long = pd.DataFrame(rows)
            res = pg.anova(data=df_long, dv="value", between="group").round(6)
            row = res.iloc[0]
            f_val = _safe_float(row.get("F"))
            p_val = _safe_float(row.get("p_unc"))
            np2 = _safe_float(row.get("np2"))
            df1 = _safe_float(row.get("ddof1"))
            df2 = _safe_float(row.get("ddof2"))
            result = {
                "test": "one-way ANOVA",
                "F": f_val,
                "p": p_val,
                "np2": np2,
                "df1": df1,
                "df2": df2,
                "source": "group",
            }
            try:
                result["apa"] = APAFormatter.anova(
                    f=f_val, df1=float(df1 or 0), df2=float(df2 or 0),
                    p=p_val, eta_sq=np2 or 0
                )
            except Exception:
                pass
            return self._final(args, result, "research_anova")

        # ---- scipy fallback ----
        if not HAS_SCIPY:
            return _json({"error": "Neither pingouin nor scipy is available."})

        if repeated:
            # Friedman test as fallback for repeated measures
            try:
                stat, p_val = sp_stats.friedmanchisquare(*groups_data)
                return self._final(args, {
                    "test": "Friedman test (scipy fallback for rm ANOVA)",
                    "chi2": float(stat),
                    "p": float(p_val),
                    "source": "group",
                }, "research_anova")
            except Exception as e:
                return _json({"error": str(e)})

        if anova_type == "welch":
            # Welch ANOVA manual computation
            means = [np.mean(g) for g in groups_data]
            vars_ = [np.var(g, ddof=1) for g in groups_data]
            ns = [len(g) for g in groups_data]
            grand_mean = np.sum([m * n for m, n in zip(means, ns)]) / np.sum(ns)
            w = [n / v for n, v in zip(ns, vars_)]
            w_sum = np.sum(w)
            w_mean = np.sum([wi * mi for wi, mi in zip(w, means)]) / w_sum
            f_num = np.sum([wi * (mi - w_mean) ** 2 for wi, mi in zip(w, means)]) / (k - 1)
            f_den = (1 + (2 * (k - 2) / (k ** 2 - 1)) *
                     np.sum([(1 - wi / w_sum) ** 2 / (ni - 1)
                             for wi, ni in zip(w, ns)]))
            f_val = f_num / f_den if f_den != 0 else 0.0
            df1 = k - 1
            df2_num = (k ** 2 - 1) / 3
            df2_den = np.sum([(1 - wi / w_sum) ** 2 / (ni - 1)
                              for wi, ni in zip(w, ns)])
            df2 = df2_num / df2_den if df2_den != 0 else 1
            p_val = float(sp_stats.f.sf(f_val, df1, df2))
            return self._final(args, {
                "test": "Welch ANOVA (scipy fallback)",
                "F": float(f_val),
                "p": p_val,
                "np2": _eta_squared(f_val, df1, df2),
                "df1": df1,
                "df2": _safe_float(df2),
                "source": "group",
            }, "research_anova")

        f_val, p_val = sp_stats.f_oneway(*groups_data)
        df1 = k - 1
        df2 = sum(len(g) for g in groups_data) - k
        return self._final(args, {
            "test": "one-way ANOVA",
            "F": float(f_val),
            "p": float(p_val),
            "np2": _eta_squared(float(f_val), df1, df2),
            "df1": df1,
            "df2": df2,
            "source": "group",
        }, "research_anova")

    # -----------------------------------------------------------------------
    # Chi-square tests
    # -----------------------------------------------------------------------

    def chi_square(self, args: dict) -> str:
        """Chi-square test.

        Args:
            table: 2-D list (contingency table), OR
                   ``row_col`` + ``col_col`` against a DataFrame source.
            test: 'independence' | 'goodness' | 'fisher' (default 'independence').

        Returns JSON with: chi2, p, dof, expected, cramers_v.
        """
        args = self._normalize_args(args)
        # Cross-tabulate columns into a contingency table when available
        if "table" not in args or args.get("table") is None:
            row_col = args.get("row_col")
            col_col = args.get("col_col")
            if row_col and col_col:
                df = self._resolve_input_df(args)
                if df is not None and row_col in df.columns and col_col in df.columns:
                    ct = pd.crosstab(df[row_col], df[col_col])
                    args = {**args, "table": ct.values.tolist()}
        raw = args.get("table")
        test_type = args.get("test", "independence")

        if not isinstance(raw, list) or len(raw) == 0:
            return _json({"error": "table must be a non-empty 2-D list."})

        try:
            table = np.asarray(raw, dtype=np.float64)
            if table.ndim == 1:
                # Goodness-of-fit: 1-D observed frequencies
                if test_type == "goodness" or len(raw) == 1:
                    table = table.reshape(1, -1)
        except (TypeError, ValueError):
            return _json({"error": "table must contain numeric values."})

        if test_type == "fisher":
            if table.shape != (2, 2):
                return _json({"error": "Fisher exact test requires a 2x2 table."})
            oddsr, p_val = sp_stats.fisher_exact(table)
            result = {
                "test": "Fisher exact test",
                "odds_ratio": float(oddsr),
                "p": float(p_val),
            }
            try:
                p_str = f"p = {result['p']:.3f}".lstrip("0") if result['p'] is not None else "p = .---"
                if result['p'] is not None and result['p'] < 0.001:
                    p_str = "p < .001"
                result["apa"] = (
                    f"A Fisher exact test showed an association with odds ratio = {result['odds_ratio']:.2f}, "
                    f"{p_str}."
                )
            except Exception:
                pass
            return self._final(args, result, "research_chi_square")

        if not HAS_SCIPY:
            return _json({"error": "scipy is required for chi-square tests."})

        chi2, p_val, dof, expected = sp_stats.chi2_contingency(table)
        cv = _cramers_v(table)

        result: Dict[str, Any] = {
            "test": "chi-square test of independence"
            if table.shape[0] > 1 and table.shape[1] > 1
            else "chi-square goodness-of-fit",
            "chi2": float(chi2),
            "p": float(p_val),
            "dof": int(dof),
            "expected": expected.tolist(),
            "cramers_v": cv,
        }

        # Goodness-of-fit: one row of observed, uniform expected
        if test_type == "goodness" or (table.shape[0] == 1):
            obs = table.ravel()
            exp_uniform = np.full_like(obs, obs.sum() / len(obs))
            chi2_gof, p_gof = sp_stats.chisquare(obs, f_exp=exp_uniform)
            result["test"] = "chi-square goodness-of-fit"
            result["chi2"] = float(chi2_gof)
            result["p"] = float(p_gof)
            result["dof"] = int(len(obs) - 1)
            result["expected"] = exp_uniform.tolist()

        try:
            n = int(np.sum(table))
            result["apa"] = APAFormatter.chi_square(
                chi2=result["chi2"], df=float(result["dof"]),
                p=result["p"], n=n, cramers_v=result.get("cramers_v")
            )
        except Exception:
            pass
        return self._final(args, result, "research_chi_square")

    # -----------------------------------------------------------------------
    # Non-parametric tests
    # -----------------------------------------------------------------------

    def nonparametric(self, args: dict) -> str:
        """Non-parametric tests.

        Args:
            groups: list of lists, OR ``value_col`` + ``group_col`` against
                    a DataFrame source.
            test: 'mann-whitney' | 'wilcoxon' | 'kruskal' | 'friedman'.
            paired: bool (used for Wilcoxon).

        Returns JSON with: statistic, p, test_name.
        """
        args = self._normalize_args(args)
        raw_groups = args.get("groups")
        test_name = args.get("test", "mann-whitney")
        paired = bool(args.get("paired", False))

        if not isinstance(raw_groups, list) or len(raw_groups) == 0:
            return _json({"error": "groups must be a non-empty list of lists."})

        groups_data = []
        for i, g in enumerate(raw_groups):
            arr = _coerce_numeric_list(g)
            if arr is None or len(arr) == 0:
                return _json({"error": f"Group {i} has no valid values."})
            groups_data.append(arr)

        if not HAS_SCIPY:
            return _json({"error": "scipy is required for non-parametric tests."})

        if test_name == "mann-whitney":
            if len(groups_data) < 2:
                return _json({"error": "Mann-Whitney requires at least 2 groups."})
            stat, p_val = sp_stats.mannwhitneyu(
                groups_data[0], groups_data[1], alternative="two-sided"
            )
            # Rank-biserial correlation as effect size
            n1, n2 = len(groups_data[0]), len(groups_data[1])
            r_rb = 1 - (2 * stat) / (n1 * n2) if (n1 * n2) != 0 else 0.0
            result = {
                "test": "Mann-Whitney U test",
                "U": float(stat),
                "p": float(p_val),
                "rank_biserial_r": float(r_rb),
            }
            try:
                result["apa"] = APAFormatter.mann_whitney(
                    u=result["U"], p=result["p"], n1=n1, n2=n2
                )
            except Exception:
                pass
            return self._final(args, result, "research_nonparametric")

        if test_name == "wilcoxon":
            if len(groups_data) < 2:
                return _json({"error": "Wilcoxon requires at least 2 groups."})
            diff = groups_data[0] - groups_data[1]
            diff = diff[diff != 0]
            if len(diff) == 0:
                return self._final(args, {"test": "Wilcoxon signed-rank test", "W": 0.0, "p": 1.0}, "research_nonparametric")
            stat, p_val = sp_stats.wilcoxon(groups_data[0], groups_data[1])
            result = {
                "test": "Wilcoxon signed-rank test",
                "W": float(stat),
                "p": float(p_val),
            }
            try:
                result["apa"] = APAFormatter.wilcoxon(
                    z=0.0, p=result["p"], n=len(diff)
                )
            except Exception:
                pass
            return self._final(args, result, "research_nonparametric")

        if test_name == "kruskal":
            if len(groups_data) < 2:
                return _json({"error": "Kruskal-Wallis requires at least 2 groups."})
            stat, p_val = sp_stats.kruskal(*groups_data)
            # Epsilon-squared effect size
            n_total = sum(len(g) for g in groups_data)
            eps_sq = (stat - len(groups_data) + 1) / (n_total - len(groups_data)) if n_total > len(groups_data) else 0.0
            result = {
                "test": "Kruskal-Wallis H test",
                "H": float(stat),
                "p": float(p_val),
                "epsilon_squared": float(eps_sq),
            }
            try:
                result["apa"] = APAFormatter.kruskal_wallis(
                    h=result["H"], p=result["p"], df=len(groups_data) - 1,
                    epsilon_squared=result["epsilon_squared"], n_total=n_total
                )
            except Exception:
                pass
            return self._final(args, result, "research_nonparametric")

        if test_name == "friedman":
            if len(groups_data) < 3:
                return _json({"error": "Friedman test requires at least 3 groups."})
            lengths = [len(g) for g in groups_data]
            if len(set(lengths)) != 1:
                return _json({"error": "Friedman test requires equal group sizes."})
            stat, p_val = sp_stats.friedmanchisquare(*groups_data)
            result = {
                "test": "Friedman test",
                "chi2": float(stat),
                "p": float(p_val),
            }
            try:
                result["apa"] = APAFormatter.friedman(
                    chi2=result["chi2"], p=result["p"],
                    n=lengths[0], k=len(groups_data)
                )
            except Exception:
                pass
            return self._final(args, result, "research_nonparametric")

        return _json({"error": f"Unknown test '{test_name}'. Use mann-whitney, wilcoxon, kruskal, or friedman."})

    # -----------------------------------------------------------------------
    # Correlation
    # -----------------------------------------------------------------------

    def correlation(self, args: dict) -> str:
        """Correlation.

        Args:
            x: list, y: list (or ``x_col``/``y_col`` against a DataFrame).
            method: 'pearson'|'spearman'|'kendall'.

        Returns JSON with: r, p, r_squared, method, n.
        """
        args = self._normalize_args(args)
        x_raw = args.get("x")
        y_raw = args.get("y")
        method = args.get("method", "pearson")

        x = _coerce_numeric_list(x_raw)
        y = _coerce_numeric_list(y_raw)

        if x is None or y is None or len(x) == 0 or len(y) == 0:
            return _json({"error": "x and y must be non-empty lists of numbers."})
        if len(x) != len(y):
            return _json({"error": "x and y must have the same length."})
        if len(x) < 3:
            return _json({"error": "At least 3 paired observations are required."})

        if not HAS_SCIPY:
            return _json({"error": "scipy is required for correlation tests."})

        if method == "spearman":
            r, p = sp_stats.spearmanr(x, y)
        elif method == "kendall":
            r, p = sp_stats.kendalltau(x, y)
        else:
            r, p = sp_stats.pearsonr(x, y)

        # 95% CI for Pearson r via Fisher z-transformation
        ci = None
        if method == "pearson" and len(x) >= 4:
            r_clamped = max(-0.9999, min(0.9999, r))
            z = math.atanh(r_clamped)
            se_z = 1 / math.sqrt(len(x) - 3)
            z_low = z - 1.96 * se_z
            z_high = z + 1.96 * se_z
            ci = [float(math.tanh(z_low)), float(math.tanh(z_high))]

        result = {
            "method": method,
            "n": len(x),
            "r": float(r),
            "p": float(p),
            "r_squared": float(r ** 2),
            "ci_95": ci,
        }
        try:
            result["apa"] = APAFormatter.correlation(
                r=result["r"], p=result["p"], n=result["n"]
            )
        except Exception:
            pass
        return self._final(args, result, "research_correlation")

    # -----------------------------------------------------------------------
    # Regression
    # -----------------------------------------------------------------------

    def regression(self, args: dict) -> str:
        """Simple / multiple regression.

        Args:
            y: list of numbers, or ``y_col`` against a DataFrame.
            X: list of lists (each inner list = one predictor's values)
               or a single list (simple regression); or ``x_cols`` (list of
               column names) when a DataFrame source is provided.
            x_names: list of str (optional).
            y_name: str (optional).

        Returns JSON with: coefficients, r_squared, adj_r_squared, F, p,
        std_errors, t_stats, p_values, residuals_summary.
        """
        args = self._normalize_args(args)
        y_raw = args.get("y")
        X_raw = args.get("X")
        x_names = args.get("x_names")
        y_name = args.get("y_name", "y")

        y = _coerce_numeric_list(y_raw)
        if y is None or len(y) == 0:
            return _json({"error": "y must be a non-empty list of numbers."})

        # Handle X: can be list of lists (multiple predictors) or single list
        if X_raw is None:
            return _json({"error": "X (predictors) is required."})

        try:
            X_arr = np.asarray(X_raw, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            elif X_arr.ndim == 2 and X_arr.shape[0] == 1 and X_arr.shape[1] == len(y):
                # Single predictor passed as [[...]]
                pass
            elif X_arr.ndim == 2:
                # Ensure rows=samples, cols=predictors
                if X_arr.shape[0] != len(y) and X_arr.shape[1] == len(y):
                    X_arr = X_arr.T
        except (TypeError, ValueError):
            return _json({"error": "X must contain numeric values."})

        if X_arr.shape[0] != len(y):
            return _json({
                "error": f"X has {X_arr.shape[0]} rows but y has {len(y)} values."
            })

        n = len(y)
        p = X_arr.shape[1]

        if x_names is None:
            x_names = [f"x{i}" for i in range(p)]
        elif len(x_names) != p:
            return _json({"error": "x_names count does not match number of predictors."})

        # Add intercept
        X_design = np.column_stack([np.ones(n), X_arr])

        # Use statsmodels if available for full summary
        if HAS_STATSMODELS:
            X_sm = sm.add_constant(X_arr)
            model = sm.OLS(y, X_sm).fit()
            coeffs = {"intercept": float(model.params[0])}
            for i, name in enumerate(x_names):
                coeffs[name] = float(model.params[i + 1])

            result = {
                "test": "OLS regression",
                "y_name": y_name,
                "coefficients": coeffs,
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "F": float(model.fvalue),
                "F_pvalue": float(model.f_pvalue),
                "std_errors": {"intercept": float(model.bse[0]),
                               **{x_names[i]: float(model.bse[i + 1]) for i in range(p)}},
                "t_stats": {"intercept": float(model.tvalues[0]),
                            **{x_names[i]: float(model.tvalues[i + 1]) for i in range(p)}},
                "p_values": {"intercept": float(model.pvalues[0]),
                             **{x_names[i]: float(model.pvalues[i + 1]) for i in range(p)}},
                "n": n,
                "predictors": p,
                "residual_se": float(np.sqrt(model.mse_resid)),
            }
            try:
                apa_lines = []
                for name in x_names:
                    b = result["coefficients"][name]
                    se = result["std_errors"][name]
                    t = result["t_stats"][name]
                    pv = result["p_values"][name]
                    apa_lines.append(APAFormatter.regression_coefficient(
                        b=b, se=se, t=t, p=pv, ci=[], predictor_name=name
                    ))
                result["apa"] = " ".join(apa_lines)
            except Exception:
                pass
            return self._final(args, result, "research_regression")

        # numpy fallback
        try:
            beta, residuals_rank, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        except np.linalg.LinAlgError:
            return _json({"error": "Regression failed: singular design matrix."})

        y_pred = X_design @ beta
        residuals = y - y_pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_sq = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        adj_r_sq = 1 - (1 - r_sq) * (n - 1) / (n - p - 1) if n > p + 1 else 0.0

        # F-test
        df_reg = p
        df_res = n - p - 1
        ms_reg = (ss_tot - ss_res) / df_reg if df_reg > 0 else 0.0
        ms_res = ss_res / df_res if df_res > 0 else 0.0
        f_val = ms_reg / ms_res if ms_res != 0 else 0.0
        p_f = float(sp_stats.f.sf(f_val, df_reg, df_res)) if HAS_SCIPY else None

        # Standard errors
        if ms_res > 0:
            try:
                cov = ms_res * np.linalg.inv(X_design.T @ X_design)
                se = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                se = np.full(p + 1, np.nan)
        else:
            se = np.full(p + 1, np.nan)

        t_stats = beta / se if not np.any(np.isnan(se)) else np.full(p + 1, np.nan)
        p_vals = 2 * sp_stats.t.sf(np.abs(t_stats), df_res) if HAS_SCIPY and not np.any(np.isnan(t_stats)) else None

        coeffs = {"intercept": float(beta[0])}
        se_dict = {"intercept": float(se[0])}
        t_dict = {"intercept": float(t_stats[0])}
        p_dict = {"intercept": float(p_vals[0]) if p_vals is not None else None}
        for i, name in enumerate(x_names):
            coeffs[name] = float(beta[i + 1])
            se_dict[name] = float(se[i + 1])
            t_dict[name] = float(t_stats[i + 1])
            p_dict[name] = float(p_vals[i + 1]) if p_vals is not None else None

        result = {
            "test": "OLS regression (numpy)",
            "y_name": y_name,
            "coefficients": coeffs,
            "r_squared": r_sq,
            "adj_r_squared": adj_r_sq,
            "F": float(f_val),
            "F_pvalue": p_f,
            "std_errors": se_dict,
            "t_stats": t_dict,
            "p_values": p_dict,
            "n": n,
            "predictors": p,
        }
        try:
            apa_lines = []
            for name in x_names:
                b = result["coefficients"][name]
                se = result["std_errors"][name]
                t = result["t_stats"][name]
                pv = result["p_values"][name]
                apa_lines.append(APAFormatter.regression_coefficient(
                    b=b, se=se, t=t, p=pv, ci=[], predictor_name=name
                ))
            result["apa"] = " ".join(apa_lines)
        except Exception:
            pass
        return self._final(args, result, "research_regression")

    # -----------------------------------------------------------------------
    # Normality tests
    # -----------------------------------------------------------------------

    def normality(self, args: dict) -> str:
        """Normality tests.

        Args:
            data: list of numbers, or ``data_col`` against a DataFrame.
            test: 'shapiro' | 'ks' | 'anderson' | 'all'.

        Returns JSON with: test name(s), statistic(s), p-value(s).
        """
        args = self._normalize_args(args)
        raw = args.get("data")
        test = args.get("test", "shapiro")
        arr = _coerce_numeric_list(raw)

        if arr is None or len(arr) < 3:
            return _json({"error": "At least 3 numeric values are required."})

        if not HAS_SCIPY:
            return _json({"error": "scipy is required for normality tests."})

        if test not in ("shapiro", "ks", "anderson", "all"):
            return _json({"error": f"Unknown test '{test}'. Use shapiro, ks, anderson, or all."})

        results: Dict[str, Any] = {"n": len(arr)}

        def _shapiro() -> dict:
            stat, p = sp_stats.shapiro(arr)
            return {"statistic": float(stat), "p": float(p)}

        def _ks() -> dict:
            stat, p = sp_stats.kstest(arr, "norm", args=(np.mean(arr), np.std(arr, ddof=1)))
            return {"statistic": float(stat), "p": float(p)}

        def _anderson() -> dict:
            res = sp_stats.anderson(arr, dist="norm")
            return {
                "statistic": float(res.statistic),
                "critical_values": res.critical_values.tolist(),
                "significance_levels": res.significance_level.tolist(),
            }

        if test in ("shapiro", "all"):
            results["shapiro_wilk"] = _shapiro()
        if test in ("ks", "all"):
            results["kolmogorov_smirnov"] = _ks()
        if test in ("anderson", "all"):
            results["anderson_darling"] = _anderson()

        return self._final(args, results, "research_normality")

    # -----------------------------------------------------------------------
    # Effect sizes
    # -----------------------------------------------------------------------

    def effect_size(self, args: dict) -> str:
        """Effect size.

        Args:
            group1: list, group2: list (or ``group1_col``/``group2_col``).
            metric: 'cohens_d' | 'hedges_g' | 'eta_squared' | 'odds_ratio'.
            table: 2x2 list (for odds_ratio).

        Returns JSON with: metric name, value, interpretation.
        """
        args = self._normalize_args(args)
        metric = args.get("metric", "cohens_d")

        if metric in ("cohens_d", "hedges_g", "eta_squared"):
            g1_raw = args.get("group1")
            g2_raw = args.get("group2")
            g1 = _coerce_numeric_list(g1_raw)
            g2 = _coerce_numeric_list(g2_raw)
            if g1 is None or g2 is None or len(g1) == 0 or len(g2) == 0:
                return _json({"error": "group1 and group2 are required for this metric."})

            if metric == "cohens_d":
                d = _cohens_d(g1, g2)
                mag = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
                return self._final(args, {"metric": "Cohen's d", "value": d, "magnitude": mag}, "research_effect_size")

            if metric == "hedges_g":
                g = _hedges_g(g1, g2)
                mag = "negligible" if abs(g) < 0.2 else "small" if abs(g) < 0.5 else "medium" if abs(g) < 0.8 else "large"
                return self._final(args, {"metric": "Hedges' g", "value": g, "magnitude": mag}, "research_effect_size")

            if metric == "eta_squared":
                if not HAS_SCIPY:
                    return _json({"error": "scipy is required for eta-squared computation."})
                f_val, _ = sp_stats.f_oneway(g1, g2)
                df1 = 1
                df2 = len(g1) + len(g2) - 2
                eta = _eta_squared(float(f_val), df1, df2)
                mag = "small" if eta < 0.06 else "medium" if eta < 0.14 else "large"
                return self._final(args, {"metric": "Eta-squared", "value": eta, "magnitude": mag}, "research_effect_size")

        if metric == "odds_ratio":
            raw_table = args.get("table")
            if raw_table is None:
                return _json({"error": "table (2x2) is required for odds_ratio."})
            try:
                t = np.asarray(raw_table, dtype=np.float64)
                if t.shape != (2, 2):
                    return _json({"error": "table must be 2x2 for odds ratio."})
            except (TypeError, ValueError):
                return _json({"error": "table must contain numeric values."})

            a, b, c, d = t[0, 0], t[0, 1], t[1, 0], t[1, 1]
            denom = (c * b)
            if denom == 0:
                return self._final(args, {"metric": "Odds ratio", "value": None, "note": "Zero cell detected"}, "research_effect_size")
            or_val = (a * d) / denom
            # 95% CI via Woolf logit method
            log_or = math.log(or_val)
            se_log = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d) if min(a, b, c, d) > 0 else None
            ci = None
            if se_log is not None:
                ci = [math.exp(log_or - 1.96 * se_log), math.exp(log_or + 1.96 * se_log)]
            return self._final(args, {
                "metric": "Odds ratio",
                "value": float(or_val),
                "ci_95": ci,
            }, "research_effect_size")

        return _json({"error": f"Unknown metric '{metric}'."})

    # -----------------------------------------------------------------------
    # Bayesian t-test
    # -----------------------------------------------------------------------

    def bayesian(self, args: dict) -> str:
        """Bayesian t-test (requires pingouin).

        Args:
            group1: list, group2: list (or ``group1_col``/``group2_col``).

        Returns JSON with: BF10 and full pingouin output.
        """
        if not HAS_PINGOUIN:
            return _json({
                "error": "Bayesian t-test requires pingouin. Install with: pip install pingouin"
            })

        args = self._normalize_args(args)
        g1_raw = args.get("group1")
        g2_raw = args.get("group2")
        g1 = _coerce_numeric_list(g1_raw)
        g2 = _coerce_numeric_list(g2_raw)

        if g1 is None or g2 is None or len(g1) == 0 or len(g2) == 0:
            return _json({"error": "group1 and group2 must be non-empty lists."})

        res = pg.ttest(g1, g2).round(6)
        row = res.iloc[0]
        bf10 = _safe_float(row.get("BF10"))

        interpretation = "anecdotal"
        if bf10 is not None:
            if bf10 > 100:
                interpretation = "decisive evidence for H1"
            elif bf10 > 30:
                interpretation = "very strong evidence for H1"
            elif bf10 > 10:
                interpretation = "strong evidence for H1"
            elif bf10 > 3:
                interpretation = "moderate evidence for H1"
            elif bf10 > 1:
                interpretation = "anecdotal evidence for H1"
            elif bf10 > 1 / 3:
                interpretation = "anecdotal evidence for H0"
            elif bf10 > 1 / 10:
                interpretation = "moderate evidence for H0"
            elif bf10 > 1 / 30:
                interpretation = "strong evidence for H0"
            else:
                interpretation = "decisive evidence for H0"

        result = {
            "test": "Bayesian independent t-test",
            "BF10": bf10,
            "t": _safe_float(row.get("T")),
            "p": _safe_float(row.get("p_val")),
            "df": _safe_float(row.get("dof")),
            "cohen_d": _safe_float(row.get("cohen_d")),
            "interpretation": interpretation,
        }
        try:
            result["apa"] = APAFormatter.bayesian_ttest(
                BF10=result["BF10"], t=result["t"], p=result["p"],
                df=result["df"], cohen_d=result["cohen_d"],
                interpretation=result["interpretation"]
            )
        except Exception:
            pass
        return self._final(args, result, "research_bayesian")

    # -----------------------------------------------------------------------
    # Auto-test selection
    # -----------------------------------------------------------------------

    def auto_test(self, args: dict) -> str:
        """Automatically select and run an appropriate statistical test.

        Args:
            data: dict with group arrays, e.g. {"group1": [...], "group2": [...]}
                  or {"groups": [[...], [...], ...]}
            research_question: str (optional hint).
            groups: list of lists (alternative to data dict).
            paired: bool.

        Returns JSON with: recommended_test, test_result, reasoning.
        """
        raw_data = args.get("data", {})
        research_q = args.get("research_question", "")
        groups_raw = args.get("groups")
        paired = bool(args.get("paired", False))

        # If long-format args present, expand into groups via _normalize_args
        if args.get("value_col") and args.get("group_col"):
            args = self._normalize_args(args)
            groups_raw = args.get("groups") or groups_raw

        # Collect groups
        group_arrays: List[np.ndarray] = []
        group_labels: List[str] = []

        if groups_raw is not None:
            for i, g in enumerate(groups_raw):
                arr = _coerce_numeric_list(g)
                if arr is not None and len(arr) > 0:
                    group_arrays.append(arr)
                    group_labels.append(f"group_{i}")
        elif isinstance(raw_data, dict):
            for key, val in raw_data.items():
                arr = _coerce_numeric_list(val)
                if arr is not None and len(arr) > 0:
                    group_arrays.append(arr)
                    group_labels.append(str(key))
        else:
            return _json({"error": "No valid data provided. Use 'groups' or 'data'."})

        if len(group_arrays) < 2:
            return _json({"error": "At least 2 groups are required."})

        reasoning: List[str] = []
        k = len(group_arrays)
        reasoning.append(f"Number of groups: {k}")

        # Step 1: Check normality of each group
        if not HAS_SCIPY:
            return _json({"error": "scipy is required for auto_test."})

        normality_results: List[dict] = []
        all_normal = True
        for i, g in enumerate(group_arrays):
            if len(g) >= 3:
                _, p_norm = sp_stats.shapiro(g)
                is_normal = p_norm > 0.05
                normality_results.append({"group": group_labels[i], "p": float(p_norm), "normal": is_normal})
                if not is_normal:
                    all_normal = False
            else:
                normality_results.append({"group": group_labels[i], "p": None, "normal": False, "note": "too few observations"})
                all_normal = False

        reasoning.append(f"Normality test (Shapiro-Wilk): {['normal' if r.get('normal') else 'non-normal' for r in normality_results]}")
        reasoning.append(f"Overall normality assumption: {'met' if all_normal else 'violated'}")

        # Temporarily disable storage so child tests do not create
        # duplicate result_ids — only the outer auto_test gets persisted.
        saved_store = self.store
        self.store = None
        try:
            # Step 2: Select test
            if k == 2:
                if paired:
                    if all_normal:
                        recommended = "paired t-test"
                        reasoning.append("Selected: paired t-test (2 groups, paired, normal)")
                        result_str = self.ttest({
                            "group1": group_arrays[0].tolist(),
                            "group2": group_arrays[1].tolist(),
                            "paired": True,
                        })
                    else:
                        recommended = "Wilcoxon signed-rank test"
                        reasoning.append("Selected: Wilcoxon signed-rank test (2 groups, paired, non-normal)")
                        result_str = self.nonparametric({
                            "groups": [g.tolist() for g in group_arrays],
                            "test": "wilcoxon",
                            "paired": True,
                        })
                else:
                    if all_normal:
                        recommended = "independent t-test"
                        reasoning.append("Selected: independent t-test (2 groups, independent, normal)")
                        result_str = self.ttest({
                            "group1": group_arrays[0].tolist(),
                            "group2": group_arrays[1].tolist(),
                            "paired": False,
                        })
                    else:
                        recommended = "Mann-Whitney U test"
                        reasoning.append("Selected: Mann-Whitney U test (2 groups, independent, non-normal)")
                        result_str = self.nonparametric({
                            "groups": [g.tolist() for g in group_arrays],
                            "test": "mann-whitney",
                        })
            else:
                if paired:
                    if all_normal:
                        recommended = "repeated-measures ANOVA"
                        reasoning.append("Selected: repeated-measures ANOVA (>2 groups, paired, normal)")
                        result_str = self.anova({
                            "data": [g.tolist() for g in group_arrays],
                            "groups": group_labels,
                            "repeated": True,
                        })
                    else:
                        recommended = "Friedman test"
                        reasoning.append("Selected: Friedman test (>2 groups, paired, non-normal)")
                        result_str = self.nonparametric({
                            "groups": [g.tolist() for g in group_arrays],
                            "test": "friedman",
                        })
                else:
                    if all_normal:
                        recommended = "one-way ANOVA"
                        reasoning.append("Selected: one-way ANOVA (>2 groups, independent, normal)")
                        result_str = self.anova({
                            "data": [g.tolist() for g in group_arrays],
                            "groups": group_labels,
                        })
                    else:
                        recommended = "Kruskal-Wallis test"
                        reasoning.append("Selected: Kruskal-Wallis test (>2 groups, independent, non-normal)")
                        result_str = self.nonparametric({
                            "groups": [g.tolist() for g in group_arrays],
                            "test": "kruskal",
                        })
        finally:
            self.store = saved_store

        # Parse the test result for inclusion
        try:
            test_result = json.loads(result_str)
        except (json.JSONDecodeError, TypeError):
            test_result = {"raw": result_str}

        return self._final(args, {
            "recommended_test": recommended,
            "test_result": test_result,
            "reasoning": reasoning,
            "normality_checks": normality_results,
            "paired": paired,
            "n_groups": k,
        }, "research_auto_test")
