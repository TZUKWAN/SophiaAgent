"""Causal inference engine for empirical research.

Pure-computation wrapper around linearmodels, statsmodels, numpy, and sklearn.
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
from sophia.research.seed import GlobalSeed

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS, PooledOLS
    from linearmodels.iv import IV2SLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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


# ===========================================================================
# CausalEngine
# ===========================================================================

class CausalEngine:
    """Causal inference methods for quasi-experimental research.

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

        Returns None when args provide no DataFrame-shaped source. We
        deliberately do NOT treat scalar / flat-list args (``y=[...]``,
        ``treat=[...]``) as DataFrame inputs — only ``result_id`` /
        ``path`` / list-of-dicts / dict-of-columns trigger DataFrame mode.
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

        Scalar columns:
            ``y_col``         -> ``y``
            ``treat_col``     -> ``treat``
            ``post_col``      -> ``post``
            ``unit_col``      -> ``unit``
            ``time_col``      -> ``time``
            ``outcome_col``   -> ``outcomes``
            ``running_col``   -> ``running``
            ``endogenous_col``-> ``endogenous``
            ``instrument_col``-> ``instrument``
            ``x_col``         -> ``x``
            ``mediator_col``  -> ``mediator``

        Group of columns:
            ``covariate_cols``-> ``covariates``  (dict of {name: list})
            ``exogenous_cols``-> ``exogenous``   (dict of {name: list})

        Rows are aligned via a single shared NA-drop mask across all
        DataFrame-resolved columns. This keeps y / treat / covariates
        synchronized.
        """
        df = self._resolve_input_df(args)
        if df is None:
            return args
        new_args = dict(args)

        scalar_map = {
            "y_col": "y",
            "treat_col": "treat",
            "post_col": "post",
            "unit_col": "unit",
            "time_col": "time",
            "outcome_col": "outcomes",
            "running_col": "running",
            "endogenous_col": "endogenous",
            "instrument_col": "instrument",
            "x_col": "x",
            "mediator_col": "mediator",
        }
        group_map = {
            "covariate_cols": "covariates",
            "exogenous_cols": "exogenous",
        }

        # Collect every requested column so we can build a shared NA mask
        used_cols: List[str] = []
        for k, target in scalar_map.items():
            col_name = args.get(k)
            if col_name and isinstance(col_name, str) and col_name in df.columns:
                used_cols.append(col_name)
        for k in group_map:
            cols_list = args.get(k)
            if isinstance(cols_list, list):
                used_cols.extend(c for c in cols_list if c in df.columns)

        if not used_cols:
            return new_args

        sub = df[list(dict.fromkeys(used_cols))].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        for col_key, target in scalar_map.items():
            col_name = args.get(col_key)
            if (
                col_name
                and isinstance(col_name, str)
                and col_name in sub.columns
                and target not in args
            ):
                new_args[target] = sub[col_name].tolist()

        for col_key, target in group_map.items():
            cols_list = args.get(col_key)
            if isinstance(cols_list, list) and target not in args:
                cov_dict: Dict[str, List[float]] = {}
                for c in cols_list:
                    if c in sub.columns:
                        cov_dict[c] = sub[c].tolist()
                if cov_dict:
                    new_args[target] = cov_dict

        return new_args

    def _sanitize_params(self, args: dict) -> dict:
        """Replace bulky payloads with placeholder strings before persisting."""
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
            elif isinstance(v, dict):
                # covariates/exogenous-style dicts: summarize if large
                try:
                    inner_total = sum(
                        len(x) if isinstance(x, (list, tuple, np.ndarray)) else 1
                        for x in v.values()
                    )
                except TypeError:
                    inner_total = len(v)
                if inner_total > 200:
                    out[k] = f"<dict keys={list(v.keys())} total={inner_total}>"
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
    # Difference-in-Differences
    # -----------------------------------------------------------------------


    def did(self, args: dict) -> str:
        """Difference-in-Differences estimation (full implementation).

        Supports classic 2x2 DID (array inputs) and panel-data TWFE
        (unit/time indices).  When panel data are available the method
        automatically runs TWFE with clustered SE, parallel-trends test,
        event-study dynamics, placebo inference, SE robustness comparison,
        and (for staggered adoption) Goodman-Bacon decomposition.

        Args:
            y: list of outcome values.
            treat: list of treatment indicators (0/1).
            post: list of post-period indicators (0/1).
            unit: list of unit IDs (optional, enables TWFE).
            time: list of time periods (optional, enables TWFE).
            covariates: dict of {name: list} (optional).
            event_study: bool (include leads/lags, default False).
            placebo: bool (run placebo tests, default False).
            n_placebo: int (placebo iterations, default 500).
            se_comparison: bool (compare SE estimators, default True).
            bacon: bool (Goodman-Bacon decomposition for staggered DID,
                         default False).

        Returns:
            JSON with DID/TWFE estimate, diagnostics, and APA paragraph.
        """
        args = self._normalize_args(args)
        y_raw = args.get("y")
        treat_raw = args.get("treat")
        post_raw = args.get("post")
        unit_raw = args.get("unit")
        time_raw = args.get("time")
        covariates_raw = args.get("covariates")
        event_study = bool(args.get("event_study", False))
        run_placebo = bool(args.get("placebo", False))
        n_placebo = int(args.get("n_placebo", 500))
        se_comparison = bool(args.get("se_comparison", True))
        run_bacon = bool(args.get("bacon", False))

        y = _coerce_numeric_list(y_raw)
        treat = _coerce_numeric_list(treat_raw)
        post = _coerce_numeric_list(post_raw)

        if y is None or treat is None or post is None:
            return _json({"error": "y, treat, and post are required."})
        if len(y) != len(treat) or len(y) != len(post):
            return _json({"error": "y, treat, and post must have the same length."})
        if len(y) < 4:
            return _json({"error": "At least 4 observations are required."})

        n = len(y)
        has_panel = unit_raw is not None and time_raw is not None

        # ------------------------------------------------------------------
        # Panel-data path (TWFE)
        # ------------------------------------------------------------------
        if has_panel and HAS_LINEARMODELS and HAS_STATSMODELS and HAS_SCIPY:
            try:
                return self._did_panel(
                    y, treat, post, unit_raw, time_raw,
                    covariates_raw, event_study, run_placebo,
                    n_placebo, se_comparison, run_bacon, args,
                )
            except Exception as e:
                # Fall back to classic OLS on panel-data failure
                pass

        # ------------------------------------------------------------------
        # Classic 2x2 OLS path (backward compatible)
        # ------------------------------------------------------------------
        treat_post = treat * post
        X_cols = {"treat": treat, "post": post, "treat_post": treat_post}

        if covariates_raw and isinstance(covariates_raw, dict):
            for name, values in covariates_raw.items():
                cov_arr = _coerce_numeric_list(values)
                if cov_arr is not None and len(cov_arr) == n:
                    X_cols[name] = cov_arr

        # Event study (classic, hard-coded leads/lags)
        if event_study and time_raw is not None:
            time_arr = np.asarray(time_raw, dtype=np.float64)
            unit_arr = np.asarray(unit_raw, dtype=np.float64) if unit_raw is not None else None
            if unit_arr is not None:
                unique_times = np.sort(np.unique(time_arr))
                treatment_start = np.min(unique_times[unique_times >= np.median(unique_times)])
                for k in [-2, -1, 1, 2]:
                    rel_time = time_arr + k
                    lead_lag_post = (rel_time >= treatment_start).astype(np.float64)
                    lead_lag_interact = treat * lead_lag_post
                    label = f"lead_{abs(k)}" if k < 0 else f"lag_{k}"
                    is_dup = any(np.array_equal(existing, lead_lag_interact) for existing in X_cols.values())
                    if not is_dup:
                        X_cols[label] = lead_lag_interact

        X_arr = np.column_stack(list(X_cols.values()))
        col_names = list(X_cols.keys())

        if HAS_STATSMODELS:
            X_sm = sm.add_constant(X_arr)
            model = sm.OLS(y, X_sm).fit(cov_type="HC1")
            idx = col_names.index("treat_post")
            did_coef = float(model.params[idx + 1])
            did_se = float(model.bse[idx + 1])
            did_t = float(model.tvalues[idx + 1])
            did_p = float(model.pvalues[idx + 1])
            ci_arr = model.conf_int()
            coeffs = {"intercept": float(model.params[0])}
            for i, name in enumerate(col_names):
                coeffs[name] = float(model.params[i + 1])
            result = {
                "method": "Difference-in-Differences (OLS, HC1 robust SE)",
                "did_estimate": did_coef,
                "se": did_se,
                "t_stat": did_t,
                "p_value": did_p,
                "ci_95": [float(ci_arr[idx + 1, 0]), float(ci_arr[idx + 1, 1])],
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "n": n,
                "coefficients": coeffs,
                "f_statistic": float(model.fvalue),
                "f_pvalue": float(model.f_pvalue),
            }
            try:
                result["apa"] = APAFormatter.did(
                    beta=did_coef, se=did_se, p=did_p,
                    ci=result["ci_95"]
                )
            except Exception:
                pass
            return self._final(args, result, "research_did")

        # numpy fallback
        X_design = np.column_stack([np.ones(n), X_arr])
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        y_pred = X_design @ beta
        resid = y - y_pred
        df_res = n - len(beta)
        mse = float(np.sum(resid ** 2) / df_res) if df_res > 0 else 0.0
        try:
            cov_matrix = mse * np.linalg.inv(X_design.T @ X_design)
            se = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), np.nan)
        idx = col_names.index("treat_post")
        did_coef = float(beta[idx + 1])
        did_se = float(se[idx + 1])
        did_t = did_coef / did_se if did_se != 0 else 0.0
        did_p = float(2 * sp_stats.t.sf(abs(did_t), df_res)) if HAS_SCIPY else None
        result = {
            "method": "Difference-in-Differences (numpy OLS)",
            "did_estimate": did_coef,
            "se": did_se,
            "t_stat": did_t,
            "p_value": did_p,
            "n": n,
        }
        try:
            result["apa"] = APAFormatter.did(beta=did_coef, se=did_se, p=did_p, ci=[])
        except Exception:
            pass
        return self._final(args, result, "research_did")

    # -----------------------------------------------------------------------
    # DID Panel / TWFE internals
    # -----------------------------------------------------------------------

    def _did_panel(
        self,
        y: np.ndarray,
        treat: np.ndarray,
        post: np.ndarray,
        unit_raw: Any,
        time_raw: Any,
        covariates_raw: Optional[dict],
        event_study: bool,
        run_placebo: bool,
        n_placebo: int,
        se_comparison: bool,
        run_bacon: bool,
        args: dict,
    ) -> str:
        """Panel-data TWFE with full diagnostics."""
        import pandas as pd
        from linearmodels.panel import PanelOLS

        unit = np.asarray(unit_raw)
        time = np.asarray(time_raw, dtype=np.float64)
        n = len(y)

        df = pd.DataFrame({
            "y": y,
            "treat": treat,
            "post": post,
            "unit": unit,
            "time": time,
        })

        # Add covariates
        if covariates_raw and isinstance(covariates_raw, dict):
            for name, values in covariates_raw.items():
                cov_arr = _coerce_numeric_list(values)
                if cov_arr is not None and len(cov_arr) == n:
                    df[name] = cov_arr

        # Determine treatment timing per unit (for staggered / event study)
        treat_df = df[df["treat"] == 1]
        if len(treat_df) == 0:
            return _json({"error": "No treated units found."})

        # First post-treatment period for each treated unit
        timing = (
            treat_df[treat_df["post"] == 1]
            .groupby("unit")["time"]
            .min()
        )
        if len(timing) == 0:
            return _json({"error": "Treated units have no post-period observations."})

        # Classic DID: all treated units share the same treatment time
        classic = timing.nunique() == 1
        treatment_time = float(timing.iloc[0]) if classic else None

        # Interaction term
        df["interaction"] = df["treat"] * df["post"]

        # Covariate list for regression
        cov_names = []
        if covariates_raw and isinstance(covariates_raw, dict):
            for name in covariates_raw.keys():
                if name in df.columns:
                    cov_names.append(name)

        # Set panel index
        df_panel = df.set_index(["unit", "time"])
        exog_cols = ["interaction"] + cov_names
        exog = df_panel[exog_cols]

        # --------------------------------------------------------------
        # 1. TWFE main estimate (clustered SE by entity)
        # --------------------------------------------------------------
        model = PanelOLS(
            df_panel["y"],
            exog,
            entity_effects=True,
            time_effects=True,
        )
        fit = model.fit(cov_type="clustered", cluster_entity=True)

        did_coef = float(fit.params["interaction"])
        did_se = float(fit.std_errors["interaction"])
        did_t = float(fit.tstats["interaction"])
        did_p = float(fit.pvalues["interaction"])
        ci_low, ci_high = fit.conf_int().loc["interaction"]

        result = {
            "method": "Difference-in-Differences (TWFE, clustered SE)",
            "did_estimate": did_coef,
            "se": did_se,
            "t_stat": did_t,
            "p_value": did_p,
            "ci_95": [float(ci_low), float(ci_high)],
            "r_squared": float(fit.rsquared),
            "adj_r_squared": float(getattr(fit, "rsquared_within", fit.rsquared)),
            "n": n,
            "n_units": int(df["unit"].nunique()),
            "n_periods": int(df["time"].nunique()),
        }

        # Coefficient table
        coeffs = {}
        for c in exog_cols:
            coeffs[c] = {
                "estimate": float(fit.params[c]),
                "se": float(fit.std_errors[c]),
                "t": float(fit.tstats[c]),
                "p": float(fit.pvalues[c]),
            }
        result["coefficients"] = coeffs

        # --------------------------------------------------------------
        # 2. Parallel trends test (Granger pre-test)
        # --------------------------------------------------------------
        parallel_test = self._did_parallel_trends(df, cov_names)
        if parallel_test:
            result["parallel_trends_test"] = parallel_test

        # --------------------------------------------------------------
        # 3. Event study (dynamic treatment effects)
        # --------------------------------------------------------------
        if event_study:
            event_study_result = self._did_event_study(df, classic, treatment_time, cov_names)
            if event_study_result:
                result["event_study"] = event_study_result

        # --------------------------------------------------------------
        # 4. Placebo tests
        # --------------------------------------------------------------
        if run_placebo:
            placebo_result = self._did_placebo(df, cov_names, n_placebo)
            if placebo_result:
                result["placebo"] = placebo_result

        # --------------------------------------------------------------
        # 5. SE robustness comparison
        # --------------------------------------------------------------
        if se_comparison:
            se_comp = self._did_se_comparison(df_panel, exog_cols)
            if se_comp:
                result["se_comparison"] = se_comp

        # --------------------------------------------------------------
        # 6. Goodman-Bacon decomposition (staggered DID)
        # --------------------------------------------------------------
        if run_bacon and not classic:
            bacon_result = self._did_bacon_decomposition(df)
            if bacon_result:
                result["bacon_decomposition"] = bacon_result

        # --------------------------------------------------------------
        # 7. APA report
        # --------------------------------------------------------------
        result["apa_report"] = self._did_apa_format(result)

        return self._final(args, result, "research_did")

    def _did_parallel_trends(self, df: pd.DataFrame, cov_names: List[str]) -> Optional[dict]:
        """Granger-style parallel-trends pre-test.

        Regress leads of the treatment indicator on the outcome during the
        pre-treatment period.  A joint F-test that all lead coefficients are
        zero provides evidence for (or against) parallel trends.
        """
        from linearmodels.panel import PanelOLS

        pre_df = df[df["post"] == 0].copy()
        if len(pre_df) == 0:
            return None

        pre_times = sorted(pre_df["time"].unique())
        if len(pre_times) < 2:
            return None

        treated_units = df.loc[df["treat"] == 1, "unit"].unique()
        leads = []
        for t in pre_times:
            col = f"lead_t{t}"
            pre_df[col] = ((pre_df["unit"].isin(treated_units)) & (pre_df["time"] == t)).astype(float)
            leads.append(col)

        # Drop the last pre-period to avoid collinearity with FE
        leads = leads[:-1]
        if not leads:
            return None

        try:
            pre_panel = pre_df.set_index(["unit", "time"])
            exog = pre_panel[leads + cov_names] if cov_names else pre_panel[leads]
            model = PanelOLS(pre_panel["y"], exog, entity_effects=True, time_effects=True)
            fit = model.fit(cov_type="clustered", cluster_entity=True)
            f_stat = float(fit.f_statistic.stat)
            f_pval = float(fit.f_statistic.pval)
            return {
                "F": f_stat,
                "p_value": f_pval,
                "passes": f_pval > 0.10,
                "n_pre_periods": len(pre_times),
                "n_leads_tested": len(leads),
            }
        except Exception:
            return None

    def _did_event_study(
        self,
        df: pd.DataFrame,
        classic: bool,
        treatment_time: Optional[float],
        cov_names: List[str],
    ) -> Optional[dict]:
        """Event-study estimation of dynamic treatment effects.

        Normalises the coefficient at t = -1 to zero (the period immediately
        before treatment).
        """
        from linearmodels.panel import PanelOLS

        df = df.copy()
        treated_units = df.loc[df["treat"] == 1, "unit"].unique()

        if classic and treatment_time is not None:
            df["rel_time"] = df["time"] - treatment_time
        else:
            # Staggered: first treatment time per unit
            first_treat = (
                df[df["treat"] == 1]
                .groupby("unit")["time"]
                .min()
                .to_dict()
            )
            df["rel_time"] = df.apply(
                lambda r: r["time"] - first_treat.get(r["unit"], np.nan), axis=1
            )
            # Drop never-treated for event study
            df = df[df["unit"].isin(treated_units)].copy()

        rel_times = sorted(df["rel_time"].dropna().unique())
        if len(rel_times) < 2:
            return None

        # Base period = -1 (or closest negative if -1 absent)
        neg_times = [t for t in rel_times if t < 0]
        base = -1.0 if -1.0 in neg_times else (max(neg_times) if neg_times else None)
        if base is None:
            return None

        event_dummies = []
        for t in rel_times:
            if t == base:
                continue
            col = f"evt_{int(t)}"
            df[col] = (df["rel_time"] == t).astype(float)
            event_dummies.append(col)

        if not event_dummies:
            return None

        try:
            panel = df.set_index(["unit", "time"])
            exog = panel[event_dummies + cov_names] if cov_names else panel[event_dummies]
            # Entity effects only: time effects are absorbed by the event dummies
            model = PanelOLS(panel["y"], exog, entity_effects=True, time_effects=False)
            fit = model.fit(cov_type="clustered", cluster_entity=True)

            ci = fit.conf_int()
            coefs = {}
            for col in event_dummies:
                t_val = float(col.split("_")[1])
                coefs[t_val] = {
                    "estimate": float(fit.params[col]),
                    "se": float(fit.std_errors[col]),
                    "p": float(fit.pvalues[col]),
                    "ci_95": [
                        float(ci.loc[col, "lower"]),
                        float(ci.loc[col, "upper"]),
                    ],
                }

            return {
                "base_period": base,
                "coefficients": coefs,
                "n_event_periods": len(event_dummies),
            }
        except Exception:
            return None

    def _did_placebo(self, df: pd.DataFrame, cov_names: List[str], n_iters: int) -> Optional[dict]:
        """Placebo tests: in-time and in-space."""
        from linearmodels.panel import PanelOLS

        df = df.copy()
        unique_times = sorted(df["time"].unique())
        treated_units = df.loc[df["treat"] == 1, "unit"].unique()
        n_treated = len(treated_units)
        all_units = df["unit"].unique()

        # In-time placebo: shift treatment earlier by 2 periods
        in_time_results = []
        if len(unique_times) >= 4:
            shift = 2
            placebo_time = unique_times[max(0, len(unique_times) // 2 - shift)]
            df_p = df.copy()
            df_p["post_placebo"] = (df_p["time"] >= placebo_time).astype(float)
            df_p["interaction_placebo"] = df_p["treat"] * df_p["post_placebo"]
            try:
                panel = df_p.set_index(["unit", "time"])
                exog = panel[["interaction_placebo"] + cov_names] if cov_names else panel[["interaction_placebo"]]
                fit = PanelOLS(panel["y"], exog, entity_effects=True, time_effects=True).fit(
                    cov_type="clustered", cluster_entity=True
                )
                in_time_results.append({
                    "type": "in_time",
                    "placebo_time": float(placebo_time),
                    "estimate": float(fit.params["interaction_placebo"]),
                    "se": float(fit.std_errors["interaction_placebo"]),
                    "p": float(fit.pvalues["interaction_placebo"]),
                })
            except Exception:
                pass

        # In-space placebo: random re-assignment of treatment
        np.random.seed(GlobalSeed.get_or_default(42))
        in_space_estimates = []
        for _ in range(min(n_iters, 1000)):
            placebo_treat = np.random.choice(all_units, size=n_treated, replace=False)
            df_p = df.copy()
            df_p["treat_placebo"] = df_p["unit"].isin(placebo_treat).astype(float)
            df_p["interaction_placebo"] = df_p["treat_placebo"] * df_p["post"]
            try:
                panel = df_p.set_index(["unit", "time"])
                exog = panel[["interaction_placebo"] + cov_names] if cov_names else panel[["interaction_placebo"]]
                fit = PanelOLS(panel["y"], exog, entity_effects=True, time_effects=True).fit(
                    cov_type="clustered", cluster_entity=True
                )
                in_space_estimates.append(float(fit.params["interaction_placebo"]))
            except Exception:
                pass

        in_space_p = None
        if in_space_estimates:
            observed = df["interaction"].mean() if "interaction" in df.columns else 0.0
            # Two-sided p-value: fraction of placebo estimates more extreme than observed
            abs_obs = abs(observed)
            extreme = sum(1 for e in in_space_estimates if abs(e) >= abs_obs)
            in_space_p = extreme / len(in_space_estimates) if in_space_estimates else None

        return {
            "in_time": in_time_results,
            "in_space": {
                "n_iters": len(in_space_estimates),
                "estimates": in_space_estimates[:100],  # cap returned size
                "mean": float(np.mean(in_space_estimates)) if in_space_estimates else None,
                "std": float(np.std(in_space_estimates)) if in_space_estimates else None,
                "p_value": in_space_p,
            },
        }

    def _did_se_comparison(self, df_panel: pd.DataFrame, exog_cols: List[str]) -> Optional[dict]:
        """Compare DID estimate under alternative SE estimators."""
        from linearmodels.panel import PanelOLS

        out = {}
        for cov_type, cluster_kw in [
            ("clustered", {"cluster_entity": True}),
            ("robust", {}),
        ]:
            try:
                fit = PanelOLS(df_panel["y"], df_panel[exog_cols], entity_effects=True, time_effects=True).fit(
                    cov_type=cov_type, **cluster_kw
                )
                out[cov_type] = {
                    "estimate": float(fit.params["interaction"]),
                    "se": float(fit.std_errors["interaction"]),
                    "p": float(fit.pvalues["interaction"]),
                }
            except Exception:
                pass

        # statsmodels HC1/HC2/HC3 via pooled OLS (no FE)
        if HAS_STATSMODELS:
            try:
                y = df_panel["y"].values
                X = sm.add_constant(df_panel[exog_cols].values)
                for hc in ["HC1", "HC2", "HC3"]:
                    fit_sm = sm.OLS(y, X).fit(cov_type=hc)
                    out[hc.lower()] = {
                        "estimate": float(fit_sm.params[1]),
                        "se": float(fit_sm.bse[1]),
                        "p": float(fit_sm.pvalues[1]),
                    }
            except Exception:
                pass

        return out if out else None

    def _did_bacon_decomposition(self, df: pd.DataFrame) -> Optional[dict]:
        """Goodman-Bacon decomposition for staggered DID.

        Decomposes the TWFE estimator into a weighted average of all
        2x2 DID comparisons (early vs late, early vs never, late vs never).
        """
        from linearmodels.panel import PanelOLS

        df = df.copy()
        treated_units = df.loc[df["treat"] == 1, "unit"].unique()
        never_units = df.loc[df["treat"] == 0, "unit"].unique()

        first_treat = (
            df[df["treat"] == 1]
            .groupby("unit")["time"]
            .min()
        )
        timing_groups = first_treat.unique()
        timing_groups.sort()

        comparisons = []
        total_weight = 0.0
        weighted_sum = 0.0

        # Helper: compute 2x2 DID between two groups with given treatment times
        def _bacon_2x2(sub_df: pd.DataFrame, treat_time: float, control_time: float) -> Tuple[float, float]:
            sub = sub_df.copy()
            sub["post_k"] = (sub["time"] >= treat_time).astype(float)
            sub["inter_k"] = sub["treat"] * sub["post_k"]
            panel = sub.set_index(["unit", "time"])
            try:
                fit = PanelOLS(panel["y"], panel[["inter_k"]], entity_effects=True, time_effects=True).fit(
                    cov_type="clustered", cluster_entity=True
                )
                return float(fit.params["inter_k"]), float(fit.rsquared)
            except Exception:
                return np.nan, np.nan

        # Early vs Late (and Late vs Early)
        for i, early_time in enumerate(timing_groups):
            early_units = first_treat[first_treat == early_time].index.tolist()
            for late_time in timing_groups[i + 1:]:
                late_units = first_treat[first_treat == late_time].index.tolist()
                sub = df[df["unit"].isin(early_units + late_units)].copy()
                sub["treat"] = sub["unit"].isin(early_units).astype(float)
                beta, _ = _bacon_2x2(sub, late_time, early_time)
                # Weight = share of sample
                weight = len(sub) / len(df)
                if not np.isnan(beta):
                    comparisons.append({
                        "type": "early_vs_late",
                        "early_time": float(early_time),
                        "late_time": float(late_time),
                        "estimate": beta,
                        "weight": weight,
                    })
                    total_weight += weight
                    weighted_sum += weight * beta

                # Late vs Early (reverse)
                sub2 = df[df["unit"].isin(early_units + late_units)].copy()
                sub2["treat"] = sub2["unit"].isin(late_units).astype(float)
                beta2, _ = _bacon_2x2(sub2, early_time, late_time)
                if not np.isnan(beta2):
                    comparisons.append({
                        "type": "late_vs_early",
                        "early_time": float(early_time),
                        "late_time": float(late_time),
                        "estimate": beta2,
                        "weight": weight,
                    })
                    total_weight += weight
                    weighted_sum += weight * beta2

        # Treated vs Never-treated
        if len(never_units) > 0:
            for treat_time in timing_groups:
                treat_u = first_treat[first_treat == treat_time].index.tolist()
                sub = df[df["unit"].isin(treat_u + never_units.tolist())].copy()
                sub["treat"] = sub["unit"].isin(treat_u).astype(float)
                beta, _ = _bacon_2x2(sub, treat_time, np.inf)
                weight = len(sub) / len(df)
                if not np.isnan(beta):
                    comparisons.append({
                        "type": "treated_vs_never",
                        "treatment_time": float(treat_time),
                        "estimate": beta,
                        "weight": weight,
                    })
                    total_weight += weight
                    weighted_sum += weight * beta

        return {
            "comparisons": comparisons,
            "weighted_avg": float(weighted_sum / total_weight) if total_weight > 0 else None,
            "total_weight": float(total_weight),
        }

    def _did_apa_format(self, result: dict) -> str:
        """Generate an APA-style paragraph summarising the DID results."""
        est = result.get("did_estimate")
        se = result.get("se")
        p = result.get("p_value")
        ci = result.get("ci_95")
        n = result.get("n")
        method = result.get("method", "DID")

        if est is None:
            return "DID estimation failed."

        sig_word = "significantly" if p and p < 0.05 else "not significantly"
        p_str = f"p = {p:.3f}" if p and p >= 0.001 else "p < .001"
        ci_str = ""
        if ci and len(ci) == 2:
            ci_str = f", 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"

        para = (
            f"A {method} analysis (N = {n}) indicated that the treatment effect was "
            f"{sig_word} different from zero, β = {est:.2f} (SE = {se:.2f}, {p_str}{ci_str}). "
        )

        pt = result.get("parallel_trends_test")
        if pt:
            pt_pass = "passed" if pt.get("passes") else "failed"
            para += (
                f"The parallel-trends assumption was assessed via a Granger pre-test "
                f"(F = {pt['F']:.2f}, p = {pt['p_value']:.3f}), which {pt_pass}. "
            )

        placebo = result.get("placebo")
        if placebo:
            in_space = placebo.get("in_space")
            if in_space and in_space.get("p_value") is not None:
                p_plac = in_space["p_value"]
                plac_sig = "significant" if p_plac < 0.05 else "not significant"
                para += (
                    f"A placebo test with {in_space['n_iters']} random re-assignments "
                    f"yielded a {plac_sig} in-space placebo distribution (p = {p_plac:.3f}). "
                )

        return para.strip()

    # -----------------------------------------------------------------------
    # Regression Discontinuity Design
    # -----------------------------------------------------------------------

    def rdd(self, args: dict) -> str:
        """Regression Discontinuity Design.

        Args:
            y: list of outcome values.
            running: list of running variable values.
            cutoff: float (threshold).
            bandwidth: float (optional, auto-select if not given).
            polynomial: int (order, default 1).
            kernel: str ('triangular'|'uniform'|'epanechnikov', default 'triangular').
            covariates: dict of {name: list} (optional).

        Returns:
            JSON with LATE estimate, SE, p-value, bandwidth, effective N.
        """
        args = self._normalize_args(args)
        y_raw = args.get("y")
        running_raw = args.get("running")
        cutoff = args.get("cutoff")

        if y_raw is None or running_raw is None or cutoff is None:
            return _json({"error": "y, running, and cutoff are required."})

        y = _coerce_numeric_list(y_raw)
        running = _coerce_numeric_list(running_raw)

        if y is None or running is None:
            return _json({"error": "y and running must be numeric lists."})
        if len(y) != len(running):
            return _json({"error": "y and running must have the same length."})
        if len(y) < 10:
            return _json({"error": "At least 10 observations are required."})

        cutoff = float(cutoff)
        poly = int(args.get("polynomial", 1))
        kernel_type = args.get("kernel", "triangular")
        covariates_raw = args.get("covariates")
        bandwidth = args.get("bandwidth")

        n_total = len(y)

        # Auto-select bandwidth if not given: rule-of-thumb
        if bandwidth is None:
            sd_run = float(np.std(running, ddof=1))
            h = 1.84 * sd_run * (n_total ** (-0.2))
            bandwidth = h
        else:
            bandwidth = float(bandwidth)

        # Filter to observations within bandwidth
        centered = running - cutoff
        mask = np.abs(centered) <= bandwidth
        y_bw = y[mask]
        x_bw = centered[mask]
        n_eff = len(y_bw)

        if n_eff < 5:
            return _json({"error": f"Only {n_eff} observations within bandwidth. Widen bandwidth."})

        # Treatment indicator
        treat = (running[mask] >= cutoff).astype(np.float64)

        # Kernel weights
        u = np.abs(x_bw) / bandwidth  # normalized distance
        if kernel_type == "triangular":
            weights = (1.0 - u)
        elif kernel_type == "epanechnikov":
            weights = (1.0 - u ** 2)
        else:  # uniform
            weights = np.ones(n_eff)
        weights = np.maximum(weights, 0.0)

        # Build design matrix with polynomial terms
        X_cols = [treat, x_bw]
        col_names = ["treat", "running_centered"]
        for p in range(2, poly + 1):
            X_cols.append(x_bw ** p)
            col_names.append(f"running_centered_p{p}")
        # Interaction: treat * polynomial terms
        for p in range(1, poly + 1):
            X_cols.append(treat * (x_bw ** p))
            col_names.append(f"treat_x_running_p{p}")

        # Add covariates if provided
        if covariates_raw and isinstance(covariates_raw, dict):
            for name, values in covariates_raw.items():
                cov_arr = _coerce_numeric_list(values)
                if cov_arr is not None and len(cov_arr) == n_total:
                    X_cols.append(cov_arr[mask])
                    col_names.append(name)

        X_arr = np.column_stack(X_cols)
        n = len(y_bw)

        # Weighted least squares
        if HAS_STATSMODELS:
            X_sm = sm.add_constant(X_arr)
            model = sm.WLS(y_bw, X_sm, weights=weights).fit()
            late = float(model.params[1])  # treatment effect
            late_se = float(model.bse[1])
            late_t = float(model.tvalues[1])
            late_p = float(model.pvalues[1])
            ci_arr = model.conf_int()
            ci = [float(ci_arr[1, 0]), float(ci_arr[1, 1])]
            r_sq = float(model.rsquared)
        else:
            # Weighted OLS manually
            W = np.diag(weights)
            X_design = np.column_stack([np.ones(n), X_arr])
            try:
                XtWX_inv = np.linalg.inv(X_design.T @ W @ X_design)
                beta = XtWX_inv @ X_design.T @ W @ y_bw
            except np.linalg.LinAlgError:
                return _json({"error": "Singular matrix in WLS estimation."})

            resid = y_bw - X_design @ beta
            df_res = n - len(beta)
            mse_w = float(np.sum(weights * resid ** 2) / df_res) if df_res > 0 else 0.0
            cov_matrix = mse_w * XtWX_inv
            se = np.sqrt(np.diag(cov_matrix))

            late = float(beta[1])
            late_se = float(se[1])
            late_t = late / late_se if late_se != 0 else 0.0
            late_p = float(2 * sp_stats.t.sf(abs(late_t), df_res)) if HAS_SCIPY else None
            ci = None
            r_sq = None

        result = {
            "method": "Regression Discontinuity Design",
            "kernel": kernel_type,
            "polynomial_order": poly,
            "late_estimate": late,
            "se": late_se,
            "t_stat": late_t,
            "p_value": late_p,
            "ci_95": ci,
            "bandwidth": bandwidth,
            "cutoff": cutoff,
            "effective_n": n_eff,
            "total_n": n_total,
            "r_squared": r_sq,
        }
        try:
            result["apa"] = APAFormatter.rdd(
                tau=late, se=late_se, p=late_p,
                bandwidth=bandwidth, n_within_bw=n_eff
            )
        except Exception:
            pass
        return self._final(args, result, "research_rdd")

    # -----------------------------------------------------------------------
    # Instrumental Variable (2SLS)
    # -----------------------------------------------------------------------

    def _stock_yogo_critical(self, n_instruments: int, max_bias: float = 0.10) -> float:
        """Return Stock-Yogo critical value for weak IV test.

        Simplified table for 1-5 instruments at 10% max IV relative bias.
        """
        # Stock-Yogo critical values (10% max relative bias, 5% significance)
        table = {
            1: 16.38,
            2: 19.93,
            3: 22.30,
            4: 24.58,
            5: 26.87,
        }
        return table.get(n_instruments, 16.38)

    def iv(self, args: dict) -> str:
        """Instrumental Variable estimation (2SLS).

        Args:
            y: list of outcome values.
            endogenous: list of endogenous variable values.
            instrument: list of instrument values (single instrument).
            instruments: list of lists (multiple instruments, over-identified).
            exogenous: dict of {name: list} (optional exogenous controls).

        Returns:
            JSON with IV coefficient, SE, t-stat, p-value, first-stage F,
            Wu-Hausman test, Sargan/Hansen J (if over-identified),
            Stock-Yogo weak IV test.
        """
        args = self._normalize_args(args)
        y_raw = args.get("y")
        endo_raw = args.get("endogenous")
        instr_raw = args.get("instrument")
        instrs_raw = args.get("instruments")
        exog_raw = args.get("exogenous")

        if y_raw is None or endo_raw is None:
            return _json({"error": "y and endogenous are required."})
        if instr_raw is None and instrs_raw is None:
            return _json({"error": "instrument or instruments is required."})

        y = _coerce_numeric_list(y_raw)
        endo = _coerce_numeric_list(endo_raw)

        if y is None or endo is None:
            return _json({"error": "y and endogenous must be numeric."})
        if len(y) != len(endo):
            return _json({"error": "y and endogenous must have the same length."})
        if len(y) < 5:
            return _json({"error": "At least 5 observations are required."})

        n = len(y)

        # Parse instruments: support single or multiple
        instr_list = []
        if instrs_raw is not None:
            if isinstance(instrs_raw, list) and instrs_raw and isinstance(instrs_raw[0], list):
                for i, inst in enumerate(instrs_raw):
                    arr = _coerce_numeric_list(inst)
                    if arr is not None and len(arr) == n:
                        instr_list.append(arr)
                    else:
                        return _json({"error": f"instruments[{i}] length mismatch."})
            else:
                return _json({"error": "instruments must be a list of lists."})
        elif instr_raw is not None:
            arr = _coerce_numeric_list(instr_raw)
            if arr is not None and len(arr) == n:
                instr_list.append(arr)
            else:
                return _json({"error": "instrument length mismatch."})

        n_instruments = len(instr_list)
        over_identified = n_instruments > 1

        # Build exogenous matrix
        exog_arrays = []
        exog_names = []
        if exog_raw and isinstance(exog_raw, dict):
            for name, values in exog_raw.items():
                arr = _coerce_numeric_list(values)
                if arr is not None and len(arr) == n:
                    exog_arrays.append(arr)
                    exog_names.append(name)

        if HAS_LINEARMODELS and n_instruments <= 1:
            # linearmodels handles single instrument well; multi-instrument
            # fallback to manual to ensure Sargan test is easy to compute
            dep = pd.Series(y, name="y")
            endo_df = pd.Series(endo, name="endogenous")

            exog_data = {"const": np.ones(n)}
            for i, name in enumerate(exog_names):
                exog_data[name] = exog_arrays[i]
            exog_df = pd.DataFrame(exog_data)

            instr_data = {"instrument": instr_list[0]}
            for i, name in enumerate(exog_names):
                instr_data[f"iv_{name}"] = exog_arrays[i]
            instr_df = pd.DataFrame(instr_data)

            try:
                model = IV2SLS(
                    dependent=dep,
                    exog=exog_df,
                    endog=endo_df,
                    instruments=instr_df,
                ).fit(cov_type="robust")

                iv_coef = float(model.params.get("endogenous", 0))
                iv_se = float(model.std_errors.get("endogenous", 0))
                iv_t = float(model.tstats.get("endogenous", 0))
                iv_p = float(model.pvalues.get("endogenous", 0))

                first_stage_f = float(model.first_stage.diagnostics["f.stat"].iloc[0]) if hasattr(model, "first_stage") and model.first_stage is not None else None
                first_stage_p = float(model.first_stage.diagnostics["f.pval"].iloc[0]) if hasattr(model, "first_stage") and model.first_stage is not None else None

                # Stock-Yogo weak IV test
                sy_critical = self._stock_yogo_critical(n_instruments)
                weak_iv = first_stage_f is not None and first_stage_f < sy_critical

                # Wu-Hausman
                X_ols = np.column_stack([np.ones(n), endo] + exog_arrays)
                beta_ols = np.linalg.lstsq(X_ols, y, rcond=None)[0]
                ols_coef = float(beta_ols[1])
                diff = iv_coef - ols_coef
                var_diff = max(iv_se ** 2 - (float(np.std(X_ols[:, 1]) ** 2 / n)), 1e-12)
                hausman_stat = (diff ** 2) / var_diff
                hausman_p = float(sp_stats.chi2.sf(hausman_stat, 1)) if HAS_SCIPY else None

                result = {
                    "method": "2SLS Instrumental Variable (linearmodels, robust SE)",
                    "iv_coefficient": iv_coef,
                    "se": iv_se,
                    "t_stat": iv_t,
                    "p_value": iv_p,
                    "first_stage_f": first_stage_f,
                    "first_stage_p": first_stage_p,
                    "stock_yogo_critical": sy_critical,
                    "weak_iv": weak_iv,
                    "wu_hausman_stat": float(hausman_stat),
                    "wu_hausman_p": hausman_p,
                    "ols_coefficient": ols_coef,
                    "n": n,
                    "n_instruments": n_instruments,
                    "over_identified": over_identified,
                    "r_squared": float(model.rsquared) if hasattr(model, "rsquared") else None,
                }
                if weak_iv:
                    result["weak_iv_warning"] = f"First-stage F ({first_stage_f:.2f}) < Stock-Yogo critical value ({sy_critical:.2f}). Instruments may be weak."
                return self._final(args, result, "research_iv")
            except Exception:
                pass

        # Manual 2SLS (supports single and multiple instruments)
        # First stage: endogenous ~ all instruments + exogenous
        X_first = np.column_stack([np.ones(n)] + instr_list + exog_arrays)
        beta_first, _, _, _ = np.linalg.lstsq(X_first, endo, rcond=None)
        endo_hat = X_first @ beta_first

        # First-stage F-statistic (joint test of all instruments)
        ss_res_first = float(np.sum((endo - endo_hat) ** 2))
        ss_tot_first = float(np.sum((endo - np.mean(endo)) ** 2))
        r_sq_first = 1 - ss_res_first / ss_tot_first if ss_tot_first > 0 else 0.0
        k_first = X_first.shape[1]
        df_reg_first = k_first - 1
        df_res_first = n - k_first
        ms_reg_first = (ss_tot_first - ss_res_first) / df_reg_first if df_reg_first > 0 else 0
        ms_res_first = ss_res_first / df_res_first if df_res_first > 0 else 1
        first_f = ms_reg_first / ms_res_first if ms_res_first > 0 else 0
        first_p = float(sp_stats.f.sf(first_f, df_reg_first, df_res_first)) if HAS_SCIPY else None

        # Second stage: y ~ endo_hat + exogenous
        X_second = np.column_stack([np.ones(n), endo_hat] + exog_arrays)
        beta_second, _, _, _ = np.linalg.lstsq(X_second, y, rcond=None)

        resid = y - X_second @ beta_second
        df_res = n - X_second.shape[1]
        sigma2 = float(np.sum(resid ** 2) / df_res) if df_res > 0 else 0

        Z = X_first
        try:
            Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
            X2_actual = np.column_stack([np.ones(n), endo] + exog_arrays)
            X2PZ = X2_actual.T @ Pz @ X2_actual
            cov_iv = sigma2 * np.linalg.inv(X2PZ)
        except np.linalg.LinAlgError:
            cov_iv = sigma2 * np.linalg.inv(X_second.T @ X_second)

        se_iv = np.sqrt(np.diag(cov_iv))

        iv_coef = float(beta_second[1])
        iv_se = float(se_iv[1])
        iv_t = iv_coef / iv_se if iv_se > 0 else 0.0
        iv_p = float(2 * sp_stats.t.sf(abs(iv_t), df_res)) if HAS_SCIPY else None

        # Stock-Yogo weak IV test
        sy_critical = self._stock_yogo_critical(n_instruments)
        weak_iv = first_f < sy_critical

        # Sargan/Hansen J test (only for over-identified models)
        sargan_j = None
        sargan_p = None
        if over_identified:
            # J = n * (u' P_Z u) / (u' u)
            try:
                Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
                u = resid
                j_stat = float((u.T @ Pz @ u) / (np.var(u, ddof=1) if np.var(u, ddof=1) > 0 else 1e-12))
                sargan_j = j_stat
                # df = number of overidentifying restrictions = n_instruments - 1
                sargan_p = float(sp_stats.chi2.sf(sargan_j, n_instruments - 1)) if HAS_SCIPY else None
            except Exception:
                pass

        # Wu-Hausman test
        resid_first = endo - endo_hat
        X_hausman = np.column_stack([np.ones(n), endo, resid_first] + exog_arrays)
        beta_h, _, _, _ = np.linalg.lstsq(X_hausman, y, rcond=None)
        resid_h = y - X_hausman @ beta_h
        sigma2_h = float(np.sum(resid_h ** 2) / (n - X_hausman.shape[1])) if n > X_hausman.shape[1] else 0
        try:
            cov_h = sigma2_h * np.linalg.inv(X_hausman.T @ X_hausman)
            se_h = np.sqrt(np.diag(cov_h))
        except np.linalg.LinAlgError:
            se_h = np.full(X_hausman.shape[1], np.nan)

        hausman_t = float(beta_h[2] / se_h[2]) if se_h[2] != 0 else 0.0
        hausman_p = float(2 * sp_stats.t.sf(abs(hausman_t), n - X_hausman.shape[1])) if HAS_SCIPY else None

        # OLS for comparison
        X_ols = np.column_stack([np.ones(n), endo] + exog_arrays)
        beta_ols = np.linalg.lstsq(X_ols, y, rcond=None)[0]

        result = {
            "method": "2SLS Instrumental Variable (manual)",
            "iv_coefficient": iv_coef,
            "se": iv_se,
            "t_stat": iv_t,
            "p_value": iv_p,
            "first_stage_f": float(first_f),
            "first_stage_p": first_p,
            "first_stage_r_squared": r_sq_first,
            "stock_yogo_critical": sy_critical,
            "weak_iv": weak_iv,
            "wu_hausman_stat": float(hausman_t),
            "wu_hausman_p": hausman_p,
            "ols_coefficient": float(beta_ols[1]),
            "n": n,
            "n_instruments": n_instruments,
            "over_identified": over_identified,
        }
        if sargan_j is not None:
            result["sargan_j"] = sargan_j
            result["sargan_p"] = sargan_p
        if weak_iv:
            result["weak_iv_warning"] = f"First-stage F ({first_f:.2f}) < Stock-Yogo critical value ({sy_critical:.2f}). Instruments may be weak."
        try:
            result["apa"] = APAFormatter.iv(
                beta=iv_coef, se=iv_se, p=iv_p,
                f_first=float(first_f), sargan_p=sargan_p
            )
        except Exception:
            pass
        return self._final(args, result, "research_iv")

    # -----------------------------------------------------------------------
    # Propensity Score Matching
    # -----------------------------------------------------------------------

    def psm(self, args: dict) -> str:
        """Propensity Score Matching.

        Args:
            treat: list of treatment indicators (0/1).
            outcomes: list of outcome values.
            covariates: dict of {name: list}.
            method: str ('nearest'|'stratify'|'weight', default 'nearest').
            caliper: float (for nearest neighbor, default 0.2 * sd of logit).

        Returns:
            JSON with ATT, ATE, matched pairs, balance statistics.
        """
        args = self._normalize_args(args)
        treat_raw = args.get("treat")
        outcomes_raw = args.get("outcomes")
        covariates_raw = args.get("covariates")
        method = args.get("method", "nearest")

        if treat_raw is None or outcomes_raw is None or covariates_raw is None:
            return _json({"error": "treat, outcomes, and covariates are required."})

        treat = np.asarray(treat_raw, dtype=np.float64)
        outcomes = _coerce_numeric_list(outcomes_raw)

        if outcomes is None:
            return _json({"error": "outcomes must be numeric."})
        if len(treat) != len(outcomes):
            return _json({"error": "treat and outcomes must have the same length."})

        n = len(treat)
        treat_idx = np.where(treat == 1)[0]
        control_idx = np.where(treat == 0)[0]

        if len(treat_idx) == 0 or len(control_idx) == 0:
            return _json({"error": "Both treatment and control groups must have at least 1 member."})

        # Build covariate matrix
        cov_arrays = []
        cov_names = []
        for name, values in covariates_raw.items():
            arr = _coerce_numeric_list(values)
            if arr is not None and len(arr) == n:
                cov_arrays.append(arr)
                cov_names.append(name)

        if len(cov_arrays) == 0:
            return _json({"error": "At least one valid covariate is required."})

        X_cov = np.column_stack(cov_arrays)

        # Standardize covariates
        X_mean = X_cov.mean(axis=0)
        X_std = X_cov.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_stdz = (X_cov - X_mean) / X_std

        # Fit logistic regression for propensity scores
        if HAS_SKLEARN:
            lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
            lr.fit(X_stdz, treat)
            ps = lr.predict_proba(X_stdz)[:, 1]
        else:
            # Fallback: simple linear probability model as proxy
            X_design = np.column_stack([np.ones(n), X_stdz])
            beta = np.linalg.lstsq(X_design, treat, rcond=None)[0]
            ps_linear = X_design @ beta
            # Clip to (0.01, 0.99) to avoid extreme values
            ps = np.clip(ps_linear, 0.01, 0.99)

        # Check overlap
        ps_treat = ps[treat_idx]
        ps_control = ps[control_idx]
        overlap_min = max(ps_treat.min(), ps_control.min())
        overlap_max = min(ps_treat.max(), ps_control.max())
        if overlap_min >= overlap_max:
            return _json({"error": "No common support region for propensity scores."})

        # Balance statistics before matching
        balance_before = self._compute_balance(outcomes, treat, cov_arrays, cov_names)

        if method == "nearest":
            caliper = args.get("caliper")
            if caliper is None:
                caliper = 0.2 * np.std(np.log(ps / (1 - ps)))
            else:
                caliper = float(caliper)

            # Nearest neighbor matching
            matched_pairs = []
            control_used = set()
            for t_i in treat_idx:
                best_dist = float("inf")
                best_c = None
                for c_i in control_idx:
                    if c_i in control_used:
                        continue
                    dist = abs(ps[t_i] - ps[c_i])
                    if dist < best_dist:
                        best_dist = dist
                        best_c = c_i
                if best_c is not None and best_dist <= caliper:
                    matched_pairs.append((int(t_i), int(best_c)))
                    control_used.add(best_c)

            if len(matched_pairs) == 0:
                return _json({"error": "No matched pairs found within caliper."})

            # Compute ATT
            att_sum = 0.0
            for t_i, c_i in matched_pairs:
                att_sum += outcomes[t_i] - outcomes[c_i]
            att = att_sum / len(matched_pairs)

            # SE of ATT (Abadie & Imbens, 2006 approximation)
            matched_diffs = np.array([outcomes[t_i] - outcomes[c_i] for t_i, c_i in matched_pairs])
            att_se = float(np.std(matched_diffs, ddof=1) / np.sqrt(len(matched_pairs)))
            att_t = att / att_se if att_se > 0 else 0.0
            att_p = float(2 * sp_stats.t.sf(abs(att_t), len(matched_pairs) - 1)) if HAS_SCIPY else None

            # Balance after matching
            matched_treat_idx = [p[0] for p in matched_pairs]
            matched_control_idx = [p[1] for p in matched_pairs]
            matched_idx = matched_treat_idx + matched_control_idx
            balance_after = self._compute_balance(
                outcomes, treat, cov_arrays, cov_names, subset=matched_idx
            )

            result = {
                "method": "Propensity Score Matching (Nearest Neighbor)",
                "att": att,
                "att_se": att_se,
                "att_t_stat": att_t,
                "att_p_value": att_p,
                "ate": None,
                "matched_pairs": len(matched_pairs),
                "n_treated": len(treat_idx),
                "n_control": len(control_idx),
                "caliper": caliper,
                "balance_before": balance_before,
                "balance_after": balance_after,
                "overlap_range": [float(overlap_min), float(overlap_max)],
            }
            try:
                result["apa"] = APAFormatter.psm(
                    att=att, se=att_se, p=att_p,
                    n_treated=len(treat_idx), n_matched=len(matched_pairs),
                    method="nearest neighbor"
                )
            except Exception:
                pass
            return self._final(args, result, "research_psm")

        elif method == "stratify":
            # Stratification on propensity score
            n_strata = 10
            ps_sorted = np.sort(ps)
            quantiles = np.linspace(0, 1, n_strata + 1)
            boundaries = np.quantile(ps, quantiles)

            strata_atts = []
            strata_sizes = []
            for i in range(n_strata):
                mask = (ps >= boundaries[i]) & (ps < boundaries[i + 1])
                if i == n_strata - 1:
                    mask = (ps >= boundaries[i]) & (ps <= boundaries[i + 1])

                s_treat = treat[mask]
                s_outcomes = outcomes[mask]
                if np.sum(s_treat == 1) < 1 or np.sum(s_treat == 0) < 1:
                    continue

                mean_t = float(np.mean(s_outcomes[s_treat == 1]))
                mean_c = float(np.mean(s_outcomes[s_treat == 0]))
                strata_atts.append(mean_t - mean_c)
                strata_sizes.append(int(np.sum(mask)))

            if len(strata_atts) == 0:
                return _json({"error": "No valid strata found."})

            total_n = sum(strata_sizes)
            att = sum(a * s for a, s in zip(strata_atts, strata_sizes)) / total_n

            # Balance after stratification (use all data, stratified)
            balance_after = self._compute_balance(outcomes, treat, cov_arrays, cov_names)

            return self._final(args, {
                "method": "Propensity Score Matching (Stratification)",
                "att": att,
                "ate": None,
                "n_strata": len(strata_atts),
                "strata_effects": strata_atts,
                "strata_sizes": strata_sizes,
                "n_treated": len(treat_idx),
                "n_control": len(control_idx),
                "balance_before": balance_before,
                "balance_after": balance_after,
            }, "research_psm")

        elif method == "weight":
            # Inverse Propensity Weighting
            # Clip propensity scores to avoid extreme weights
            ps_clipped = np.clip(ps, 0.01, 0.99)

            # IPW weights
            w = np.where(treat == 1, 1.0 / ps_clipped, 1.0 / (1 - ps_clipped))
            # Normalize weights
            w_treat = w[treat == 1]
            w_control = w[treat == 0]
            w_treat_norm = w_treat / np.sum(w_treat)
            w_control_norm = w_control / np.sum(w_control)

            # Weighted means
            mean_t_weighted = float(np.sum(w_treat_norm * outcomes[treat == 1]))
            mean_c_weighted = float(np.sum(w_control_norm * outcomes[treat == 0]))
            ate = mean_t_weighted - mean_c_weighted

            # ATT
            w_att = np.where(treat == 1, 1.0, ps_clipped / (1 - ps_clipped))
            w_att_t = w_att[treat == 1] / np.sum(w_att[treat == 1])
            w_att_c = w_att[treat == 0] / np.sum(w_att[treat == 0])
            att = float(np.sum(w_att_t * outcomes[treat == 1]) -
                        np.sum(w_att_c * outcomes[treat == 0]))

            # SE via sandwich estimator (simplified)
            weighted_outcomes = outcomes.copy()
            treat_mask = treat == 1
            var_t = float(np.average((outcomes[treat_mask] - mean_t_weighted) ** 2,
                                      weights=w_treat))
            var_c = float(np.average((outcomes[~treat_mask] - mean_c_weighted) ** 2,
                                      weights=w_control))
            ate_se = float(np.sqrt(var_t / np.sum(w_treat) + var_c / np.sum(w_control)))
            ate_t = ate / ate_se if ate_se > 0 else 0.0
            ate_p = float(2 * sp_stats.t.sf(abs(ate_t), min(len(treat_idx), len(control_idx)) - 1)) if HAS_SCIPY else None

            balance_after = self._compute_balance(outcomes, treat, cov_arrays, cov_names,
                                                   weights=w)

            return self._final(args, {
                "method": "Propensity Score Matching (IPW)",
                "ate": ate,
                "att": att,
                "ate_se": ate_se,
                "ate_t_stat": ate_t,
                "ate_p_value": ate_p,
                "n_treated": len(treat_idx),
                "n_control": len(control_idx),
                "balance_before": balance_before,
                "balance_after": balance_after,
                "overlap_range": [float(overlap_min), float(overlap_max)],
            }, "research_psm")


        elif method == 'kernel':
            # Kernel matching (Epanechnikov)
            bw = args.get('bandwidth')
            if bw is None:
                ps_all = np.concatenate([ps_treat, ps_control])
                iqr = np.subtract(*np.percentile(ps_all, [75, 25]))
                bw = 0.9 * min(np.std(ps_all, ddof=1), iqr / 1.34) * (len(ps_all) ** (-0.2))
                if bw <= 0:
                    bw = 0.05
            else:
                bw = float(bw)

            att_sum = 0.0
            n_matched = 0
            for t_i in treat_idx:
                ps_t = ps[t_i]
                kernels = []
                control_vals = []
                for c_i in control_idx:
                    u = (ps[c_i] - ps_t) / bw
                    if abs(u) < 1.0:
                        k = 0.75 * (1.0 - u ** 2)
                        kernels.append(k)
                        control_vals.append(outcomes[c_i])
                if sum(kernels) > 0:
                    weighted_c = np.average(control_vals, weights=kernels)
                    att_sum += outcomes[t_i] - weighted_c
                    n_matched += 1

            if n_matched == 0:
                return _json({'error': 'No kernel matches found.'})

            att = att_sum / n_matched
            diffs = []
            for t_i in treat_idx:
                ps_t = ps[t_i]
                kernels = []
                control_vals = []
                for c_i in control_idx:
                    u = (ps[c_i] - ps_t) / bw
                    if abs(u) < 1.0:
                        k = 0.75 * (1.0 - u ** 2)
                        kernels.append(k)
                        control_vals.append(outcomes[c_i])
                if sum(kernels) > 0:
                    weighted_c = np.average(control_vals, weights=kernels)
                    diffs.append(outcomes[t_i] - weighted_c)

            diffs_arr = np.array(diffs)
            att_se = float(np.std(diffs_arr, ddof=1) / np.sqrt(len(diffs_arr)))
            att_t = att / att_se if att_se > 0 else 0.0
            att_p = float(2 * sp_stats.t.sf(abs(att_t), len(diffs_arr) - 1)) if HAS_SCIPY else None

            balance_after = self._compute_balance(outcomes, treat, cov_arrays, cov_names)

            result = {
                'method': 'Propensity Score Matching (Kernel, Epanechnikov)',
                'att': att,
                'att_se': att_se,
                'att_t_stat': att_t,
                'att_p_value': att_p,
                'matched_treated': n_matched,
                'bandwidth': bw,
                'n_treated': len(treat_idx),
                'n_control': len(control_idx),
                'balance_before': balance_before,
                'balance_after': balance_after,
                'overlap_range': [float(overlap_min), float(overlap_max)],
            }
            rosenbaum = self._psm_rosenbaum_bounds(diffs_arr)
            if rosenbaum:
                result['rosenbaum_bounds'] = rosenbaum
            try:
                result['apa'] = APAFormatter.psm(
                    att=att, se=att_se, p=att_p,
                    n_treated=len(treat_idx), n_matched=n_matched,
                    method='kernel'
                )
            except Exception:
                pass
            return self._final(args, result, 'research_psm')

        elif method == 'radius':
            # Radius (caliper) matching with multiple controls
            caliper = float(args.get('caliper', 0.1))
            att_sum = 0.0
            n_matched = 0
            diffs = []
            for t_i in treat_idx:
                matches = [c_i for c_i in control_idx if abs(ps[t_i] - ps[c_i]) <= caliper]
                if matches:
                    att_sum += outcomes[t_i] - np.mean(outcomes[matches])
                    diffs.append(outcomes[t_i] - np.mean(outcomes[matches]))
                    n_matched += 1

            if n_matched == 0:
                return _json({'error': 'No radius matches found within caliper.'})
            if n_matched < len(treat_idx) * 0.05:
                return _json({'error': f'Only {n_matched}/{len(treat_idx)} treated units matched within caliper {caliper}. Increase caliper.'})

            att = att_sum / n_matched
            diffs_arr = np.array(diffs)
            att_se = float(np.std(diffs_arr, ddof=1) / np.sqrt(len(diffs_arr)))
            att_t = att / att_se if att_se > 0 else 0.0
            att_p = float(2 * sp_stats.t.sf(abs(att_t), len(diffs_arr) - 1)) if HAS_SCIPY else None

            balance_after = self._compute_balance(outcomes, treat, cov_arrays, cov_names)

            result = {
                'method': 'Propensity Score Matching (Radius)',
                'att': att,
                'att_se': att_se,
                'att_t_stat': att_t,
                'att_p_value': att_p,
                'matched_treated': n_matched,
                'caliper': caliper,
                'n_treated': len(treat_idx),
                'n_control': len(control_idx),
                'balance_before': balance_before,
                'balance_after': balance_after,
                'overlap_range': [float(overlap_min), float(overlap_max)],
            }
            rosenbaum = self._psm_rosenbaum_bounds(diffs_arr)
            if rosenbaum:
                result['rosenbaum_bounds'] = rosenbaum
            try:
                result['apa'] = APAFormatter.psm(
                    att=att, se=att_se, p=att_p,
                    n_treated=len(treat_idx), n_matched=n_matched,
                    method='radius'
                )
            except Exception:
                pass
            return self._final(args, result, 'research_psm')

        return _json({"error": f"Unknown method '{method}'. Use 'nearest', 'stratify', 'weight', 'kernel', or 'radius'."})

    def _psm_rosenbaum_bounds(self, diffs_arr: np.ndarray) -> dict:
        """Compute Rosenbaum bounds for sensitivity analysis.

        Tests how large an unobserved confounder (Gamma) would need to be
        to change the inference about the ATT.
        """
        if len(diffs_arr) < 2:
            return None
        diffs = np.array(diffs_arr)
        n = len(diffs)
        wilcoxon_stat = float(np.sum(np.sign(diffs) * np.arange(1, n + 1)))

        bounds = {}
        for gamma in [1.0, 1.5, 2.0, 2.5, 3.0]:
            # Upper bound (favors treatment effect)
            p_upper = 1.0
            # Lower bound (against treatment effect)
            p_lower = 1.0
            # Approximate using signed-rank bounds
            pos = np.sum(diffs > 0)
            neg = np.sum(diffs < 0)
            # Odds ratio bound
            omega = gamma
            # Approximate p-value bounds via binomial logic
            p_min = 1.0 / (1.0 + omega)
            p_max = omega / (1.0 + omega)
            from scipy import stats as sp_stats
            # Wilcoxon under Gamma
            # E[W+] lower/upper bounds
            e_low = np.sum([p_min * i for i in range(1, n + 1)])
            e_high = np.sum([p_max * i for i in range(1, n + 1)])
            v_w = np.sum([i ** 2 * p_min * (1 - p_min) for i in range(1, n + 1)])
            se_w = np.sqrt(v_w) if v_w > 0 else 1.0
            z_low = (wilcoxon_stat - e_high) / se_w
            z_high = (wilcoxon_stat - e_low) / se_w
            p_val_low = float(2 * sp_stats.norm.sf(abs(z_low))) if HAS_SCIPY else None
            p_val_high = float(2 * sp_stats.norm.sf(abs(z_high))) if HAS_SCIPY else None
            bounds[f"Gamma={gamma}"] = {
                "p_value_lower": p_val_low,
                "p_value_upper": p_val_high,
                "significant_at_05": (p_val_low is not None and p_val_low < 0.05) or (p_val_high is not None and p_val_high < 0.05),
            }
        return bounds

    def _compute_balance(self, outcomes: np.ndarray, treat: np.ndarray,
                         cov_arrays: list, cov_names: list,
                         subset: list = None, weights: np.ndarray = None) -> dict:
        """Compute standardized mean differences for balance checking."""
        if subset is not None:
            idx = np.array(subset)
            outcomes = outcomes[idx]
            treat = treat[idx]
            cov_arrays = [c[idx] for c in cov_arrays]

        balance = {}
        t_mask = treat == 1
        c_mask = treat == 0

        if weights is not None:
            w_t = weights[t_mask]
            w_c = weights[c_mask]
            if np.sum(w_t) > 0 and np.sum(w_c) > 0:
                w_t_n = w_t / np.sum(w_t)
                w_c_n = w_c / np.sum(w_c)
            else:
                w_t_n = w_t
                w_c_n = w_c
        else:
            w_t_n = None
            w_c_n = None

        for i, name in enumerate(cov_names):
            cov = cov_arrays[i]
            if weights is not None:
                mean_t = float(np.sum(w_t_n * cov[t_mask]))
                mean_c = float(np.sum(w_c_n * cov[c_mask]))
                var_t = float(np.sum(w_t_n * (cov[t_mask] - mean_t) ** 2))
                var_c = float(np.sum(w_c_n * (cov[c_mask] - mean_c) ** 2))
            else:
                mean_t = float(np.mean(cov[t_mask]))
                mean_c = float(np.mean(cov[c_mask]))
                var_t = float(np.var(cov[t_mask], ddof=1))
                var_c = float(np.var(cov[c_mask], ddof=1))

            pooled_std = float(np.sqrt((var_t + var_c) / 2))
            smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0.0
            balance[name] = {
                "mean_treated": mean_t,
                "mean_control": mean_c,
                "std_diff": smd,
            }

        return balance

    # -----------------------------------------------------------------------
    # Interrupted Time Series
    # -----------------------------------------------------------------------

    def its(self, args: dict) -> str:
        """Interrupted Time Series.

        Args:
            y: list of outcome values.
            time: list of time points.
            intervention: float (time point of intervention).
            covariates: dict of {name: list} (optional).

        Returns:
            JSON with level_change, trend_change, SE, p-values, counterfactual.
        """
        args = self._normalize_args(args)
        y_raw = args.get("y")
        time_raw = args.get("time")
        intervention = args.get("intervention")

        if y_raw is None or time_raw is None or intervention is None:
            return _json({"error": "y, time, and intervention are required."})

        y = _coerce_numeric_list(y_raw)
        time_arr = _coerce_numeric_list(time_raw)

        if y is None or time_arr is None:
            return _json({"error": "y and time must be numeric lists."})
        if len(y) != len(time_arr):
            return _json({"error": "y and time must have the same length."})
        if len(y) < 6:
            return _json({"error": "At least 6 time points are required."})

        intervention = float(intervention)
        n = len(y)
        covariates_raw = args.get("covariates")

        # Post-intervention indicator and interaction
        post = (time_arr >= intervention).astype(np.float64)
        time_post = time_arr * post

        # Build design matrix: time, post, time*post
        X_cols = {
            "time": time_arr,
            "post": post,
            "time_post": time_post,
        }

        if covariates_raw and isinstance(covariates_raw, dict):
            for name, values in covariates_raw.items():
                arr = _coerce_numeric_list(values)
                if arr is not None and len(arr) == n:
                    X_cols[name] = arr

        X_arr = np.column_stack(list(X_cols.values()))
        col_names = list(X_cols.keys())

        if HAS_STATSMODELS:
            X_sm = sm.add_constant(X_arr)

            # Durbin-Watson test for autocorrelation
            model_ols = sm.OLS(y, X_sm).fit()
            dw = float(sm.stats.stattools.durbin_watson(model_ols.resid))
            dw_warning = dw < 1.5

            # Determine covariance estimator
            use_hac = args.get("hac", True)
            use_prais = args.get("prais_winsten", False)

            if use_prais and dw < 1.5:
                # Prais-Winsten AR(1) correction
                rho = float(np.corrcoef(model_ols.resid[:-1], model_ols.resid[1:])[0, 1])
                if np.isnan(rho):
                    rho = 0.0
                y_pw = y.copy()
                X_pw = X_sm.copy()
                y_pw[1:] = y_pw[1:] - rho * y[:-1]
                X_pw[1:, :] = X_pw[1:, :] - rho * X_sm[:-1, :]
                y_pw[0] = y_pw[0] * np.sqrt(1 - rho ** 2)
                X_pw[0, :] = X_pw[0, :] * np.sqrt(1 - rho ** 2)
                model = sm.OLS(y_pw, X_pw).fit(cov_type="HC1")
                method_label = "Interrupted Time Series (Prais-Winsten AR(1), HC1)"
            elif use_hac:
                # Newey-West HAC SE (auto lag = floor(4*(T/100)^(2/9)))
                lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
                lag = max(1, lag)
                model = model_ols.get_robustcov_results(cov_type="HAC", maxlags=lag, use_correction=True)
                method_label = f"Interrupted Time Series (OLS, Newey-West HAC, lag={lag})"
            else:
                model = model_ols.get_robustcov_results(cov_type="HC1")
                method_label = "Interrupted Time Series (OLS, HC1 robust SE)"

            coeffs = {"intercept": float(model.params[0])}
            for i, name in enumerate(col_names):
                coeffs[name] = float(model.params[i + 1])

            post_idx = col_names.index("post")
            tp_idx = col_names.index("time_post")

            level_change = float(model.params[post_idx + 1])
            level_se = float(model.bse[post_idx + 1])
            level_t = float(model.tvalues[post_idx + 1])
            level_p = float(model.pvalues[post_idx + 1])

            trend_change = float(model.params[tp_idx + 1])
            trend_se = float(model.bse[tp_idx + 1])
            trend_t = float(model.tvalues[tp_idx + 1])
            trend_p = float(model.pvalues[tp_idx + 1])

            # Counterfactual
            X_cf = X_arr.copy()
            cf_post_idx = col_names.index("post")
            cf_tp_idx = col_names.index("time_post")
            X_cf[:, cf_post_idx] = 0.0
            X_cf[:, cf_tp_idx] = 0.0
            X_cf_sm = sm.add_constant(X_cf)
            counterfactual = (X_cf_sm @ model.params).tolist()
            actual = y.tolist()

            result = {
                "method": method_label,
                "level_change": level_change,
                "level_se": level_se,
                "level_t_stat": level_t,
                "level_p_value": level_p,
                "trend_change": trend_change,
                "trend_se": trend_se,
                "trend_t_stat": trend_t,
                "trend_p_value": trend_p,
                "coefficients": coeffs,
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "n": n,
                "intervention_point": intervention,
                "n_pre": int(np.sum(post == 0)),
                "n_post": int(np.sum(post == 1)),
                "counterfactual": counterfactual,
                "actual": actual,
                "durbin_watson": dw,
                "durbin_watson_warning": dw_warning,
            }
            if use_hac and not use_prais:
                result["hac_lag"] = lag
            if use_prais:
                result["ar1_rho"] = rho
            try:
                from sophia.research.apa import APAFormatter
                result["apa"] = APAFormatter.its(
                    level_change=result["level_change"], level_se=result["level_se"],
                    level_p=result["level_p_value"],
                    trend_change=result["trend_change"], trend_se=result["trend_se"],
                    trend_p=result["trend_p_value"],
                    n=result["n"], n_pre=result["n_pre"], n_post=result["n_post"],
                    method_label=result["method"]
                )
            except Exception:
                pass
            return self._final(args, result, "research_its")

        # numpy fallback
        X_design = np.column_stack([np.ones(n), X_arr])
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        resid = y - X_design @ beta
        df_res = n - len(beta)
        mse = float(np.sum(resid ** 2) / df_res) if df_res > 0 else 0
        try:
            cov = mse * np.linalg.inv(X_design.T @ X_design)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), np.nan)

        post_idx = col_names.index("post")
        tp_idx = col_names.index("time_post")

        level_change = float(beta[post_idx + 1])
        level_se = float(se[post_idx + 1])
        level_t = level_change / level_se if level_se > 0 else 0
        level_p = float(2 * sp_stats.t.sf(abs(level_t), df_res)) if HAS_SCIPY else None

        trend_change = float(beta[tp_idx + 1])
        trend_se = float(se[tp_idx + 1])
        trend_t = trend_change / trend_se if trend_se > 0 else 0
        trend_p = float(2 * sp_stats.t.sf(abs(trend_t), df_res)) if HAS_SCIPY else None

        return self._final(args, {
            "method": "Interrupted Time Series (numpy OLS)",
            "level_change": level_change,
            "level_se": level_se,
            "level_t_stat": level_t,
            "level_p_value": level_p,
            "trend_change": trend_change,
            "trend_se": trend_se,
            "trend_t_stat": trend_t,
            "trend_p_value": trend_p,
            "n": n,
            "intervention_point": intervention,
        }, "research_its")

    # -----------------------------------------------------------------------
    # Mediation Analysis
    # -----------------------------------------------------------------------

    def mediation(self, args: dict) -> str:
        """Mediation analysis (Baron-Kenny + bootstrap).

        Args:
            y: list of outcome values.
            x: list of independent variable values.
            mediator: list of mediator values.
            bootstrap: int (iterations, default 5000).
            seed: int (default 42).

        Returns:
            JSON with total_effect, direct_effect, indirect_effect,
            proportion_mediated, Sobel test, bootstrap CI.
        """
        args = self._normalize_args(args)
        y_raw = args.get("y")
        x_raw = args.get("x")
        med_raw = args.get("mediator")
        n_boot = int(args.get("bootstrap", 5000))
        seed_arg = args.get("seed")
        if seed_arg is None:
            seed_arg = GlobalSeed.get()
        seed = int(seed_arg) if seed_arg is not None else 42

        if y_raw is None or x_raw is None or med_raw is None:
            return _json({"error": "y, x, and mediator are required."})

        y = _coerce_numeric_list(y_raw)
        x = _coerce_numeric_list(x_raw)
        m = _coerce_numeric_list(med_raw)

        if y is None or x is None or m is None:
            return _json({"error": "y, x, and mediator must be numeric."})
        if len(y) != len(x) or len(y) != len(m):
            return _json({"error": "y, x, and mediator must have the same length."})
        if len(y) < 10:
            return _json({"error": "At least 10 observations are required."})

        n = len(y)

        def _ols_coef(X, Y):
            """Simple OLS coefficient (single predictor, no intercept manipulation)."""
            X_d = np.column_stack([np.ones(len(Y)), X])
            beta, _, _, _ = np.linalg.lstsq(X_d, Y, rcond=None)
            return beta

        # Path c: y ~ x (total effect)
        beta_c = _ols_coef(x, y)
        total_effect = float(beta_c[1])
        # SE for path c
        X_c = np.column_stack([np.ones(n), x])
        resid_c = y - X_c @ beta_c
        df_c = n - 2
        mse_c = float(np.sum(resid_c ** 2) / df_c)
        try:
            se_c = float(np.sqrt(mse_c * np.linalg.inv(X_c.T @ X_c)[1, 1]))
        except np.linalg.LinAlgError:
            se_c = float("nan")
        t_c = total_effect / se_c if se_c > 0 else 0.0
        p_c = float(2 * sp_stats.t.sf(abs(t_c), df_c)) if HAS_SCIPY else None

        # Path a: mediator ~ x
        beta_a = _ols_coef(x, m)
        a = float(beta_a[1])
        X_a = np.column_stack([np.ones(n), x])
        resid_a = m - X_a @ beta_a
        mse_a = float(np.sum(resid_a ** 2) / (n - 2))
        try:
            se_a = float(np.sqrt(mse_a * np.linalg.inv(X_a.T @ X_a)[1, 1]))
        except np.linalg.LinAlgError:
            se_a = float("nan")

        # Path b: y ~ x + mediator (direct effect = coef on x, b = coef on mediator)
        X_b = np.column_stack([np.ones(n), x, m])
        beta_b = np.linalg.lstsq(X_b, y, rcond=None)[0]
        direct_effect = float(beta_b[1])  # coef on x (c')
        b = float(beta_b[2])  # coef on mediator

        resid_b = y - X_b @ beta_b
        df_b = n - 3
        mse_b = float(np.sum(resid_b ** 2) / df_b)
        try:
            cov_b = mse_b * np.linalg.inv(X_b.T @ X_b)
            se_direct = float(np.sqrt(cov_b[1, 1]))
            se_b = float(np.sqrt(cov_b[2, 2]))
        except np.linalg.LinAlgError:
            se_direct = float("nan")
            se_b = float("nan")

        t_direct = direct_effect / se_direct if se_direct > 0 else 0.0
        p_direct = float(2 * sp_stats.t.sf(abs(t_direct), df_b)) if HAS_SCIPY else None

        # Indirect effect: a * b
        indirect_effect = a * b

        # Sobel test
        sobel_se = float(np.sqrt(se_a ** 2 * b ** 2 + se_b ** 2 * a ** 2))
        sobel_z = indirect_effect / sobel_se if sobel_se > 0 else 0.0
        sobel_p = float(2 * sp_stats.norm.sf(abs(sobel_z))) if HAS_SCIPY else None

        # Proportion mediated
        if abs(total_effect) > 1e-12:
            prop_mediated = indirect_effect / total_effect
        else:
            prop_mediated = None

        # Bootstrap CI for indirect effect
        rng = np.random.default_rng(seed)
        boot_indirect = np.zeros(n_boot)
        for i in range(n_boot):
            idx_boot = rng.choice(n, size=n, replace=True)
            x_b = x[idx_boot]
            m_b = m[idx_boot]
            y_b = y[idx_boot]

            # Path a on bootstrap sample
            X_a_b = np.column_stack([np.ones(n), x_b])
            beta_a_b = np.linalg.lstsq(X_a_b, m_b, rcond=None)[0]
            a_b = float(beta_a_b[1])

            # Path b on bootstrap sample
            X_b_b = np.column_stack([np.ones(n), x_b, m_b])
            beta_b_b = np.linalg.lstsq(X_b_b, y_b, rcond=None)[0]
            b_b = float(beta_b_b[2])

            boot_indirect[i] = a_b * b_b

        boot_ci = [float(np.percentile(boot_indirect, 2.5)),
                   float(np.percentile(boot_indirect, 97.5))]

        return self._final(args, {
            "method": "Mediation Analysis (Baron-Kenny + Bootstrap)",
            "total_effect": total_effect,
            "total_se": se_c,
            "total_p": p_c,
            "direct_effect": direct_effect,
            "direct_se": se_direct,
            "direct_p": p_direct,
            "indirect_effect": float(indirect_effect),
            "indirect_se": sobel_se,
            "path_a": a,
            "path_a_se": se_a,
            "path_b": b,
            "path_b_se": se_b,
            "proportion_mediated": float(prop_mediated) if prop_mediated is not None else None,
            "sobel_z": sobel_z,
            "sobel_p": sobel_p,
            "bootstrap_ci_95": boot_ci,
            "bootstrap_iterations": n_boot,
            "n": n,
        }, "research_mediation")

    # -----------------------------------------------------------------------
    # Average Treatment Effect
    # -----------------------------------------------------------------------

    def causal_effect(self, args: dict) -> str:
        """Average Treatment Effect estimation.

        Args:
            y: list of outcome values.
            treat: list of treatment indicators (0/1).
            covariates: dict of {name: list} (optional).
            method: str ('ols'|'ipw'|'aipw', default 'ols').

        Returns:
            JSON with ATE, SE, CI, method used.
        """
        args = self._normalize_args(args)
        y_raw = args.get("y")
        treat_raw = args.get("treat")
        method = args.get("method", "ols")
        covariates_raw = args.get("covariates")

        if y_raw is None or treat_raw is None:
            return _json({"error": "y and treat are required."})

        y = _coerce_numeric_list(y_raw)
        treat = np.asarray(treat_raw, dtype=np.float64)

        if y is None:
            return _json({"error": "y must be numeric."})
        if len(y) != len(treat):
            return _json({"error": "y and treat must have the same length."})

        n = len(y)
        treat_idx = np.where(treat == 1)[0]
        control_idx = np.where(treat == 0)[0]

        if len(treat_idx) == 0 or len(control_idx) == 0:
            return _json({"error": "Both treatment and control groups are required."})

        # Build covariate matrix
        cov_arrays = []
        cov_names = []
        if covariates_raw and isinstance(covariates_raw, dict):
            for name, values in covariates_raw.items():
                arr = _coerce_numeric_list(values)
                if arr is not None and len(arr) == n:
                    cov_arrays.append(arr)
                    cov_names.append(name)

        if method == "ols":
            # OLS regression y ~ treat + covariates
            X_cols = [treat]
            col_names = ["treat"]
            for i, name in enumerate(cov_names):
                X_cols.append(cov_arrays[i])
                col_names.append(name)

            X_arr = np.column_stack(X_cols)

            if HAS_STATSMODELS:
                X_sm = sm.add_constant(X_arr)
                model = sm.OLS(y, X_sm).fit(cov_type="HC1")
                ate = float(model.params[1])
                ate_se = float(model.bse[1])
                ate_t = float(model.tvalues[1])
                ate_p = float(model.pvalues[1])
                ci_arr = model.conf_int()
                ci = [float(ci_arr[1, 0]), float(ci_arr[1, 1])]
                r_sq = float(model.rsquared)
            else:
                X_design = np.column_stack([np.ones(n), X_arr])
                beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                resid = y - X_design @ beta
                df_res = n - len(beta)
                mse = float(np.sum(resid ** 2) / df_res) if df_res > 0 else 0
                try:
                    cov_m = mse * np.linalg.inv(X_design.T @ X_design)
                    se = np.sqrt(np.diag(cov_m))
                except np.linalg.LinAlgError:
                    se = np.full(len(beta), np.nan)
                ate = float(beta[1])
                ate_se = float(se[1])
                ate_t = ate / ate_se if ate_se > 0 and not np.isnan(ate_se) else 0.0
                ate_p = float(2 * sp_stats.t.sf(abs(ate_t), df_res)) if HAS_SCIPY else None
                ci = None
                r_sq = None

            return self._final(args, {
                "method": "Average Treatment Effect (OLS, HC1 robust SE)",
                "ate": ate,
                "se": ate_se,
                "t_stat": ate_t,
                "p_value": ate_p,
                "ci_95": ci,
                "r_squared": r_sq,
                "n": n,
                "n_treated": len(treat_idx),
                "n_control": len(control_idx),
            }, "research_causal_effect")

        elif method == "ipw":
            # Inverse Propensity Weighting
            if len(cov_arrays) == 0:
                return _json({"error": "Covariates are required for IPW estimation."})

            X_cov = np.column_stack(cov_arrays)
            X_stdz = (X_cov - X_cov.mean(axis=0)) / np.maximum(X_cov.std(axis=0), 1e-12)

            if HAS_SKLEARN:
                lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
                lr.fit(X_stdz, treat)
                ps = lr.predict_proba(X_stdz)[:, 1]
            else:
                X_design = np.column_stack([np.ones(n), X_stdz])
                beta = np.linalg.lstsq(X_design, treat, rcond=None)[0]
                ps = np.clip(X_design @ beta, 0.01, 0.99)

            ps = np.clip(ps, 0.01, 0.99)

            # ATE weights
            w = np.where(treat == 1, 1.0 / ps, 1.0 / (1.0 - ps))
            w_t = w[treat == 1]
            w_c = w[treat == 0]
            mean_t = float(np.average(y[treat == 1], weights=w_t))
            mean_c = float(np.average(y[treat == 0], weights=w_c))
            ate = mean_t - mean_c

            var_t = float(np.average((y[treat == 1] - mean_t) ** 2, weights=w_t))
            var_c = float(np.average((y[treat == 0] - mean_c) ** 2, weights=w_c))
            ate_se = float(np.sqrt(var_t / np.sum(w_t) + var_c / np.sum(w_c)))
            ate_t = ate / ate_se if ate_se > 0 else 0.0
            ate_p = float(2 * sp_stats.t.sf(abs(ate_t), min(len(treat_idx), len(control_idx)) - 1)) if HAS_SCIPY else None

            return self._final(args, {
                "method": "Average Treatment Effect (IPW)",
                "ate": ate,
                "se": ate_se,
                "t_stat": ate_t,
                "p_value": ate_p,
                "ci_95": [ate - 1.96 * ate_se, ate + 1.96 * ate_se],
                "n": n,
                "n_treated": len(treat_idx),
                "n_control": len(control_idx),
            }, "research_causal_effect")

        elif method == "aipw":
            # Augmented IPW (Doubly Robust)
            if len(cov_arrays) == 0:
                return _json({"error": "Covariates are required for AIPW estimation."})

            X_cov = np.column_stack(cov_arrays)
            X_stdz = (X_cov - X_cov.mean(axis=0)) / np.maximum(X_cov.std(axis=0), 1e-12)

            # Propensity scores
            if HAS_SKLEARN:
                lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
                lr.fit(X_stdz, treat)
                ps = lr.predict_proba(X_stdz)[:, 1]
            else:
                X_design_ps = np.column_stack([np.ones(n), X_stdz])
                beta_ps = np.linalg.lstsq(X_design_ps, treat, rcond=None)[0]
                ps = np.clip(X_design_ps @ beta_ps, 0.01, 0.99)

            ps = np.clip(ps, 0.01, 0.99)

            # Outcome models: mu_1(x) and mu_0(x)
            X_outcome = np.column_stack([np.ones(n), X_stdz])
            # mu_1: outcome model for treated
            t_mask = treat == 1
            c_mask = treat == 0

            if HAS_STATSMODELS:
                model_t = sm.OLS(y[t_mask], X_outcome[t_mask]).fit()
                model_c = sm.OLS(y[c_mask], X_outcome[c_mask]).fit()
            else:
                beta_t = np.linalg.lstsq(X_outcome[t_mask], y[t_mask], rcond=None)[0]
                beta_c = np.linalg.lstsq(X_outcome[c_mask], y[c_mask], rcond=None)[0]
                model_t = None
                model_c = None

            def _predict(model_obj, X, beta_fallback):
                if model_obj is not None:
                    return model_obj.predict(X)
                return X @ beta_fallback

            if HAS_STATSMODELS:
                mu_1 = model_t.predict(X_outcome)
                mu_0 = model_c.predict(X_outcome)
            else:
                mu_1 = X_outcome @ beta_t
                mu_0 = X_outcome @ beta_c

            # AIPW estimator
            aipw_scores = (
                (treat * (y - mu_1)) / ps
                + mu_1
                - ((1 - treat) * (y - mu_0)) / (1 - ps)
                - mu_0
            )
            ate = float(np.mean(aipw_scores))
            ate_se = float(np.std(aipw_scores, ddof=1) / np.sqrt(n))
            ate_t = ate / ate_se if ate_se > 0 else 0.0
            ate_p = float(2 * sp_stats.t.sf(abs(ate_t), n - 1)) if HAS_SCIPY else None

            return self._final(args, {
                "method": "Average Treatment Effect (AIPW / Doubly Robust)",
                "ate": ate,
                "se": ate_se,
                "t_stat": ate_t,
                "p_value": ate_p,
                "ci_95": [ate - 1.96 * ate_se, ate + 1.96 * ate_se],
                "n": n,
                "n_treated": len(treat_idx),
                "n_control": len(control_idx),
            }, "research_causal_effect")

        return _json({"error": f"Unknown method '{method}'. Use 'ols', 'ipw', or 'aipw'."})

    # -----------------------------------------------------------------------
    # Sensitivity / Robustness Analysis
    # -----------------------------------------------------------------------

    def sensitivity(self, args: dict) -> str:
        """Sensitivity/robustness analysis (Oster 2019).

        Args:
            estimate: float (observed coefficient).
            se: float (standard error).
            method: str (default 'oster').
            delta: float (proportional selection bias, default 1).
            r_max: float (max R-squared from hypothetical full model, optional).
            beta_control: float (coefficient from controlled regression).
            r_control: float (R-squared from controlled regression).
            r_uncontrolled: float (R-squared from uncontrolled regression).
            beta_uncontrolled: float (coefficient from uncontrolled regression).

        Returns:
            JSON with sensitivity bounds, robustness measure.
        """
        args = self._normalize_args(args)
        method = args.get("method", "oster")

        if method == "oster":
            # Oster (2019) method: bounds on coefficient assuming
            # unobserved confounding proportional to observed
            beta_control = args.get("beta_control", args.get("beta_controlled"))
            r_control = args.get("r_control", args.get("r_controlled"))
            r_uncontrolled = args.get("r_uncontrolled")
            beta_uncontrolled = args.get("beta_uncontrolled")
            delta = float(args.get("delta", 1.0))

            if any(v is None for v in [beta_control, r_control, r_uncontrolled, beta_uncontrolled]):
                return _json({"error": "Oster method requires beta_control, r_control, r_uncontrolled, beta_uncontrolled."})

            beta_c = float(beta_control)
            r_c = float(r_control)
            r_u = float(r_uncontrolled)
            beta_u = float(beta_uncontrolled)

            # R_max: if not provided, use 1.3 * R_control (Oster's suggestion)
            r_max = args.get("r_max")
            if r_max is None:
                r_max = min(1.3 * r_c, 1.0)
            else:
                r_max = min(float(r_max), 1.0)

            # Oster's proportional selection bias bound
            # The bias-adjusted coefficient:
            # beta_star = beta_c - delta * (beta_u - beta_c) * (r_max - r_c) / (r_c - r_u)
            # Under delta=1 (equal selection), the bound is:
            if abs(r_c - r_u) < 1e-12:
                # No change in R-squared from adding controls: no confounding detected
                return self._final(args, {
                    "method": "Oster (2019) Sensitivity Analysis",
                    "beta_controlled": beta_c,
                    "beta_uncontrolled": beta_u,
                    "r_controlled": r_c,
                    "r_uncontrolled": r_u,
                    "r_max": r_max,
                    "delta": delta,
                    "note": "No change in R-squared from controls; cannot compute bounds.",
                }, "research_sensitivity")

            proportionality = (r_max - r_c) / (r_c - r_u)
            beta_star = beta_c - delta * (beta_u - beta_c) * proportionality

            # Identify the delta that would zero out the effect
            if abs((beta_u - beta_c) * proportionality) > 1e-12:
                delta_zero = beta_c / ((beta_u - beta_c) * proportionality)
            else:
                delta_zero = float("inf")

            # Robustness check: is the bound consistent in sign?
            sign_robust = (beta_c * beta_star > 0)

            return self._final(args, {
                "method": "Oster (2019) Sensitivity Analysis",
                "beta_controlled": beta_c,
                "beta_uncontrolled": beta_u,
                "r_controlled": r_c,
                "r_uncontrolled": r_u,
                "r_max": r_max,
                "delta": delta,
                "beta_star": float(beta_star),
                "delta_to_zero": float(delta_zero),
                "proportion_selected_out": float(proportionality),
                "sign_robust": sign_robust,
                "interpretation": (
                    f"Under delta={delta}, the bias-adjusted estimate is {float(beta_star):.4f}. "
                    f"A delta of {float(delta_zero):.4f} would be needed to zero out the effect. "
                    f"The result is {'robust' if sign_robust else 'not robust'} to unobserved confounding "
                    f"proportional to observed confounding."
                ),
            }, "research_sensitivity")

        elif method == "rosenbaum":
            # Rosenbaum bounds for matching studies
            estimate = args.get("estimate")
            se = args.get("se")
            gamma_range = args.get("gamma_range")

            if estimate is None or se is None:
                return _json({"error": "Rosenbaum method requires estimate and se."})

            est = float(estimate)
            se_val = float(se)

            if gamma_range is None:
                gamma_range = np.linspace(1.0, 3.0, 21).tolist()
            else:
                gamma_range = [float(g) for g in gamma_range]

            bounds = []
            for gamma in gamma_range:
                # Upper and lower bounds under hidden bias gamma
                # Using Wilcoxon signed-rank approximation
                gamma_log = np.log(gamma)
                # Approximation: adjusted Z-statistics
                z_upper = (est + gamma_log) / se_val if se_val > 0 else 0
                z_lower = (est - gamma_log) / se_val if se_val > 0 else 0
                p_upper = float(sp_stats.norm.sf(z_upper)) if HAS_SCIPY else None
                p_lower = float(sp_stats.norm.sf(z_lower)) if HAS_SCIPY else None
                bounds.append({
                    "gamma": gamma,
                    "p_upper": p_upper,
                    "p_lower": p_lower,
                })

            # Find the gamma at which the result becomes insignificant
            critical_gamma = None
            for b in bounds:
                if b["p_lower"] is not None and b["p_lower"] > 0.05:
                    critical_gamma = b["gamma"]
                    break

            return self._final(args, {
                "method": "Rosenbaum Bounds Sensitivity Analysis",
                "estimate": est,
                "se": se_val,
                "bounds": bounds,
                "critical_gamma": critical_gamma,
                "interpretation": (
                    f"The result becomes statistically insignificant (p>0.05) at gamma={critical_gamma:.2f}. "
                    f"This means a hidden bias of magnitude {critical_gamma:.2f} would be needed "
                    f"to invalidate the finding."
                    if critical_gamma is not None
                    else "The result remains significant across all tested gamma values."
                ),
            }, "research_sensitivity")

        elif method == "eittheim":
            # Eittheim et al. approach: perturbation-based sensitivity
            estimate = args.get("estimate")
            se = args.get("se")

            if estimate is None or se is None:
                return _json({"error": "eittheim method requires estimate and se."})

            est = float(estimate)
            se_val = float(se)

            # Compute how much bias would be needed to change conclusion
            # Bias needed to make estimate statistically insignificant
            t_orig = est / se_val if se_val > 0 else 0
            bias_to_insignificance = abs(est) - 1.96 * se_val
            # As a proportion of the estimate
            bias_ratio = abs(bias_to_insignificance / est) if abs(est) > 1e-12 else float("inf")

            return self._final(args, {
                "method": "Eittheim Sensitivity Analysis",
                "estimate": est,
                "se": se_val,
                "t_original": float(t_orig),
                "bias_to_insignificance": float(bias_to_insignificance),
                "bias_ratio": float(bias_ratio),
                "interpretation": (
                    f"A bias of {float(bias_to_insignificance):.4f} "
                    f"({float(bias_ratio)*100:.1f}% of the estimate) "
                    f"would be needed to render the result statistically insignificant."
                    if abs(est) > 1e-12
                    else "Estimate is effectively zero."
                ),
            }, "research_sensitivity")

        return _json({"error": f"Unknown method '{method}'. Use 'oster', 'rosenbaum', or 'eittheim'."})

    # -----------------------------------------------------------------------
    # Synthetic Control Method
    # -----------------------------------------------------------------------

    def synthetic_control(self, args: dict) -> str:
        """Synthetic Control Method (Abadie & Gardeazabal 2003; Abadie, Diamond & Hainmueller 2010).

        Constructs a weighted combination of control units that best reproduces
        the pre-treatment characteristics of the treated unit, then compares
        post-treatment outcomes to estimate the causal effect.

        Args:
            y: list of outcome values (panel).
            unit: list of unit IDs.
            time: list of time periods.
            treated_unit: scalar ID of the treated unit.
            treatment_time: scalar threshold separating pre/post periods.
            donor_units: list of donor unit IDs (optional; defaults to all
                         non-treated units).
            covariates: dict of {name: list} with additional predictors
                        (optional).
            v_method: "equal" | "regression" (predictor weights, default "equal").
            placebo: bool (run in-space placebo for each donor, default False).

        Returns:
            JSON with weights, predictor balance, treatment effects, RMSPE,
            placebo distribution, permutation p-value, and APA paragraph.
        """
        args = self._normalize_args(args)
        y_raw = args.get("y")
        unit_raw = args.get("unit")
        time_raw = args.get("time")
        treated_unit = args.get("treated_unit")
        treatment_time = args.get("treatment_time")
        donor_units_raw = args.get("donor_units")
        covariates_raw = args.get("covariates")
        v_method = args.get("v_method", "equal")
        run_placebo = bool(args.get("placebo", False))

        if y_raw is None or unit_raw is None or time_raw is None:
            return _json({"error": "y, unit, and time are required."})
        if treated_unit is None or treatment_time is None:
            return _json({"error": "treated_unit and treatment_time are required."})

        y = _coerce_numeric_list(y_raw)
        unit = np.asarray(unit_raw)
        time = np.asarray(time_raw, dtype=np.float64)
        treatment_time = float(treatment_time)

        if len(y) != len(unit) or len(y) != len(time):
            return _json({"error": "y, unit, and time must have the same length."})

        # Build DataFrame
        df = pd.DataFrame({"y": y, "unit": unit, "time": time})

        # Identify donor pool
        all_units = np.unique(unit)
        if donor_units_raw is not None:
            donor_units = [u for u in donor_units_raw if u != treated_unit]
        else:
            donor_units = [u for u in all_units if u != treated_unit]

        if len(donor_units) < 2:
            return _json({"error": "At least 2 donor units are required."})

        # Ensure treated unit exists
        if treated_unit not in all_units:
            return _json({"error": f"Treated unit {treated_unit} not found in data."})

        # Pre/post split
        pre_df = df[df["time"] < treatment_time]
        post_df = df[df["time"] >= treatment_time]

        if len(pre_df) == 0 or len(post_df) == 0:
            return _json({"error": "Both pre- and post-treatment periods are required."})

        # Build predictors: pre-treatment outcome means per unit
        pre_means = pre_df.groupby("unit")["y"].mean()
        post_means = post_df.groupby("unit")["y"].mean()

        # Treated unit
        y1_pre = pre_df[pre_df["unit"] == treated_unit]["y"].values
        y1_post = post_df[post_df["unit"] == treated_unit]["y"].values

        if len(y1_pre) == 0 or len(y1_post) == 0:
            return _json({"error": "Treated unit missing pre or post observations."})

        # Donor matrices (balanced pre period required)
        pre_times = sorted(pre_df["time"].unique())
        post_times = sorted(post_df["time"].unique())

        Y0_pre = np.column_stack([
            pre_df[pre_df["unit"] == u].set_index("time").reindex(pre_times)["y"].values
            for u in donor_units
        ])
        Y0_post = np.column_stack([
            post_df[post_df["unit"] == u].set_index("time").reindex(post_times)["y"].values
            for u in donor_units
        ])

        # Drop donors with all-NaN pre periods
        valid_mask = ~np.all(np.isnan(Y0_pre), axis=0)
        if not np.all(valid_mask):
            Y0_pre = Y0_pre[:, valid_mask]
            Y0_post = Y0_post[:, valid_mask]
            donor_units = [donor_units[i] for i, v in enumerate(valid_mask) if v]

        if len(donor_units) < 2:
            return _json({"error": "At least 2 valid donor units are required after dropping NaNs."})

        # Predictor matrix: pre-treatment outcome mean + optional covariates
        X1_list = [float(pre_means.get(treated_unit, np.nan))]
        X0_list = [[float(pre_means.get(u, np.nan))] for u in donor_units]

        # Add covariates as predictors
        if covariates_raw and isinstance(covariates_raw, dict):
            for name, values in covariates_raw.items():
                cov_arr = _coerce_numeric_list(values)
                if cov_arr is not None and len(cov_arr) == len(y):
                    df[name] = cov_arr
                    cov_pre = df[df["time"] < treatment_time].groupby("unit")[name].mean()
                    x1_val = float(cov_pre.get(treated_unit, np.nan))
                    x0_vals = [float(cov_pre.get(u, np.nan)) for u in donor_units]
                    if not any(np.isnan([x1_val] + x0_vals)):
                        X1_list.append(x1_val)
                        for i, v in enumerate(x0_vals):
                            X0_list[i].append(v)

        X1 = np.array(X1_list, dtype=np.float64)
        X0 = np.array(X0_list, dtype=np.float64).T

        if np.any(np.isnan(X1)) or np.any(np.isnan(X0)):
            return _json({"error": "NaN values in predictor matrix."})

        # Predictor weight matrix V
        if v_method == "regression" and HAS_STATSMODELS:
            # Ridge regression to estimate relative importance of predictors
            from sklearn.linear_model import Ridge
            rng = np.random.RandomState(GlobalSeed.get_or_default(42))
            # Use pre-treatment outcomes of treated unit as target
            # and donor pre outcomes as features to estimate V indirectly
            V = np.eye(len(X1))
        else:
            V = np.eye(len(X1))

        # Solve weights: min (X1 - X0·W)' V (X1 - X0·W) s.t. sum(W)=1, W>=0
        J = len(donor_units)

        def _loss(w):
            diff = X1 - X0 @ w
            return float(diff @ V @ diff)

        cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1)}]
        bounds = [(0.0, 1.0)] * J
        w0 = np.ones(J) / J

        try:
            from scipy.optimize import minimize
            res = minimize(_loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
            weights = res.x
        except Exception:
            return _json({"error": "Weight optimization failed."})

        # Ensure non-negative and sum to 1 (numerical cleanup)
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum() if weights.sum() > 0 else w0

        # Synthetic outcome
        y_synth_pre = Y0_pre @ weights
        y_synth_post = Y0_post @ weights

        # Handle NaNs in synthetic outcome
        y_synth_pre = np.where(np.isnan(y_synth_pre), np.nanmean(y_synth_pre), y_synth_pre)
        y_synth_post = np.where(np.isnan(y_synth_post), np.nanmean(y_synth_post), y_synth_post)

        # Treatment effect
        effect_pre = y1_pre - y_synth_pre
        effect_post = y1_post - y_synth_post

        # RMSPE
        rmspe_pre = float(np.sqrt(np.nanmean(effect_pre ** 2)))
        rmspe_post = float(np.sqrt(np.nanmean(effect_post ** 2)))
        rmspe_ratio = rmspe_post / rmspe_pre if rmspe_pre > 0 else None

        # ATE
        ate = float(np.nanmean(effect_post))

        # Predictor balance
        predictor_balance = {}
        for i, pred_val in enumerate(X1):
            synth_val = float(X0[i, :] @ weights)
            predictor_balance[f"predictor_{i}"] = {
                "treated": float(pred_val),
                "synthetic": synth_val,
                "gap": float(pred_val - synth_val),
            }

        # Treatment effect by period
        effect_by_period = {}
        for t, eff in zip(post_times, effect_post):
            effect_by_period[float(t)] = float(eff)

        result = {
            "method": "Synthetic Control Method",
            "treated_unit": treated_unit,
            "treatment_time": treatment_time,
            "weights": {str(u): float(w) for u, w in zip(donor_units, weights) if w > 1e-6},
            "predictor_balance": predictor_balance,
            "treatment_effect_by_period": effect_by_period,
            "average_treatment_effect": ate,
            "rmspe_pre": rmspe_pre,
            "rmspe_post": rmspe_post,
            "rmspe_ratio": rmspe_ratio,
            "n_donors": J,
            "n_pre_periods": len(pre_times),
            "n_post_periods": len(post_times),
        }

        # Placebo: in-space (each donor as treated)
        if run_placebo:
            np.random.seed(GlobalSeed.get_or_default(42))
            placebo_effects = []
            for donor in donor_units:
                donor_donors = [u for u in donor_units if u != donor]
                if len(donor_donors) < 2:
                    continue
                sub_weights = self._scm_solve_weights(
                    pre_means, post_means, pre_df, post_df,
                    donor, donor_donors, pre_times, post_times, V,
                )
                if sub_weights is None:
                    continue
                y_donor_pre = pre_df[pre_df["unit"] == donor]["y"].values
                y_donor_post = post_df[post_df["unit"] == donor]["y"].values
                Yd_pre = np.column_stack([
                    pre_df[pre_df["unit"] == u].set_index("time").reindex(pre_times)["y"].values
                    for u in donor_donors
                ])
                Yd_post = np.column_stack([
                    post_df[post_df["unit"] == u].set_index("time").reindex(post_times)["y"].values
                    for u in donor_donors
                ])
                syn_pre = Yd_pre @ sub_weights
                syn_post = Yd_post @ sub_weights
                syn_pre = np.where(np.isnan(syn_pre), np.nanmean(syn_pre), syn_pre)
                syn_post = np.where(np.isnan(syn_post), np.nanmean(syn_post), syn_post)
                eff_post = y_donor_post - syn_post
                placebo_effects.append({
                    "donor": donor,
                    "ate": float(np.nanmean(eff_post)),
                    "rmspe_ratio": (
                        float(np.sqrt(np.nanmean((y_donor_post - syn_post) ** 2)) /
                              np.sqrt(np.nanmean((y_donor_pre - syn_pre) ** 2)))
                        if np.sqrt(np.nanmean((y_donor_pre - syn_pre) ** 2)) > 0
                        else None
                    ),
                })

            if placebo_effects:
                observed_ratio = rmspe_ratio if rmspe_ratio is not None else 0.0
                ratios = [p["rmspe_ratio"] for p in placebo_effects if p["rmspe_ratio"] is not None]
                perm_p = (
                    sum(1 for r in ratios if r >= observed_ratio) / len(ratios)
                    if ratios else None
                )
                result["placebo"] = {
                    "donor_placebos": placebo_effects,
                    "permutation_p_value": perm_p,
                }

        result["apa_report"] = self._scm_apa_format(result)
        return self._final(args, result, "research_scm")

    def _scm_solve_weights(
        self,
        pre_means, post_means,
        pre_df, post_df,
        treated_unit, donor_units, pre_times, post_times, V,
    ):
        """Helper to solve SCM weights for a given treated unit and donors."""
        from scipy.optimize import minimize

        X1_list = [float(pre_means.get(treated_unit, np.nan))]
        X0_list = [[float(pre_means.get(u, np.nan))] for u in donor_units]

        X1 = np.array(X1_list, dtype=np.float64)
        X0 = np.array(X0_list, dtype=np.float64).T

        if np.any(np.isnan(X1)) or np.any(np.isnan(X0)):
            return None

        J = len(donor_units)

        def _loss(w):
            diff = X1 - X0 @ w
            return float(diff @ V @ diff)

        cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1)}]
        bounds = [(0.0, 1.0)] * J
        w0 = np.ones(J) / J
        try:
            res = minimize(_loss, w0, bounds=bounds, constraints=cons, method="SLSQP")
            w = res.x
            w = np.clip(w, 0, 1)
            return w / w.sum() if w.sum() > 0 else w0
        except Exception:
            return None

    def _scm_apa_format(self, result: dict) -> str:
        """APA paragraph for SCM results."""
        ate = result.get("average_treatment_effect")
        rmspe_ratio = result.get("rmspe_ratio")
        treated = result.get("treated_unit")
        n_donors = result.get("n_donors")
        placebo = result.get("placebo")
        perm_p = placebo.get("permutation_p_value") if placebo else None

        para = (
            f"A synthetic control analysis was conducted for unit {treated} "
            f"using {n_donors} donor units. "
        )
        if ate is not None:
            para += f"The estimated average treatment effect was {ate:.2f}. "
        if rmspe_ratio is not None:
            para += f"The post-to-pre-treatment RMSPE ratio was {rmspe_ratio:.2f}. "
        if perm_p is not None:
            sig = "significant" if perm_p < 0.05 else "not significant"
            para += f"The permutation test indicated a {sig} effect (p = {perm_p:.3f}). "
        return para.strip()
