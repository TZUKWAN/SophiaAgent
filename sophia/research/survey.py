"""Survey research engine: reliability, validity, sampling.

Pure-computation engine for survey methodology.
All public methods accept ``args: dict`` (from tool dispatch) and return
``str`` (JSON).  Optional dependencies are handled gracefully.

When constructed with a ``ResultStore`` (P1.4c), each method also persists
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
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import factor_analyzer as fa
    HAS_FACTOR_ANALYZER = True
except ImportError:
    HAS_FACTOR_ANALYZER = False


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


def _coerce_2d(data: Any) -> Optional[np.ndarray]:
    """Coerce *data* to a 2-D numpy float64 array."""
    if data is None:
        return None
    try:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            return None
        return arr
    except (TypeError, ValueError):
        return None


# ===========================================================================
# SurveyEngine
# ===========================================================================

class SurveyEngine:
    """Survey research methodology engine.

    Every public method:

    1. Accepts ``args: dict`` (tool-dispatch payload).
    2. Validates inputs.
    3. Runs the real computation.
    4. Returns a JSON string with full results.

    When a ``store`` (ResultStore) is configured, each call also:

    - Resolves DataFrame-shaped inputs (``result_id`` / ``path`` / dict-of-cols).
    - Supports column-name selectors (``*_col`` / ``*_cols`` args) on top of
      legacy list-based inputs.
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

        Returns None when args provides no DataFrame-shaped source. We do
        NOT coerce a flat ``data=[[...], [...]]`` 2-D legacy payload into a
        DataFrame because survey methods consume those as numpy arrays.
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
        """Apply column-name resolution to common survey arg shapes.

        Supported substitutions (when a DataFrame can be resolved):

        - ``items_cols`` (List[str])  -> ``items`` as (n_items, n_responses)
        - ``data_cols``  (List[str])  -> ``data``  as (n_responses, n_items)
        - ``total_score_col`` (str)   -> ``total_score``
        - When ``items_cols``/``data_cols`` is given, also sets
          ``item_names`` to the column names unless explicitly provided.
        """
        df = self._resolve_input_df(args)
        if df is None:
            return args
        new_args = dict(args)

        items_cols = args.get("items_cols")
        if (
            isinstance(items_cols, list)
            and items_cols
            and all(isinstance(c, str) for c in items_cols)
        ):
            valid_cols = [c for c in items_cols if c in df.columns]
            if valid_cols:
                subset = df[valid_cols].dropna()
                # cronbach / item_analysis use shape (n_items, n_responses)
                arr = subset.to_numpy(dtype=np.float64).T
                new_args.setdefault("items", arr.tolist())
                new_args.setdefault("item_names", valid_cols)

        data_cols = args.get("data_cols")
        if (
            isinstance(data_cols, list)
            and data_cols
            and all(isinstance(c, str) for c in data_cols)
        ):
            valid_cols = [c for c in data_cols if c in df.columns]
            if valid_cols:
                subset = df[valid_cols].dropna()
                arr = subset.to_numpy(dtype=np.float64)
                new_args.setdefault("data", arr.tolist())
                new_args.setdefault("item_names", valid_cols)

        total_score_col = args.get("total_score_col")
        if (
            isinstance(total_score_col, str)
            and total_score_col in df.columns
            and "total_score" not in args
        ):
            new_args["total_score"] = self._column(df, total_score_col).tolist()

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
                try:
                    total = sum(
                        len(sub) if hasattr(sub, "__len__") else 1
                        for sub in v.values()
                    )
                except TypeError:
                    total = len(v)
                if total > 200:
                    out[k] = f"<dict keys={len(v)} total={total}>"
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    def _final(self, args: dict, result: dict, tool_name: str) -> str:
        """Strip numpy types from *result*, optionally persist it, return JSON.

        Stores only successful results (no ``error`` key). Records lineage
        from any ``res_*`` references in args via ``resolve_parent_ids``.
        Failures in the store path are swallowed -- they must not block the
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
    # Cronbach's alpha
    # -----------------------------------------------------------------------

    def cronbach(self, args: dict) -> str:
        """Cronbach's alpha reliability.

        Args:
            items: list of lists (each inner list = one item's responses).
            item_names: list of str (optional).

        Returns JSON with: alpha, n_items, n_responses, item_total_corr,
        alpha_if_deleted.
        """
        args = self._normalize_args(args)
        raw_items = args.get("items")
        item_names = args.get("item_names")

        if not isinstance(raw_items, list) or len(raw_items) < 2:
            return _json({"error": "items must be a list of at least 2 item arrays."})

        try:
            items_arr = np.asarray(raw_items, dtype=np.float64)
            # items_arr: shape (n_items, n_responses)
            if items_arr.ndim == 1:
                return _json({"error": "items must be a 2-D array (items x responses)."})
        except (TypeError, ValueError):
            return _json({"error": "items must contain numeric values."})

        # Ensure shape is (n_items, n_responses)
        if items_arr.ndim != 2:
            return _json({"error": "items must be a 2-D array."})

        # Validate: all items must have same number of responses
        n_items, n_responses = items_arr.shape
        if n_responses < 2:
            return _json({"error": "Each item must have at least 2 responses."})

        if item_names is None:
            item_names = [f"item_{i}" for i in range(n_items)]
        elif len(item_names) != n_items:
            return _json({"error": "item_names count does not match number of items."})

        # Use pingouin if available
        if HAS_PINGOUIN:
            try:
                df = pd.DataFrame(items_arr.T, columns=item_names)
                alpha_result = pg.cronbach_alpha(df)
                alpha_val = float(alpha_result[0])
                # Still compute item-total correlations and alpha-if-deleted manually
                # for the full output
            except Exception:
                alpha_val = None
        else:
            alpha_val = None

        # Manual computation (always computed for full output)
        # Total score per respondent
        total_scores = np.nansum(items_arr, axis=0)  # (n_responses,)
        var_items = np.nanvar(items_arr, axis=1, ddof=1)  # (n_items,)
        var_total = np.nanvar(total_scores, ddof=1)

        if var_total == 0:
            return _json({"error": "Total score variance is zero; cannot compute alpha."})

        k = n_items
        sum_var_items = np.nansum(var_items)
        alpha_manual = (k / (k - 1)) * (1 - sum_var_items / var_total)

        # Use manual value if pingouin was not available
        if alpha_val is None:
            alpha_val = alpha_manual

        # Item-total correlations (corrected: total minus item)
        item_total_corr = []
        for i in range(n_items):
            remaining_total = total_scores - items_arr[i]
            if HAS_SCIPY:
                r, _ = sp_stats.pearsonr(items_arr[i], remaining_total)
            else:
                # Manual Pearson correlation
                x = items_arr[i]
                y = remaining_total
                xm = x - np.nanmean(x)
                ym = y - np.nanmean(y)
                num = np.nansum(xm * ym)
                den = math.sqrt(np.nansum(xm ** 2) * np.nansum(ym ** 2))
                r = float(num / den) if den != 0 else 0.0
            item_total_corr.append(float(r))

        # Alpha-if-deleted
        alpha_if_deleted = []
        for i in range(n_items):
            remaining_items = np.delete(items_arr, i, axis=0)
            remaining_total = np.nansum(remaining_items, axis=0)
            var_remaining_items = np.nanvar(remaining_items, axis=1, ddof=1)
            var_remaining_total = np.nanvar(remaining_total, ddof=1)
            if var_remaining_total == 0:
                alpha_if_deleted.append(None)
            else:
                k_rem = k - 1
                a = (k_rem / (k_rem - 1)) * (1 - np.nansum(var_remaining_items) / var_remaining_total)
                alpha_if_deleted.append(float(a))

        result = {
            "alpha": alpha_val,
            "alpha_manual": float(alpha_manual),
            "n_items": n_items,
            "n_responses": n_responses,
            "item_total_corr": dict(zip(item_names, item_total_corr)),
            "alpha_if_deleted": dict(zip(item_names, alpha_if_deleted)),
        }
        try:
            from sophia.research.apa import APAFormatter
            result["apa"] = (
                f"Cronbach's alpha for the {n_items}-item scale was {alpha_val:.3f} "
                f"(N = {n_responses})."
            )
        except Exception:
            pass
        return self._final(args, result, "research_cronbach")

    # -----------------------------------------------------------------------
    # Factor analysis
    # -----------------------------------------------------------------------

    def factor_analysis(self, args: dict) -> str:
        """Exploratory factor analysis.

        Args:
            data: list of lists (rows=respondents, cols=items).
            n_factors: int (default 2).
            rotation: str ('varimax'|'oblimin'|'none', default 'varimax').
            method: str ('ml'|'minres'|'principal', default 'principal').

        Returns JSON with: loadings, variance_explained, communalities,
        factor_correlations.
        """
        args = self._normalize_args(args)
        raw = args.get("data")
        n_factors = int(args.get("n_factors", 2))
        rotation = args.get("rotation", "varimax")
        method = args.get("method", "principal")

        data = _coerce_2d(raw)
        if data is None:
            return _json({"error": "data must be a 2-D list (rows=respondents, cols=items)."})

        n_obs, n_vars = data.shape
        if n_obs < 3:
            return _json({"error": "At least 3 observations are required."})
        if n_vars < 2:
            return _json({"error": "At least 2 variables are required."})
        if n_factors < 1 or n_factors > n_vars:
            return _json({"error": f"n_factors must be between 1 and {n_vars}."})

        # Standardize data
        data_centered = data - np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0, ddof=1)
        std[std == 0] = 1.0  # avoid division by zero
        data_std = data_centered / std

        # Correlation matrix
        corr_matrix = np.corrcoef(data_std, rowvar=False)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[float(corr_matrix)]])

        # Use factor_analyzer package if available
        if HAS_FACTOR_ANALYZER and method in ("ml", "minres"):
            try:
                fa_model = fa.FactorAnalyzer(
                    n_factors=n_factors,
                    rotation=rotation if rotation != "none" else None,
                    method=method,
                )
                fa_model.fit(data)
                loadings = fa_model.loadings_
                var_exp = fa_model.get_factor_variance()
                communalities = fa_model.get_communalities()

                loadings_list = loadings.tolist()
                var_names_list = [f"Factor_{i + 1}" for i in range(n_factors)]
                variance_explained = {
                    "variance": var_exp[0].tolist() if len(var_exp) > 0 else None,
                    "proportional_variance": var_exp[1].tolist() if len(var_exp) > 1 else None,
                    "cumulative_variance": var_exp[2].tolist() if len(var_exp) > 2 else None,
                }

                # Factor correlations (only for oblique rotations)
                phi = None
                if rotation == "oblimin":
                    phi = fa_model.phi_.tolist() if hasattr(fa_model, "phi_") else None

                result = {
                    "method": method,
                    "rotation": rotation,
                    "n_factors": n_factors,
                    "n_obs": n_obs,
                    "n_vars": n_vars,
                    "loadings": loadings_list,
                    "variance_explained": variance_explained,
                    "communalities": communalities.tolist(),
                    "factor_correlations": phi,
                }
                try:
                    result["apa"] = (
                        f"Exploratory factor analysis ({method}, {rotation} rotation) extracted "
                        f"{n_factors} factors from {n_vars} variables (N = {n_obs})."
                    )
                except Exception:
                    pass
                return self._final(args, result, "research_factor_analysis")
            except Exception:
                pass  # Fall through to manual computation

        # Manual computation: principal component extraction
        # SVD of correlation matrix
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep only n_factors
        eigenvalues_nf = eigenvalues[:n_factors]
        eigenvectors_nf = eigenvectors[:, :n_factors]

        # Initial loadings (unrotated)
        loadings = eigenvectors_nf * np.sqrt(np.maximum(eigenvalues_nf, 0))

        # Variance explained
        total_variance = np.sum(np.maximum(eigenvalues, 0))
        var_per_factor = np.maximum(eigenvalues_nf, 0)
        prop_var = var_per_factor / total_variance if total_variance > 0 else var_per_factor
        cum_var = np.cumsum(prop_var)

        # Communalities
        communalities = np.sum(loadings ** 2, axis=1)

        # Rotation
        phi = np.eye(n_factors).tolist()  # default: orthogonal (identity)
        if rotation == "varimax" and n_factors > 1:
            loadings = self._varimax_rotation(loadings)
            # Recompute communalities after rotation
            communalities = np.sum(loadings ** 2, axis=1)
        elif rotation == "oblimin" and n_factors > 1:
            loadings, phi = self._oblimin_rotation(loadings)
            communalities = np.sum(loadings ** 2, axis=1)

        variance_explained = {
            "ss_loadings": [float(np.sum(loadings[:, j] ** 2)) for j in range(n_factors)],
            "proportional_variance": prop_var.tolist(),
            "cumulative_variance": cum_var.tolist(),
        }

        result = {
            "method": "principal" if method == "principal" else method,
            "rotation": rotation,
            "n_factors": n_factors,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "loadings": loadings.tolist(),
            "variance_explained": variance_explained,
            "communalities": communalities.tolist(),
            "factor_correlations": phi if isinstance(phi, list) else phi.tolist() if phi is not None else None,
            "eigenvalues_all": eigenvalues.tolist(),
        }
        try:
            result["apa"] = (
                f"Principal component analysis ({rotation} rotation) extracted "
                f"{n_factors} factors from {n_vars} variables (N = {n_obs})."
            )
        except Exception:
            pass
        return self._final(args, result, "research_factor_analysis")

    @staticmethod
    def _varimax_rotation(loadings: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """Varimax (orthogonal) rotation via Kaiser normalization."""
        n, k = loadings.shape
        if k < 2:
            return loadings

        # Normalize by communalities (Kaiser normalization)
        h = np.sqrt(np.sum(loadings ** 2, axis=1))
        h[h == 0] = 1.0
        L = loadings / h[:, np.newaxis]

        R = np.eye(k)
        d = 0
        for _ in range(max_iter):
            B = L @ R
            # Gram-Schmidt-like varimax criterion
            U, S, Vt = np.linalg.svd(
                L.T @ (B ** 3 - (1.0 / n) * B @ np.diag(np.sum(B ** 2, axis=0)))
            )
            R = U @ Vt
            d_new = np.sum(S)
            if abs(d_new - d) / (abs(d_new) + 1e-12) < tol:
                break
            d = d_new

        rotated = L @ R
        # Denormalize
        rotated = rotated * h[:, np.newaxis]
        return rotated

    @staticmethod
    def _oblimin_rotation(loadings: np.ndarray, gamma: float = 0.0,
                          max_iter: int = 1000, tol: float = 1e-6):
        """Oblimin (oblique) rotation. Returns rotated loadings and factor correlation matrix."""
        n, k = loadings.shape
        if k < 2:
            return loadings, np.eye(k)

        h = np.sqrt(np.sum(loadings ** 2, axis=1))
        h[h == 0] = 1.0
        L = loadings / h[:, np.newaxis]

        # Simple quartimin (gamma=0) via iterative algorithm
        A = L.copy()
        phi = np.eye(k)

        for _ in range(max_iter):
            # Target: minimize sum of off-diagonal elements of pattern matrix
            C = A.T @ A / n
            # Try to make factors less correlated
            W = np.diag(1.0 / np.diag(C))
            target = C - gamma * np.diag(np.diag(C))
            try:
                phi_new = np.linalg.inv(W) @ target
                # Ensure positive definite
                eigvals = np.linalg.eigvalsh(phi_new)
                if np.any(eigvals <= 0):
                    break
                phi = phi_new
            except np.linalg.LinAlgError:
                break

            # Update loadings
            try:
                B = A @ np.linalg.inv(np.diag(np.sqrt(np.maximum(np.diag(phi), 1e-12))))
            except np.linalg.LinAlgError:
                break

            if np.max(np.abs(B - A)) < tol:
                A = B
                break
            A = B

        rotated = A * h[:, np.newaxis]

        # Factor correlation matrix from pattern
        phi_final = np.corrcoef(rotated, rowvar=False)
        if phi_final.ndim == 0:
            phi_final = np.array([[1.0]])

        return rotated, phi_final

    # -----------------------------------------------------------------------
    # Item analysis
    # -----------------------------------------------------------------------

    def item_analysis(self, args: dict) -> str:
        """Item analysis for scale development.

        Args:
            items: list of lists (each inner list = one item's responses).
            total_score: list (optional, computed if not given).
            item_names: list of str (optional).

        Returns JSON with: difficulty (mean), discrimination (item-total r),
        alpha_if_deleted for each item.
        """
        args = self._normalize_args(args)
        raw_items = args.get("items")
        total_score_raw = args.get("total_score")
        item_names = args.get("item_names")

        if not isinstance(raw_items, list) or len(raw_items) < 1:
            return _json({"error": "items must be a non-empty list of item arrays."})

        try:
            items_arr = np.asarray(raw_items, dtype=np.float64)
        except (TypeError, ValueError):
            return _json({"error": "items must contain numeric values."})

        if items_arr.ndim == 1:
            return _json({"error": "items must be a 2-D array (items x responses)."})

        n_items, n_responses = items_arr.shape

        if item_names is None:
            item_names = [f"item_{i}" for i in range(n_items)]
        elif len(item_names) != n_items:
            return _json({"error": "item_names count does not match number of items."})

        # Compute total score if not provided
        if total_score_raw is not None:
            total_score = np.asarray(total_score_raw, dtype=np.float64)
            if len(total_score) != n_responses:
                return _json({
                    "error": f"total_score length ({len(total_score)}) does not match "
                             f"number of responses ({n_responses})."
                })
        else:
            total_score = np.nansum(items_arr, axis=0)

        # Determine scale max (for difficulty index)
        scale_max = float(np.nanmax(items_arr))
        scale_min = float(np.nanmin(items_arr))
        scale_range = scale_max - scale_min if scale_max != scale_min else 1.0

        results = []
        for i in range(n_items):
            item_vals = items_arr[i]

            # Difficulty: proportion of maximum (mean / max)
            item_mean = float(np.nanmean(item_vals))
            difficulty = item_mean / scale_max if scale_max != 0 else 0.0

            # Discrimination: item-total correlation (corrected)
            remaining_total = total_score - item_vals
            if HAS_SCIPY:
                r, _ = sp_stats.pearsonr(item_vals, remaining_total)
                discrimination = float(r)
            else:
                xm = item_vals - np.nanmean(item_vals)
                ym = remaining_total - np.nanmean(remaining_total)
                num = np.nansum(xm * ym)
                den = math.sqrt(np.nansum(xm ** 2) * np.nansum(ym ** 2))
                discrimination = float(num / den) if den != 0 else 0.0

            # Alpha-if-deleted
            if n_items > 1:
                remaining_items = np.delete(items_arr, i, axis=0)
                rem_total = np.nansum(remaining_items, axis=0)
                var_rem_items = np.nanvar(remaining_items, axis=1, ddof=1)
                var_rem_total = np.nanvar(rem_total, ddof=1)
                if var_rem_total == 0:
                    alpha_del = None
                else:
                    k_rem = n_items - 1
                    if k_rem > 1:
                        alpha_del = float(
                            (k_rem / (k_rem - 1)) * (1 - np.nansum(var_rem_items) / var_rem_total)
                        )
                    else:
                        alpha_del = None
            else:
                alpha_del = None

            results.append({
                "item": item_names[i],
                "mean": item_mean,
                "std": float(np.nanstd(item_vals, ddof=1)),
                "difficulty": difficulty,
                "discrimination": discrimination,
                "alpha_if_deleted": alpha_del,
            })

        # Overall scale statistics
        total_mean = float(np.nanmean(total_score))
        total_std = float(np.nanstd(total_score, ddof=1))

        result = {
            "items": results,
            "overall": {
                "n_items": n_items,
                "n_responses": n_responses,
                "total_mean": total_mean,
                "total_std": total_std,
                "scale_min": float(scale_min),
                "scale_max": float(scale_max),
            },
        }
        try:
            result["apa"] = (
                f"Item analysis of the {n_items}-item scale (N = {n_responses}) "
                f"showed a total mean of {total_mean:.2f} (SD = {total_std:.2f})."
            )
        except Exception:
            pass
        return self._final(args, result, "research_item_analysis")

    # -----------------------------------------------------------------------
    # Sample size
    # -----------------------------------------------------------------------

    def sample_size(self, args: dict) -> str:
        """Sample size calculation for surveys.

        Args:
            population: int (population size, 0 for infinite).
            margin_error: float (default 0.05).
            confidence: float (default 0.95).
            proportion: float (default 0.5, worst case).
            design_effect: float (default 1.0).

        Returns JSON with: n_simple, n_adjusted (with design effect),
        n_per_stratum.
        """
        args = self._normalize_args(args)
        population = int(args.get("population", 0))
        margin_error = float(args.get("margin_error", 0.05))
        confidence = float(args.get("confidence", 0.95))
        proportion = float(args.get("proportion", 0.5))
        design_effect = float(args.get("design_effect", 1.0))

        if not (0 < margin_error < 1):
            return _json({"error": "margin_error must be between 0 and 1 (exclusive)."})

        if not (0 < confidence < 1):
            return _json({"error": "confidence must be between 0 and 1 (exclusive)."})

        if not (0 <= proportion <= 1):
            return _json({"error": "proportion must be between 0 and 1."})

        if design_effect <= 0:
            return _json({"error": "design_effect must be positive."})

        # Z-value for desired confidence level
        if HAS_SCIPY:
            z = float(sp_stats.norm.ppf(1 - (1 - confidence) / 2))
        else:
            # Approximate z-values for common confidence levels
            z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z = z_table.get(confidence, 1.96)

        p = proportion
        q = 1 - p
        e = margin_error

        # Cochran's formula: n0 = (z^2 * p * q) / e^2
        n0 = (z ** 2 * p * q) / (e ** 2)
        n_simple = math.ceil(n0)

        # Adjust for finite population if specified
        if population > 0:
            n_adjusted_fp = n0 / (1 + (n0 - 1) / population)
            n_adjusted_fp = math.ceil(n_adjusted_fp)
        else:
            n_adjusted_fp = n_simple

        # Apply design effect
        n_final = math.ceil(n_adjusted_fp * design_effect)

        # Cap at population size
        if population > 0:
            n_final = min(n_final, population)
            n_adjusted_fp = min(n_adjusted_fp, population)

        result = {
            "method": "Cochran's formula",
            "parameters": {
                "population": population if population > 0 else "infinite",
                "margin_error": margin_error,
                "confidence": confidence,
                "proportion": proportion,
                "design_effect": design_effect,
                "z_value": z,
            },
            "n_simple": n_simple,
            "n_adjusted_finite_population": n_adjusted_fp if population > 0 else None,
            "n_adjusted_design_effect": n_final,
            "n_per_stratum": {
                "2_strata_equal": math.ceil(n_final / 2),
                "3_strata_equal": math.ceil(n_final / 3),
                "4_strata_equal": math.ceil(n_final / 4),
                "5_strata_equal": math.ceil(n_final / 5),
            },
        }
        try:
            result["apa"] = (
                f"Sample size calculation (Cochran's formula) indicated a required sample of "
                f"n = {n_final} ({confidence*100:.0f}% confidence, margin of error = {margin_error})."
            )
        except Exception:
            pass
        return self._final(args, result, "research_sample_size")

    # -----------------------------------------------------------------------
    # Likert scale analysis
    # -----------------------------------------------------------------------

    def likert_analysis(self, args: dict) -> str:
        """Likert scale analysis.

        Args:
            data: list of lists (rows=respondents, cols=items).
            scale_min: int (default 1).
            scale_max: int (default 5).
            item_names: list of str (optional).

        Returns JSON with: frequency distribution, median, IQR, top-box %,
        bottom-box %, inter-item consistency.
        """
        args = self._normalize_args(args)
        raw = args.get("data")
        scale_min = int(args.get("scale_min", 1))
        scale_max = int(args.get("scale_max", 5))
        item_names = args.get("item_names")

        data = _coerce_2d(raw)
        if data is None:
            return _json({"error": "data must be a 2-D list (rows=respondents, cols=items)."})

        n_obs, n_items = data.shape
        if n_obs == 0 or n_items == 0:
            return _json({"error": "data must not be empty."})

        if item_names is None:
            item_names = [f"item_{i}" for i in range(n_items)]
        elif len(item_names) != n_items:
            return _json({"error": "item_names count does not match number of items."})

        scale_range = list(range(scale_min, scale_max + 1))

        # Per-item analysis
        item_results = []
        for j in range(n_items):
            col = data[:, j]

            # Frequency distribution
            freq = {}
            for val in scale_range:
                count = int(np.sum(col == val))
                pct = count / n_obs * 100 if n_obs > 0 else 0.0
                freq[str(val)] = {"count": count, "percentage": float(pct)}

            # Median
            median_val = float(np.median(col))

            # IQR
            q1 = float(np.percentile(col, 25))
            q3 = float(np.percentile(col, 75))
            iqr = q3 - q1

            # Top-box percentage (proportion selecting highest option)
            top_box = float(np.sum(col == scale_max) / n_obs * 100) if n_obs > 0 else 0.0

            # Bottom-box percentage (proportion selecting lowest option)
            bottom_box = float(np.sum(col == scale_min) / n_obs * 100) if n_obs > 0 else 0.0

            item_results.append({
                "item": item_names[j],
                "mean": float(np.mean(col)),
                "median": median_val,
                "std": float(np.std(col, ddof=1)) if n_obs > 1 else 0.0,
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "top_box_pct": top_box,
                "bottom_box_pct": bottom_box,
                "frequency": freq,
            })

        # Inter-item consistency (Cronbach's alpha across items)
        items_t = data.T  # (n_items, n_obs)
        total_scores = np.nansum(items_t, axis=0)
        var_items = np.nanvar(items_t, axis=1, ddof=1)
        var_total = np.nanvar(total_scores, ddof=1)

        if var_total > 0 and n_items > 1:
            inter_item_alpha = float(
                (n_items / (n_items - 1)) * (1 - np.nansum(var_items) / var_total)
            )
        else:
            inter_item_alpha = None

        # Inter-item correlation matrix
        corr_matrix = np.corrcoef(items_t)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[float(corr_matrix)]])

        # Overall scale statistics
        overall_mean = float(np.mean(data))
        overall_median = float(np.median(data))
        overall_top_box = float(np.sum(data == scale_max) / data.size * 100)
        overall_bottom_box = float(np.sum(data == scale_min) / data.size * 100)

        result = {
            "n_respondents": n_obs,
            "n_items": n_items,
            "scale_range": {"min": scale_min, "max": scale_max},
            "items": item_results,
            "overall": {
                "mean": overall_mean,
                "median": overall_median,
                "top_box_pct": overall_top_box,
                "bottom_box_pct": overall_bottom_box,
            },
            "inter_item_consistency": {
                "cronbach_alpha": inter_item_alpha,
                "mean_inter_item_r": float(np.mean(corr_matrix[np.triu_indices(n_items, k=1)]))
                if n_items > 1 else None,
            },
        }
        try:
            result["apa"] = (
                f"Likert scale analysis of {n_items} items (N = {n_obs}, range {scale_min}-{scale_max}) "
                f"yielded an overall mean of {overall_mean:.2f} (median = {overall_median:.2f})."
            )
        except Exception:
            pass
        return self._final(args, result, "research_likert_analysis")
