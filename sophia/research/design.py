"""Research design engine: DOE, power analysis, random assignment.

Pure computation module -- no file I/O.  Wraps pyDOE3 (or pyDOE2),
pingouin, and statsmodels to provide a unified interface for experimental
design generation, statistical power / sample-size calculation, and
randomisation of experimental units.
"""

import json
import math
from typing import Any, Dict, List, Optional

import numpy as np

from sophia.research._input import resolve_parent_ids

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
try:
    import pyDOE3 as doe

    HAS_DOE = True
except ImportError:
    try:
        import pyDOE2 as doe  # type: ignore[no-redef]

        HAS_DOE = True
    except ImportError:
        HAS_DOE = False

try:
    import pingouin as pg

    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

try:
    from statsmodels.stats import power as sm_power

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def _json_serializable(obj: Any) -> Any:
    """Recursively convert numpy / python objects to JSON-safe types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(v) for v in obj]
    return obj


def _result(**fields: Any) -> str:
    """Build a JSON string from keyword arguments, ensuring serialisable."""
    import json

    return json.dumps(_json_serializable(fields), indent=2, default=str)


# ======================================================================
# Main engine
# ======================================================================

class ResearchDesignEngine:
    """Stateless computation engine for experimental research design."""

    def __init__(self, store=None, guard=None):
        self.store = store
        self.guard = guard

    # ------------------------------------------------------------------
    # ResultStore plumbing
    # ------------------------------------------------------------------

    def _sanitize_params(self, args: dict) -> dict:
        """Replace bulky arrays / long strings with summaries."""
        clean: Dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, list):
                if len(v) > 80:
                    clean[k] = f"<list len={len(v)}>"
                elif v and isinstance(v[0], str):
                    total_chars = sum(len(s) for s in v)
                    if total_chars > 4000:
                        clean[k] = f"<list of {len(v)} strings, total_chars={total_chars}>"
                    else:
                        clean[k] = v
                elif v and isinstance(v[0], (list, tuple)):
                    total = sum(len(row) if hasattr(row, "__len__") else 1 for row in v)
                    if total > 200:
                        clean[k] = f"<nested list outer={len(v)} total={total}>"
                    else:
                        clean[k] = v
                else:
                    clean[k] = v
            elif isinstance(v, dict):
                total = sum(len(str(x)) for x in v.values())
                if total > 4000:
                    clean[k] = f"<dict keys={len(v)}>"
                else:
                    clean[k] = v
            elif isinstance(v, str) and len(v) > 2000:
                clean[k] = f"<str len={len(v)}>"
            else:
                clean[k] = v
        return clean

    def _final(self, args: dict, result: dict, tool_name: str) -> str:
        """Persist a successful result to the store and embed result_id."""
        if "error" in result:
            return _result(**result)
        if self.store is None:
            return _result(**result)
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
        return _result(**result)

    # ------------------------------------------------------------------
    # Factorial designs
    # ------------------------------------------------------------------

    def factorial_design(self, args: dict) -> str:
        """Generate a factorial experimental design.

        Parameters
        ----------
        args : dict
            factors : int
                Number of factors (independent variables).
            levels : int or list[int]
                Number of levels per factor.  If a single int, every factor
                has that many levels.  If a list, its length must equal
                *factors* and each entry is the level count for the
                corresponding factor.  Ignored for ``fractional`` and
                ``plackett-burman`` designs (those are always 2-level).
            type : str
                ``full`` (default), ``fractional``, or ``plackett-burman``.
            generators : str
                For fractional designs only.  A space-separated string of
                generator expressions, e.g. ``"a b ab"``.  Ignored for
                other types.

        Returns
        -------
        str
            JSON with keys ``design`` (list of lists), ``runs``,
            ``factors``, ``factor_names``, ``design_type`` and optionally
            ``resolution``.
        """
        if not HAS_DOE:
            return _result(
                error="No DOE library available. Install pyDOE3 or pyDOE2.",
            )

        n_factors: int = int(args.get("factors", 3))
        levels_input = args.get("levels", 2)
        design_type: str = str(args.get("type", "full")).lower().strip()
        generators: Optional[str] = args.get("generators")

        factor_names = [f"X{i + 1}" for i in range(n_factors)]

        if design_type == "full":
            if isinstance(levels_input, (list, tuple)):
                level_list = [int(lv) for lv in levels_input]
            else:
                level_list = [int(levels_input)] * n_factors

            if all(lv == 2 for lv in level_list):
                matrix = doe.ff2n(n_factors)
            else:
                matrix = doe.fullfact(level_list)

            return self._final(args, {
                "design": matrix.tolist(),
                "runs": int(matrix.shape[0]),
                "factors": n_factors,
                "factor_names": factor_names,
                "design_type": "full_factorial",
            }, "research_factorial_design")

        elif design_type == "fractional":
            if not generators:
                # Default: first n_factors-1 generators are basic factors,
                # last is the product of all.
                gen_parts = []
                if n_factors == 2:
                    gen_parts = ["a", "b"]
                elif n_factors == 3:
                    gen_parts = ["a", "b", "ab"]
                elif n_factors == 4:
                    gen_parts = ["a", "b", "c", "abc"]
                elif n_factors == 5:
                    gen_parts = ["a", "b", "c", "d", "bcd"]
                elif n_factors >= 6:
                    # General: first n-1 are basic, last is product of all basic
                    basic = [chr(ord('a') + i) for i in range(n_factors - 1)]
                    gen_parts = basic + ["".join(basic)]
                generators = " ".join(gen_parts)

            matrix = doe.fracfact(generators)

            # Attempt to determine resolution from number of runs / factors
            n_runs = int(matrix.shape[0])
            # Resolution heuristic: full factorial for n_factors would be
            # 2**n_factors.  Ratio gives the fraction.
            full_runs = 2 ** n_factors
            fraction = full_runs // n_runs if n_runs > 0 else 0
            if fraction <= 1:
                resolution = "full"
            elif fraction == 2:
                resolution = "III"
            elif fraction == 4:
                resolution = "IV"
            else:
                resolution = "V+"

            return self._final(args, {
                "design": matrix.tolist(),
                "runs": n_runs,
                "factors": n_factors,
                "factor_names": factor_names,
                "design_type": "fractional_factorial",
                "generators": generators,
                "resolution": resolution,
            }, "research_factorial_design")

        elif design_type == "plackett-burman":
            matrix = doe.pbdesign(n_factors)
            return self._final(args, {
                "design": matrix.tolist(),
                "runs": int(matrix.shape[0]),
                "factors": n_factors,
                "factor_names": factor_names,
                "design_type": "plackett_burman",
            }, "research_factorial_design")

        else:
            return _result(
                error=f"Unknown design type '{design_type}'. "
                      f"Use 'full', 'fractional', or 'plackett-burman'.",
            )

    # ------------------------------------------------------------------
    # Response-surface designs
    # ------------------------------------------------------------------

    def response_surface(self, args: dict) -> str:
        """Generate a response-surface design (Box-Behnken or CCD).

        Parameters
        ----------
        args : dict
            factors : int
                Number of factors (3 or more).
            type : str
                ``box-behnken``, ``ccf`` (faced CCD), ``ccc``
                (circumscribed CCD), or ``cci`` (inscribed CCD).
            center : int
                Number of centre-point replicates (default 1).

        Returns
        -------
        str
            JSON with ``design``, ``runs``, ``factors``, ``center_points``,
            ``design_type``.
        """
        if not HAS_DOE:
            return _result(
                error="No DOE library available. Install pyDOE3 or pyDOE2.",
            )

        n_factors: int = int(args.get("factors", 3))
        rs_type: str = str(args.get("type", "box-behnken")).lower().strip()
        center: int = int(args.get("center", 1))

        if rs_type == "box-behnken":
            # pyDOE3 bbdesign accepts center as an int (number of centre
            # points) or None for the library default.
            if center <= 0:
                matrix = doe.bbdesign(n_factors)
            else:
                matrix = doe.bbdesign(n_factors, center=center)

            # Count actual centre points (rows of all zeros)
            center_count = int(np.sum(np.all(np.abs(matrix) < 1e-10, axis=1)))

            return self._final(args, {
                "design": matrix.tolist(),
                "runs": int(matrix.shape[0]),
                "factors": n_factors,
                "center_points": center_count,
                "design_type": "box_behnken",
            }, "research_response_surface")

        elif rs_type in ("ccf", "ccc", "cci"):
            face_map = {
                "ccf": "faced",
                "ccc": "circumscribed",
                "cci": "inscribed",
            }
            face = face_map[rs_type]
            # pyDOE3 ccdesign expects center as a tuple (nc, n0) where nc
            # is centre points in the factorial block and n0 in the axial
            # block.  We replicate the user's request for both blocks.
            center_tuple = (max(center, 1), max(center, 1))
            matrix = doe.ccdesign(n_factors, center=center_tuple, face=face)

            center_count = int(np.sum(np.all(np.abs(matrix) < 1e-10, axis=1)))

            return self._final(args, {
                "design": matrix.tolist(),
                "runs": int(matrix.shape[0]),
                "factors": n_factors,
                "center_points": center_count,
                "design_type": f"central_composite_{rs_type}",
            }, "research_response_surface")

        else:
            return _result(
                error=f"Unknown RSM type '{rs_type}'. Use 'box-behnken', "
                      f"'ccf', 'ccc', or 'cci'.",
            )

    # ------------------------------------------------------------------
    # Latin Hypercube sampling
    # ------------------------------------------------------------------

    def latin_hypercube(self, args: dict) -> str:
        """Generate a Latin Hypercube sample.

        Parameters
        ----------
        args : dict
            dimensions : int
                Number of dimensions (variables).
            samples : int
                Number of sample points.
            criterion : str
                One of ``center``, ``maximin``, ``centermaximin``,
                ``correlation`` (default ``maximin``).

        Returns
        -------
        str
            JSON with ``samples_matrix``, ``dimensions``, ``samples``,
            ``criterion``, ``range`` (min and max values observed).
        """
        if not HAS_DOE:
            return _result(
                error="No DOE library available. Install pyDOE3 or pyDOE2.",
            )

        dimensions: int = int(args.get("dimensions", 2))
        n_samples: int = int(args.get("samples", 10))
        criterion: str = str(args.get("criterion", "maximin")).lower().strip()

        matrix = doe.lhs(dimensions, samples=n_samples, criterion=criterion)

        return self._final(args, {
            "samples_matrix": matrix.tolist(),
            "dimensions": dimensions,
            "samples": n_samples,
            "criterion": criterion,
            "range": {
                "min": float(np.min(matrix)),
                "max": float(np.max(matrix)),
            },
        }, "research_latin_hypercube")

    # ------------------------------------------------------------------
    # Power analysis
    # ------------------------------------------------------------------

    def power_analysis(self, args: dict) -> str:
        """Statistical power analysis and sample-size calculation.

        Parameters
        ----------
        args : dict
            test : str
                ``ttest``, ``anova``, ``correlation``, ``chi2``,
                ``proportion``.
            effect_size : float
                Standardised effect size (Cohen's *d* for t-test, *f* for
                ANOVA, *r* for correlation, *w* for chi-square, Cohen's *h*
                for proportion).
            alpha : float
                Significance level (default 0.05).
            power : float
                Desired power (default 0.80).  Provide this **or** ``n``.
            k_groups : int
                Number of groups for ANOVA (default 2).
            n : int
                Sample size per group.  If given, power is computed instead
                of sample size.
            alternative : str
                ``two-sided`` (default), ``greater``, or ``less``.

        Returns
        -------
        str
            JSON with ``result_type`` (``power`` or ``sample_size``),
            ``result_value``, and all parameters used.
        """
        test: str = str(args.get("test", "ttest")).lower().strip()
        effect_size: float = float(args.get("effect_size", 0.5))
        alpha: float = float(args.get("alpha", 0.05))
        power_target: float = float(args.get("power", 0.80))
        k_groups: int = int(args.get("k_groups", 2))
        n_obs: Optional[int] = args.get("n")
        if n_obs is not None:
            n_obs = int(n_obs)
        alternative: str = str(
            args.get("alternative", "two-sided")
        ).lower().strip()

        # Compute power (n is given) vs. compute sample size (power is given)
        compute_power = n_obs is not None

        # ---- t-test ---------------------------------------------------
        if test == "ttest":
            inner = self._power_ttest(
                effect_size, alpha, power_target, n_obs,
                alternative, compute_power,
            )

        # ---- one-way ANOVA --------------------------------------------
        elif test == "anova":
            inner = self._power_anova(
                effect_size, alpha, power_target, n_obs,
                k_groups, compute_power,
            )

        # ---- correlation ----------------------------------------------
        elif test == "correlation":
            inner = self._power_corr(
                effect_size, alpha, power_target, n_obs,
                alternative, compute_power,
            )

        # ---- chi-square goodness-of-fit / independence ----------------
        elif test == "chi2":
            inner = self._power_chi2(
                effect_size, alpha, power_target, n_obs,
                k_groups, compute_power,
            )

        # ---- two-proportion z-test ------------------------------------
        elif test == "proportion":
            inner = self._power_proportion(
                effect_size, alpha, power_target, n_obs,
                alternative, compute_power,
            )

        else:
            return _result(
                error=f"Unknown test '{test}'. Supported: ttest, anova, "
                      f"correlation, chi2, proportion.",
            )

        parsed = json.loads(inner)
        return self._final(args, parsed, "research_power_analysis")

    # -- power helpers --------------------------------------------------

    def _power_ttest(
        self, d, alpha, power_target, n_obs, alternative, compute_power,
    ):
        params = {
            "test": "ttest", "effect_size": d, "alpha": alpha,
            "alternative": alternative,
        }
        if HAS_PINGOUIN:
            if compute_power:
                pwr = pg.power_ttest(
                    d=d, n=n_obs, alpha=alpha, alternative=alternative,
                )
                return _result(
                    result_type="power",
                    result_value=float(pwr),
                    n=n_obs,
                    **params,
                )
            else:
                n_calc = pg.power_ttest(
                    d=d, power=power_target, alpha=alpha,
                    alternative=alternative,
                )
                return _result(
                    result_type="sample_size",
                    result_value=float(math.ceil(n_calc)),
                    target_power=power_target,
                    **params,
                )
        elif HAS_STATSMODELS:
            ttp = sm_power.TTestPower()
            if compute_power:
                pwr = ttp.power(
                    effect_size=d, nobs=n_obs, alpha=alpha,
                    alternative=alternative,
                )
                return _result(
                    result_type="power",
                    result_value=float(pwr),
                    n=n_obs,
                    **params,
                )
            else:
                n_calc = ttp.solve_power(
                    effect_size=d, power=power_target, alpha=alpha,
                    alternative=alternative,
                )
                return _result(
                    result_type="sample_size",
                    result_value=float(math.ceil(n_calc)),
                    target_power=power_target,
                    **params,
                )
        else:
            return _result(
                error="Neither pingouin nor statsmodels available for "
                      "t-test power analysis.",
            )

    def _power_anova(
        self, es, alpha, power_target, n_obs, k_groups, compute_power,
    ):
        params = {
            "test": "anova", "effect_size": es, "alpha": alpha,
            "k_groups": k_groups,
        }
        # pingouin uses eta_squared, not Cohen's f.
        # Cohen's f relates to eta^2 via: eta^2 = f^2 / (1 + f^2)
        eta_sq = es ** 2 / (1.0 + es ** 2)

        if HAS_PINGOUIN:
            if compute_power:
                pwr = pg.power_anova(
                    eta_squared=eta_sq, k=k_groups, n=n_obs, alpha=alpha,
                )
                return _result(
                    result_type="power",
                    result_value=float(pwr),
                    n_per_group=n_obs,
                    eta_squared=eta_sq,
                    **params,
                )
            else:
                n_calc = pg.power_anova(
                    eta_squared=eta_sq, k=k_groups,
                    power=power_target, alpha=alpha,
                )
                return _result(
                    result_type="sample_size",
                    result_value=float(math.ceil(n_calc)),
                    target_power=power_target,
                    eta_squared=eta_sq,
                    **params,
                )
        elif HAS_STATSMODELS:
            fap = sm_power.FTestAnovaPower()
            if compute_power:
                pwr = fap.power(
                    effect_size=es, nobs=n_obs, alpha=alpha,
                    k_groups=k_groups,
                )
                return _result(
                    result_type="power",
                    result_value=float(pwr),
                    n_per_group=n_obs,
                    **params,
                )
            else:
                n_calc = fap.solve_power(
                    effect_size=es, power=power_target, alpha=alpha,
                    k_groups=k_groups,
                )
                return _result(
                    result_type="sample_size",
                    result_value=float(math.ceil(n_calc)),
                    target_power=power_target,
                    **params,
                )
        else:
            return _result(
                error="Neither pingouin nor statsmodels available for "
                      "ANOVA power analysis.",
            )

    def _power_corr(
        self, r, alpha, power_target, n_obs, alternative, compute_power,
    ):
        params = {
            "test": "correlation", "effect_size": r, "alpha": alpha,
            "alternative": alternative,
        }
        if HAS_PINGOUIN:
            if compute_power:
                pwr = pg.power_corr(
                    r=r, n=n_obs, alpha=alpha, alternative=alternative,
                )
                return _result(
                    result_type="power",
                    result_value=float(pwr),
                    n=n_obs,
                    **params,
                )
            else:
                n_calc = pg.power_corr(
                    r=r, power=power_target, alpha=alpha,
                    alternative=alternative,
                )
                return _result(
                    result_type="sample_size",
                    result_value=float(math.ceil(n_calc)),
                    target_power=power_target,
                    **params,
                )
        else:
            return _result(
                error="pingouin is required for correlation power analysis.",
            )

    def _power_chi2(
        self, w, alpha, power_target, n_obs, k_groups, compute_power,
    ):
        params = {
            "test": "chi2", "effect_size": w, "alpha": alpha,
            "dof": k_groups - 1,
        }
        dof = max(k_groups - 1, 1)
        if HAS_PINGOUIN:
            if compute_power:
                pwr = pg.power_chi2(
                    dof=dof, w=w, n=n_obs * k_groups, alpha=alpha,
                )
                return _result(
                    result_type="power",
                    result_value=float(pwr),
                    total_n=n_obs * k_groups,
                    **params,
                )
            else:
                n_total = pg.power_chi2(
                    dof=dof, w=w, power=power_target, alpha=alpha,
                )
                return _result(
                    result_type="sample_size",
                    result_value=float(math.ceil(n_total / k_groups)),
                    total_n=float(math.ceil(n_total)),
                    target_power=power_target,
                    **params,
                )
        elif HAS_STATSMODELS:
            gcp = sm_power.GofChisquarePower()
            if compute_power:
                pwr = gcp.power(
                    effect_size=w, nobs=n_obs * k_groups, alpha=alpha,
                    n_bins=k_groups,
                )
                return _result(
                    result_type="power",
                    result_value=float(pwr),
                    total_n=n_obs * k_groups,
                    **params,
                )
            else:
                n_total = gcp.solve_power(
                    effect_size=w, power=power_target, alpha=alpha,
                    n_bins=k_groups,
                )
                return _result(
                    result_type="sample_size",
                    result_value=float(math.ceil(n_total / k_groups)),
                    total_n=float(math.ceil(n_total)),
                    target_power=power_target,
                    **params,
                )
        else:
            return _result(
                error="Neither pingouin nor statsmodels available for "
                      "chi-square power analysis.",
            )

    def _power_proportion(
        self, h, alpha, power_target, n_obs, alternative, compute_power,
    ):
        params = {
            "test": "proportion", "effect_size": h, "alpha": alpha,
            "alternative": alternative,
        }
        if HAS_STATSMODELS:
            nip = sm_power.NormalIndPower()
            if compute_power:
                pwr = nip.power(
                    effect_size=h, nobs1=n_obs, alpha=alpha,
                    alternative=alternative,
                )
                return _result(
                    result_type="power",
                    result_value=float(pwr),
                    n_per_group=n_obs,
                    **params,
                )
            else:
                n_calc = nip.solve_power(
                    effect_size=h, power=power_target, alpha=alpha,
                    alternative=alternative,
                )
                return _result(
                    result_type="sample_size",
                    result_value=float(math.ceil(n_calc)),
                    target_power=power_target,
                    **params,
                )
        else:
            return _result(
                error="statsmodels is required for proportion power analysis.",
            )

    # ------------------------------------------------------------------
    # Random assignment
    # ------------------------------------------------------------------

    def random_assignment(self, args: dict) -> str:
        """Randomly assign experimental units to groups.

        Parameters
        ----------
        args : dict
            n : int
                Total number of units.
            n_groups : int
                Number of groups (default 2).
            method : str
                ``simple``, ``block``, or ``stratified``.
            block_size : int
                Block size for block randomisation (must divide *n* evenly).
                If not given, defaults to ``n_groups``.
            strata : list[list[int]]
                For stratified randomisation.  Each inner list contains the
                unit indices belonging to one stratum.  Units are randomised
                independently within each stratum.
            seed : int
                Random seed for reproducibility.

        Returns
        -------
        str
            JSON with ``assignments`` (list of group labels, 0-indexed),
            ``n``, ``n_groups``, ``method``.
        """
        n: int = int(args.get("n", 20))
        n_groups: int = int(args.get("n_groups", 2))
        method: str = str(args.get("method", "simple")).lower().strip()
        block_size_arg = args.get("block_size")
        strata_arg = args.get("strata")
        seed = args.get("seed")

        rng = np.random.default_rng(seed)

        if method == "simple":
            assignments = self._simple_random(n, n_groups, rng)

        elif method == "block":
            if block_size_arg is not None:
                block_size = int(block_size_arg)
            else:
                block_size = n_groups
            assignments = self._block_random(n, n_groups, block_size, rng)

        elif method == "stratified":
            if strata_arg is None:
                return _result(
                    error="Stratified randomisation requires 'strata' -- "
                          "a list of lists, each containing unit indices "
                          "for one stratum.",
                )
            strata = [list(s) for s in strata_arg]
            assignments = self._stratified_random(
                n, n_groups, strata, rng,
            )

        else:
            return _result(
                error=f"Unknown method '{method}'. "
                      f"Use 'simple', 'block', or 'stratified'.",
            )

        # Build a summary of group sizes
        group_counts: Dict[int, int] = {}
        for g in assignments:
            group_counts[g] = group_counts.get(g, 0) + 1

        return self._final(args, {
            "assignments": assignments,
            "n": n,
            "n_groups": n_groups,
            "method": method,
            "group_sizes": group_counts,
        }, "research_random_assignment")

    # -- randomisation helpers ------------------------------------------

    @staticmethod
    def _simple_random(
        n: int, n_groups: int, rng: np.random.Generator,
    ) -> List[int]:
        """Simple (unrestricted) random allocation."""
        indices = np.arange(n)
        rng.shuffle(indices)
        assignments = [0] * n
        for i, idx in enumerate(indices):
            assignments[idx] = i % n_groups
        return assignments

    @staticmethod
    def _block_random(
        n: int, n_groups: int, block_size: int,
        rng: np.random.Generator,
    ) -> List[int]:
        """Block randomisation.

        Units are divided into blocks of *block_size*.  Within each block,
        group labels are randomly permuted so that every group appears
        ``block_size // n_groups`` times in the block.
        """
        if block_size % n_groups != 0:
            # Round block_size up to the nearest multiple of n_groups
            block_size = ((block_size // n_groups) + 1) * n_groups

        assignments: List[int] = []
        base_block = []
        repeats = block_size // n_groups
        for g in range(n_groups):
            base_block.extend([g] * repeats)

        n_blocks = math.ceil(n / block_size)
        for _ in range(n_blocks):
            block = list(base_block)
            rng.shuffle(block)
            assignments.extend(block)

        # Trim to exact n
        assignments = assignments[:n]
        return assignments

    @staticmethod
    def _stratified_random(
        n: int, n_groups: int, strata: List[List[int]],
        rng: np.random.Generator,
    ) -> List[int]:
        """Stratified random allocation.

        Each stratum is independently randomised via simple random
        allocation, then results are merged.
        """
        assignments = [0] * n

        for stratum_units in strata:
            stratum_n = len(stratum_units)
            indices = np.arange(stratum_n)
            rng.shuffle(indices)
            for i, idx in enumerate(indices):
                assignments[stratum_units[idx]] = i % n_groups

        return assignments
