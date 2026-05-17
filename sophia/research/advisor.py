"""Methodology advisor: recommend research methods based on design and data."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional


class MethodologyAdvisor:
    """Recommend empirical research methods given a study question and data profile.

    When linked to a MethodCatalog, recommendations are filtered to only
    methods that are actually available (status='installed') and enriched
    with live metadata (description, keywords, verified flag).
    """

    def __init__(self, catalog=None):
        self.catalog = catalog  # Optional MethodCatalog instance

    # ------------------------------------------------------------------
    # Decision-tree rule base
    # ------------------------------------------------------------------

    _METHOD_RULES: List[Dict[str, Any]] = [
        # --- Quasi-experimental / Causal ---
        {
            "method_id": "did",
            "tool_name": "research_did",
            "categories": {"causal"},
            "designs": {"quasi-experimental", "observational"},
            "outcomes": {"continuous", "binary", "count"},
            "requires": ["panel", "treatment_var", "post_var"],
            "rationale": "Difference-in-differences leverages pre-post variation in a treated group versus an untreated control, exploiting panel structure.",
            "preconditions": ["Parallel trends assumption must hold (testable via event-study leads).", "Treatment timing must be exogenous or as-good-as-random."],
            "alternatives": ["psm if only cross-sectional data", "iv if a valid instrument exists", "scm if only one treated unit"],
            "confidence_boost": {"panel": 0.15, "staggered": 0.05},
        },
        {
            "method_id": "scm",
            "tool_name": "research_scm",
            "categories": {"causal"},
            "designs": {"quasi-experimental", "observational"},
            "outcomes": {"continuous", "binary"},
            "requires": ["single_treated_unit", "panel", "donor_pool"],
            "rationale": "Synthetic control constructs a weighted combination of donor units to mimic the treated unit's pre-intervention trajectory, then compares post-intervention outcomes.",
            "preconditions": ["A sizeable donor pool (J >= 10) with good pre-treatment fit.", "No simultaneous shocks to donor pool in post-period."],
            "alternatives": ["did if multiple treated units", "its if no comparable donors"],
            "confidence_boost": {"single_treated": 0.20, "panel": 0.10},
        },
        {
            "method_id": "rdd",
            "tool_name": "research_rdd",
            "categories": {"causal"},
            "designs": {"quasi-experimental", "observational"},
            "outcomes": {"continuous", "binary", "count"},
            "requires": ["running_var", "cutoff"],
            "rationale": "Regression discontinuity exploits a sharp cutoff in a running variable to compare units just above and below the threshold, yielding a local average treatment effect.",
            "preconditions": ["A sharp and deterministic cutoff.", "No manipulation of the running variable near the threshold.", "Sufficient density of observations around the cutoff (bandwidth > 50 observations)."],
            "alternatives": ["did if treatment is not threshold-based", "iv if instrument available"],
            "confidence_boost": {"sharp_cutoff": 0.15},
        },
        {
            "method_id": "iv",
            "tool_name": "research_iv",
            "categories": {"causal"},
            "designs": {"quasi-experimental", "observational", "randomized"},
            "outcomes": {"continuous", "binary", "count"},
            "requires": ["instrument"],
            "rationale": "Instrumental variables use an exogenous instrument correlated with treatment but not the outcome except through treatment, addressing endogeneity.",
            "preconditions": ["The instrument must be relevant (first-stage F > 10).", "The instrument must satisfy the exclusion restriction (affects outcome only through treatment).", "Monotonicity (no defiers) for LATE interpretation."],
            "alternatives": ["did if panel data available", "psm if selection on observables", "rdd if threshold-based assignment"],
            "confidence_boost": {"strong_instrument": 0.15},
        },
        {
            "method_id": "psm",
            "tool_name": "research_psm",
            "categories": {"causal"},
            "designs": {"quasi-experimental", "observational"},
            "outcomes": {"continuous", "binary", "count"},
            "requires": ["treatment_var", "covariates"],
            "rationale": "Propensity score matching balances observed covariates between treated and control units to estimate an average treatment effect on the treated (ATT).",
            "preconditions": ["Selection on observables must hold (unconfoundedness).", "Sufficient overlap in propensity scores between groups (common support).", "Covariates must adequately predict treatment assignment."],
            "alternatives": ["did if panel data available", "iv if valid instrument exists", "regression with controls if overlap is good"],
            "confidence_boost": {"cross_sectional": 0.10, "rich_covariates": 0.10},
        },
        {
            "method_id": "its",
            "tool_name": "research_its",
            "categories": {"causal"},
            "designs": {"quasi-experimental", "observational"},
            "outcomes": {"continuous", "count"},
            "requires": ["time_series", "intervention_point"],
            "rationale": "Interrupted time series models the pre-intervention trend and tests for a sudden level shift and/or slope change immediately after the intervention.",
            "preconditions": ["A sufficiently long pre-intervention period (>= 12 time points).", "No concurrent interventions or confounding events at the same time point.", "Autocorrelation must be addressed (Newey-West HAC SE)."],
            "alternatives": ["did if a control group exists", "scm if comparable units available"],
            "confidence_boost": {"long_pre_period": 0.15, "no_control": -0.05},
        },
        {
            "method_id": "mediation",
            "tool_name": "research_mediation",
            "categories": {"causal"},
            "designs": {"observational", "randomized", "quasi-experimental"},
            "outcomes": {"continuous", "binary"},
            "requires": ["mediator", "outcome", "treatment"],
            "rationale": "Mediation analysis decomposes the total effect of treatment on outcome into direct and indirect (mediated) pathways.",
            "preconditions": ["Temporal ordering: treatment -> mediator -> outcome.", "No unmeasured confounders of mediator-outcome relationship.", "For Baron-Kenny: linear relationships assumed."],
            "alternatives": ["structural equation modeling if multiple mediators", "causal_effect for total effect only"],
            "confidence_boost": {"temporal_ordering": 0.10},
        },
        # --- Statistics ---
        {
            "method_id": "ttest",
            "tool_name": "research_ttest",
            "categories": {"statistics"},
            "designs": {"randomized", "observational", "quasi-experimental", "survey"},
            "outcomes": {"continuous"},
            "requires": ["two_groups"],
            "rationale": "A t-test compares means between two groups, with options for independent, paired, or Welch (unequal variances) variants.",
            "preconditions": ["Normality of residuals (or large N > 30 for CLT).", "Independent observations (except for paired design)."],
            "alternatives": ["mann-whitney if non-normal and small sample", "anova if more than 2 groups", "bayesian_ttest if prior information available"],
            "confidence_boost": {"two_groups": 0.10, "normal": 0.05},
        },
        {
            "method_id": "anova",
            "tool_name": "research_anova",
            "categories": {"statistics"},
            "designs": {"randomized", "observational", "quasi-experimental", "survey"},
            "outcomes": {"continuous"},
            "requires": ["three_plus_groups"],
            "rationale": "ANOVA tests for mean differences across three or more groups; repeated-measures ANOVA handles within-subject designs.",
            "preconditions": ["Normality of residuals within each group.", "Homogeneity of variance (unless Welch ANOVA).", "Independent observations (or proper repeated-measures structure)."],
            "alternatives": ["kruskal-wallis if non-normal", "welch_anova if unequal variances", "regression if covariates needed"],
            "confidence_boost": {"three_plus_groups": 0.10, "repeated_measures": 0.05},
        },
        {
            "method_id": "regression",
            "tool_name": "research_regression",
            "categories": {"statistics"},
            "designs": {"randomized", "observational", "quasi-experimental", "survey"},
            "outcomes": {"continuous", "binary", "count"},
            "requires": ["predictors", "outcome"],
            "rationale": "Linear regression models the relationship between predictors and a continuous outcome; extensions include logistic and Poisson regression.",
            "preconditions": ["Linearity of relationship (for OLS).", "Independence of observations.", "No severe multicollinearity among predictors (VIF < 10)."],
            "alternatives": ["correlation if only bivariate", "anova if predictors are categorical", "ml_train if prediction is the goal"],
            "confidence_boost": {"multiple_predictors": 0.10, "continuous_outcome": 0.05},
        },
        {
            "method_id": "correlation",
            "tool_name": "research_correlation",
            "categories": {"statistics"},
            "designs": {"randomized", "observational", "survey"},
            "outcomes": {"continuous"},
            "requires": ["two_continuous_vars"],
            "rationale": "Correlation quantifies the strength and direction of linear (Pearson) or monotonic (Spearman, Kendall) association between two variables.",
            "preconditions": ["Bivariate normality for Pearson r (or large N).", "Monotonic relationship for Spearman/Kendall."],
            "alternatives": ["regression if directionality or control variables needed", "chi_square if variables are categorical"],
            "confidence_boost": {"two_vars": 0.10},
        },
        {
            "method_id": "chi_square",
            "tool_name": "research_chi_square",
            "categories": {"statistics"},
            "designs": {"randomized", "observational", "survey"},
            "outcomes": {"binary", "categorical"},
            "requires": ["categorical_vars", "contingency_table"],
            "rationale": "Chi-square tests examine associations between categorical variables; Fisher exact test handles small expected cell counts.",
            "preconditions": ["Expected cell counts >= 5 for chi-square (Fisher exact if violated).", "Independent observations."],
            "alternatives": ["fisher_exact if small expected counts", "logistic_regression if controlling for covariates"],
            "confidence_boost": {"categorical": 0.10},
        },
        {
            "method_id": "nonparametric",
            "tool_name": "research_nonparametric",
            "categories": {"statistics"},
            "designs": {"randomized", "observational", "quasi-experimental", "survey"},
            "outcomes": {"continuous", "ordinal"},
            "requires": ["small_sample_or_nonnormal"],
            "rationale": "Non-parametric tests (Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman) do not assume normality and are robust for ordinal data or small samples.",
            "preconditions": ["Data are at least ordinal.", "Independent observations (or paired for Wilcoxon/Friedman)."],
            "alternatives": ["ttest if normal and large N", "anova if normal and 3+ groups"],
            "confidence_boost": {"small_sample": 0.15, "non_normal": 0.10},
        },
        # --- Survey ---
        {
            "method_id": "cronbach",
            "tool_name": "research_cronbach",
            "categories": {"survey"},
            "designs": {"survey", "observational", "randomized"},
            "outcomes": {"continuous"},
            "requires": ["multi_item_scale", "likert_data"],
            "rationale": "Cronbach's alpha measures internal consistency reliability of a multi-item scale; values >= 0.70 are generally acceptable.",
            "preconditions": ["Items must measure the same underlying construct.", "At least 2 items.", "Data should be interval-level (Likert scales treated as continuous)."],
            "alternatives": ["factor_analysis if dimensionality is uncertain", "krippendorff if assessing inter-coder reliability"],
            "confidence_boost": {"multi_item": 0.10, "survey_design": 0.10},
        },
        {
            "method_id": "factor_analysis",
            "tool_name": "research_factor_analysis",
            "categories": {"survey"},
            "designs": {"survey", "observational"},
            "outcomes": {"continuous"},
            "requires": ["multi_item_scale", "correlation_matrix"],
            "rationale": "Exploratory factor analysis (EFA) uncovers latent dimensions underlying a set of observed items, with varimax or oblimin rotation.",
            "preconditions": ["Sample size >= 10 * number of items (or 200+).", "Bartlett's test of sphericity significant.", "KMO >= 0.50."],
            "alternatives": ["cronbach if only assessing reliability of a known scale", "item_analysis if evaluating individual item performance"],
            "confidence_boost": {"many_items": 0.10, "large_n": 0.05},
        },
        {
            "method_id": "sample_size",
            "tool_name": "research_sample_size",
            "categories": {"survey", "design"},
            "designs": {"survey", "randomized", "observational", "quasi-experimental"},
            "outcomes": {"continuous", "binary", "count"},
            "requires": ["planning_stage"],
            "rationale": "Sample size calculation ensures adequate statistical power given desired margin of error, confidence level, and design effect.",
            "preconditions": ["A priori estimate of population proportion or effect size.", "Desired confidence level (typically 95%).", "Acceptable margin of error."],
            "alternatives": ["power_analysis if effect size and desired power are known"],
            "confidence_boost": {"planning": 0.10},
        },
        # --- Meta-analysis ---
        {
            "method_id": "meta_random",
            "tool_name": "research_meta_random",
            "categories": {"meta"},
            "designs": {"observational", "randomized", "quasi-experimental"},
            "outcomes": {"continuous", "binary", "count"},
            "requires": ["multiple_studies", "effect_sizes", "variances"],
            "rationale": "Random-effects meta-analysis (DerSimonian-Laird) pools effect sizes across studies while accounting for between-study heterogeneity.",
            "preconditions": ["At least 2 independent studies.", "Effect sizes and their variances must be computable.", "Studies should be sufficiently homogeneous (I2 < 75% is manageable)."],
            "alternatives": ["fixed_effect if heterogeneity is negligible", "subgroup_analysis if moderator variables exist", "bias_test if publication bias suspected"],
            "confidence_boost": {"multiple_studies": 0.15, "heterogeneity_expected": 0.05},
        },
        {
            "method_id": "meta_fixed",
            "tool_name": "research_meta_fixed",
            "categories": {"meta"},
            "designs": {"observational", "randomized", "quasi-experimental"},
            "outcomes": {"continuous", "binary", "count"},
            "requires": ["multiple_studies", "effect_sizes", "variances"],
            "rationale": "Fixed-effect meta-analysis assumes all studies estimate the same underlying effect and weights by inverse variance.",
            "preconditions": ["At least 2 studies with low heterogeneity (I2 < 25%).", "Effect sizes and variances available."],
            "alternatives": ["random_effects if heterogeneity is moderate or high", "subgroup_analysis if moderators suspected"],
            "confidence_boost": {"low_heterogeneity": 0.15},
        },
        # --- Qualitative ---
        {
            "method_id": "thematic",
            "tool_name": "research_thematic",
            "categories": {"qualitative"},
            "designs": {"observational", "survey"},
            "outcomes": {"text"},
            "requires": ["texts", "interviews_or_documents"],
            "rationale": "Thematic analysis identifies, organizes, and describes patterns (themes) within qualitative text data through iterative coding.",
            "preconditions": ["Textual data of sufficient richness (interviews, open-ended responses, documents).", "Clear research question to guide coding.", "For LLM-assisted: a reliable language model provider."],
            "alternatives": ["content_analysis if quantitative word counts are sufficient", "grounded_code if theory-building is the goal"],
            "confidence_boost": {"rich_text": 0.10, "llm_available": 0.10},
        },
        {
            "method_id": "grounded_code",
            "tool_name": "research_grounded_code",
            "categories": {"qualitative"},
            "designs": {"observational", "survey"},
            "outcomes": {"text"},
            "requires": ["texts", "theory_building_goal"],
            "rationale": "Grounded theory coding progresses from open coding (initial concepts) through axial coding (category relationships) to selective coding (core category).",
            "preconditions": ["Iterative engagement with the data.", "Theoretical sampling (if ongoing data collection).", "Memo-writing to capture analytic insights."],
            "alternatives": ["thematic if focused on theme extraction rather than theory", "content_analysis if quantification is desired"],
            "confidence_boost": {"theory_building": 0.15},
        },
        {
            "method_id": "coding_reliability",
            "tool_name": "research_coding_reliability",
            "categories": {"qualitative"},
            "designs": {"observational", "survey", "randomized", "quasi-experimental"},
            "outcomes": {"text"},
            "requires": ["multiple_coders", "coded_data"],
            "rationale": "Inter-coder reliability (Cohen's Kappa, Krippendorff's alpha) quantifies agreement among independent coders, ensuring coding consistency.",
            "preconditions": ["At least 2 independent coders.", "A coding scheme or codebook must be defined.", "Sufficient overlap in coding (all coders code a subset of the same units)."],
            "alternatives": ["krippendorff for more than 2 coders or missing data", "percent_agreement as a simple fallback"],
            "confidence_boost": {"multiple_coders": 0.10},
        },
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise(self, args: dict) -> str:
        """
        Recommend research methods.

        Args:
            research_question: str
            data_description: dict with keys like N, type, units, periods, variables
            design: str ('observational' | 'randomized' | 'quasi-experimental' | 'survey')
            outcome_type: str ('continuous' | 'binary' | 'count' | 'time' | 'text' | 'categorical')
            constraints: list of str
            llm_provider: optional LLM provider for enhanced rationale

        Returns JSON with recommended_methods, preflight_checks, decision_tree_trace.
        """
        rq = str(args.get("research_question", "")).lower()
        dd = args.get("data_description", {}) or {}
        design = str(args.get("design", "observational")).lower()
        outcome = str(args.get("outcome_type", "continuous")).lower()
        constraints = [str(c).lower() for c in (args.get("constraints") or [])]
        provider = args.get("llm_provider")

        # Build feature flags from data_description
        features = self._extract_features(dd, rq, constraints, design)

        # Score each method rule
        scored = []
        trace = []
        for rule in self._METHOD_RULES:
            score, reasons = self._score_rule(rule, design, outcome, features, constraints)
            if score > 0:
                scored.append((score, rule))
                trace.append({
                    "method_id": rule["method_id"],
                    "score": round(score, 3),
                    "reasons": reasons,
                })

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build recommendations, enriched by catalog if available
        recommendations = []
        for rank, (score, rule) in enumerate(scored[:5], start=1):
            rec = {
                "method_id": rule["method_id"],
                "tool_name": rule["tool_name"],
                "rationale": rule["rationale"],
                "preconditions": rule["preconditions"],
                "alternatives": rule["alternatives"],
                "rank": rank,
                "confidence": round(min(score, 1.0), 2),
            }

            # Enrich from catalog if linked
            if self.catalog is not None:
                cat_entry = self.catalog.get_by_tool(rule["tool_name"])
                if cat_entry:
                    # Prefer catalog description if more recent / richer
                    if cat_entry.get("description"):
                        rec["rationale"] = cat_entry["description"]
                    rec["catalog_status"] = cat_entry.get("status", "unknown")
                    rec["catalog_verified"] = bool(cat_entry.get("verified", 0))
                    rec["catalog_source"] = cat_entry.get("source", "builtin")
                    # Filter out methods that are not installed (skill_linked is also valid)
                    if cat_entry.get("status") not in ("installed", "skill_linked"):
                        continue
                else:
                    # Method not in catalog at all -
                    # skip unless it is a core builtin we know exists
                    continue

            # If LLM provider available, ask it to polish rationale
            if provider is not None and rank <= 3:
                try:
                    polished = self._llm_polish_rationale(
                        provider, rq, design, outcome, rule["method_id"], rec["rationale"]
                    )
                    if polished:
                        rec["rationale"] = polished
                except Exception:
                    pass
            recommendations.append(rec)

        # Preflight checks
        preflight = self._preflight_checks(dd, design, outcome, features, constraints)

        result = {
            "recommended_methods": recommendations,
            "preflight_checks": preflight,
            "decision_tree_trace": trace[:10],
            "input_summary": {
                "design": design,
                "outcome_type": outcome,
                "features": features,
                "constraints": constraints,
            },
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(dd: dict, rq: str, constraints: List[str], design: str = "observational") -> Dict[str, Any]:
        """Infer data features from description + research question."""
        n = dd.get("N", dd.get("n", 0))
        if isinstance(n, str):
            try:
                n = int(n)
            except ValueError:
                n = 0
        units = dd.get("units", 0)
        periods = dd.get("periods", 0)
        n_vars = dd.get("variables", dd.get("n_vars", 0))

        features = {
            "n": n,
            "panel": units > 1 and periods > 1,
            "original_panel": units > 1 and periods > 1,
            "time_series": periods >= 3,
            "cross_sectional": units <= 1 and periods <= 1,
            "small_sample": 0 < n < 30,
            "large_sample": n >= 1000,
            "many_items": n_vars >= 10,
            "multi_item_scale": n_vars >= 3,
            "single_treated": "single" in rq or "one unit" in rq or "one country" in rq,
            "donor_pool": units > 5,
            "has_instrument": "instrument" in rq or "instrumental" in rq,
            "has_cutoff": "cutoff" in rq or "threshold" in rq or "score" in rq,
            "multiple_studies": "studies" in rq or "meta-analysis" in rq or "meta analysis" in rq,
            "rich_covariates": n_vars >= 5,
            "text_data": "text" in rq or "interview" in rq or "document" in rq or "transcript" in rq,
            "theory_building": "theory" in rq or "grounded" in rq,
            "rich_text": "interview" in rq or "transcript" in rq,
            "llm_available": False,
            "planning": n == 0,
            "staggered": "staggered" in rq or "different timing" in rq,
            "sharp_cutoff": "sharp" in rq,
            "treatment_var": any(k in rq for k in ("effect", "treatment", "policy", "intervention", "program", "training", "impact", "causal")),
            "post_var": "post" in rq or "after" in rq or "follow" in rq,
            "survey_design": design == "survey",
        }

        # Override from constraints
        for c in constraints:
            if "no pre-treatment" in c or "no pretreatment" in c:
                features["panel"] = False
                features["time_series"] = False
            if "small sample" in c:
                features["small_sample"] = True
                features["large_sample"] = False
            if "no instrument" in c:
                features["has_instrument"] = False
            if "no randomization" in c:
                features["randomized"] = False

        return features

    @classmethod
    def _score_rule(cls, rule: dict, design: str, outcome: str,
                    features: dict, constraints: list) -> tuple:
        """Score a method rule; returns (score, reasons)."""
        score = 0.5  # base score
        reasons = []

        # Design match
        if design in rule["designs"]:
            score += 0.15
            reasons.append(f"design '{design}' matches")
        else:
            score -= 0.3
            reasons.append(f"design '{design}' does not match")

        # Outcome match
        if outcome in rule["outcomes"]:
            score += 0.15
            reasons.append(f"outcome '{outcome}' matches")
        else:
            score -= 0.2
            reasons.append(f"outcome '{outcome}' does not match")

        # Required features
        missing_reqs = []
        for req in rule.get("requires", []):
            if req == "panel" and not features.get("panel"):
                missing_reqs.append(req)
            elif req == "time_series" and not features.get("time_series"):
                missing_reqs.append(req)
            elif req == "treatment_var" and not features.get("treatment_var"):
                missing_reqs.append(req)
            elif req == "single_treated_unit" and not features.get("single_treated"):
                missing_reqs.append(req)
            elif req == "donor_pool" and not features.get("donor_pool"):
                missing_reqs.append(req)
            elif req == "instrument" and not features.get("has_instrument"):
                missing_reqs.append(req)
            elif req == "running_var" and not features.get("has_cutoff"):
                missing_reqs.append(req)
            elif req == "intervention_point" and not features.get("time_series"):
                missing_reqs.append(req)
            elif req == "multiple_studies" and not features.get("multiple_studies"):
                missing_reqs.append(req)
            elif req == "small_sample_or_nonnormal" and not (features.get("small_sample") or features.get("non_normal")):
                missing_reqs.append(req)
            elif req == "three_plus_groups" and features.get("n", 0) < 3:
                # weak check; caller should explicitly pass groups
                pass
            elif req == "two_groups" and features.get("n", 0) < 2:
                pass
            elif req == "categorical_vars" and outcome not in ("binary", "categorical"):
                missing_reqs.append(req)
            elif req == "multi_item_scale" and not features.get("multi_item_scale"):
                missing_reqs.append(req)
            elif req == "planning_stage" and not features.get("planning"):
                missing_reqs.append(req)
            elif req == "multiple_coders" and not features.get("multiple_coders", True):
                missing_reqs.append(req)
            elif req == "texts" and not features.get("text_data"):
                missing_reqs.append(req)
            elif req == "theory_building_goal" and not features.get("theory_building"):
                missing_reqs.append(req)

        if missing_reqs:
            score -= 0.25 * len(missing_reqs)
            reasons.append(f"missing requirements: {missing_reqs}")
        else:
            score += 0.1
            reasons.append("all required features present")

        # Confidence boosts
        for feat, boost in rule.get("confidence_boost", {}).items():
            if features.get(feat):
                score += boost
                reasons.append(f"boost '{feat}' +{boost}")

        # Constraints penalty
        for c in constraints:
            if any(banned in c for banned in ("no " + rule["method_id"], "skip " + rule["method_id"])):
                score = 0.0
                reasons.append(f"explicitly excluded by constraint")
                break

        score = max(0.0, min(1.0, score))
        return score, reasons

    @staticmethod
    def _preflight_checks(dd: dict, design: str, outcome: str,
                          features: dict, constraints: list) -> List[dict]:
        """Generate actionable preflight checks."""
        checks = []
        n = features.get("n", 0)

        if n > 0 and n < 30:
            checks.append({
                "check": "small_sample",
                "status": "warning",
                "message": "Sample size < 30. Consider non-parametric tests or exact methods.",
            })
        if features.get("original_panel") and "no pre-treatment" in " ".join(constraints):
            checks.append({
                "check": "panel_without_pretest",
                "status": "warning",
                "message": "Panel data declared but no pre-treatment data available. DiD/ITS may be infeasible.",
            })
        if design == "quasi-experimental" and not any(
            features.get(k) for k in ("panel", "has_cutoff", "has_instrument", "time_series")
        ):
            checks.append({
                "check": "weak_quasi_design",
                "status": "warning",
                "message": "Quasi-experimental design but no panel, cutoff, instrument, or time series detected. Consider PSM or regression with rich controls.",
            })
        if outcome == "continuous" and n > 1000:
            checks.append({
                "check": "large_n_continuous",
                "status": "info",
                "message": "Large sample with continuous outcome. Standard parametric methods (t-test, ANOVA, regression) are well-powered.",
            })
        if features.get("multiple_studies") and n == 0:
            checks.append({
                "check": "meta_data_needed",
                "status": "info",
                "message": "Meta-analysis requires effect sizes and variances from each study. Ensure these are extracted or computable.",
            })
        if features.get("text_data") and not features.get("multiple_coders", True):
            checks.append({
                "check": "coder_reliability",
                "status": "info",
                "message": "For qualitative coding, consider inter-coder reliability checks (Cohen's Kappa or Krippendorff's alpha).",
            })

        return checks

    @staticmethod
    def _llm_polish_rationale(provider, rq: str, design: str, outcome: str,
                              method_id: str, base_rationale: str) -> Optional[str]:
        """Ask LLM to tailor the rationale to the specific research question."""
        prompt = (
            f"Research question: '{rq}'\n"
            f"Design: {design}, Outcome: {outcome}\n"
            f"Recommended method: {method_id}\n"
            f"Base rationale: {base_rationale}\n\n"
            f"Rewrite the rationale in 1-2 sentences, directly addressing why this method is suitable for the research question. "
            f"Be concise and technical. Return only the rewritten rationale, no extra commentary."
        )
        resp = provider.chat([{"role": "user", "content": prompt}])
        content = getattr(resp, "content", str(resp))
        return content.strip() if content else None
