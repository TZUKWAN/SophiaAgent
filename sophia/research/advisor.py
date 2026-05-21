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

    # ------------------------------------------------------------------
    # H-1: Research question diagnosis
    # ------------------------------------------------------------------

    def diagnose_question(self, question: str) -> Dict[str, Any]:
        """Diagnose research question type and recommend methods."""
        q_lower = question.lower()

        # Question type detection
        type_indicators = {
            "描述性": ["现状", "情况", "分布", "特征", "比例", "多少", "如何分布", "什么样"],
            "解释性": ["影响", "效应", "因果", "机制", "为什么", "导致", "作用", "路径", "中介", "调节"],
            "探索性": ["探索", "发现", "理解", "如何", "过程", "经历", "体验"],
            "评价性": ["评估", "效果", "成效", "有效性", "价值", "优劣", "比较"],
            "设计性": ["开发", "设计", "构建", "方案", "模型", "框架", "系统"],
        }

        type_scores = {t: 0 for t in type_indicators}
        for qtype, keywords in type_indicators.items():
            for kw in keywords:
                if kw in q_lower:
                    type_scores[qtype] += 1

        question_type = max(type_scores, key=type_scores.get) if max(type_scores.values()) > 0 else "描述性"

        # Paradigm detection
        paradigm_indicators = {
            "实证主义": ["变量", "假设", "检验", "显著", "回归", "因果", "效应", "验证"],
            "解释主义": ["理解", "意义", "建构", "体验", "叙事", "诠释", "主观"],
            "批判理论": ["权力", "不平等", "压迫", "解放", "意识形态", "批判", "结构"],
            "实用主义": ["解决", "实用", "效果", "混合", "多元", "整合"],
        }

        paradigm_scores = {p: 0 for p in paradigm_indicators}
        for paradigm, keywords in paradigm_indicators.items():
            for kw in keywords:
                if kw in q_lower:
                    paradigm_scores[paradigm] += 1

        paradigm = max(paradigm_scores, key=paradigm_scores.get) if max(paradigm_scores.values()) > 0 else "实证主义"

        # Time dimension
        time_indicators = {
            "横截面": ["现状", "当前", "目前", "某时", "截面"],
            "纵向": ["变化", "发展", "趋势", "追踪", "历时", "演变", "过程"],
            "回溯性": ["历史", "回顾", "过去", "追溯", "既往"],
        }

        time_scores = {t: 0 for t in time_indicators}
        for tdim, keywords in time_indicators.items():
            for kw in keywords:
                if kw in q_lower:
                    time_scores[tdim] += 1

        time_dimension = max(time_scores, key=time_scores.get) if max(time_scores.values()) > 0 else "横截面"

        # Analysis level
        level_indicators = {
            "个体": ["个体", "个人", "心理", "认知", "态度", "行为", "感受"],
            "群体": ["群体", "团队", "组织", "班级", "社区", "家庭"],
            "组织": ["组织", "机构", "学校", "企业", "单位", "制度"],
            "社会": ["社会", "国家", "政策", "文化", "结构", "制度", "宏观"],
        }

        level_scores = {l: 0 for l in level_indicators}
        for level, keywords in level_indicators.items():
            for kw in keywords:
                if kw in q_lower:
                    level_scores[level] += 1

        analysis_level = max(level_scores, key=level_scores.get) if max(level_scores.values()) > 0 else "个体"

        # Method recommendations
        recommendations = self._recommend_by_diagnosis(
            question_type, paradigm, time_dimension, analysis_level
        )

        return {
            "question": question,
            "diagnosis": {
                "question_type": question_type,
                "paradigm": paradigm,
                "time_dimension": time_dimension,
                "analysis_level": analysis_level,
            },
            "recommended_methods": recommendations,
        }

    @staticmethod
    def _recommend_by_diagnosis(qtype, paradigm, time_dim, level) -> List[Dict[str, Any]]:
        """Recommend methods based on diagnosis."""
        recs = []

        method_map = {
            ("解释性", "实证主义", "横截面", "个体"): [
                {"method": "回归分析", "score": 0.95, "reason": "适合检验个体层面的横截面因果/相关关系"},
                {"method": "结构方程模型", "score": 0.85, "reason": "适合检验复杂的中介/调节机制"},
            ],
            ("解释性", "实证主义", "纵向", "个体"): [
                {"method": "交叉滞后模型", "score": 0.95, "reason": "适合分析个体层面变量间的纵向因果关系"},
                {"method": "增长曲线模型", "score": 0.90, "reason": "适合追踪个体发展趋势"},
            ],
            ("解释性", "实证主义", "横截面", "群体"): [
                {"method": "多层线性模型", "score": 0.95, "reason": "适合处理群体嵌套结构的数据"},
                {"method": "双重差分", "score": 0.85, "reason": "适合评估群体层面的政策效应"},
            ],
            ("探索性", "解释主义", "横截面", "个体"): [
                {"method": "半结构化访谈", "score": 0.95, "reason": "适合深入探索个体经验和意义建构"},
                {"method": "焦点小组", "score": 0.80, "reason": "适合通过群体互动发现新视角"},
            ],
            ("探索性", "解释主义", "纵向", "个体"): [
                {"method": "生命史访谈", "score": 0.95, "reason": "适合探索个体生命历程中的变化"},
                {"method": "民族志", "score": 0.85, "reason": "适合长期参与观察下的深度理解"},
            ],
            ("描述性", "实证主义", "横截面", "个体"): [
                {"method": "问卷调查", "score": 0.95, "reason": "适合大规模描述个体特征和态度分布"},
                {"method": "描述性统计", "score": 0.90, "reason": "适合呈现变量分布和基本特征"},
            ],
            ("评价性", "实用主义", "横截面", "群体"): [
                {"method": "准实验设计", "score": 0.95, "reason": "适合评估干预措施在群体中的效果"},
                {"method": "混合方法", "score": 0.90, "reason": "适合结合定量效果与定性过程评价"},
            ],
            ("设计性", "实用主义", "横截面", "组织"): [
                {"method": "设计型研究", "score": 0.95, "reason": "适合在真实情境中迭代开发解决方案"},
                {"method": "行动研究", "score": 0.90, "reason": "适合研究者与实践者协作改进实践"},
            ],
        }

        key = (qtype, paradigm, time_dim, level)
        if key in method_map:
            recs = method_map[key]
        else:
            # Generic fallback
            recs = [
                {"method": "问卷调查", "score": 0.70, "reason": "通用数据收集方法"},
                {"method": "半结构化访谈", "score": 0.65, "reason": "适合补充定量数据的深度信息"},
                {"method": "文献分析", "score": 0.60, "reason": "适合梳理现有研究基础"},
            ]

        return recs

    # ------------------------------------------------------------------
    # H-2: Mixed method design generator
    # ------------------------------------------------------------------

    def design_mixed_method(
        self,
        qual_question: str,
        quant_question: str,
        priority: str = "equal",
    ) -> Dict[str, Any]:
        """Generate a mixed-methods research design."""
        # Auto-select design type
        design_type = self._select_mixed_design(qual_question, quant_question, priority)

        designs = {
            "聚合式设计": {
                "rationale": "同时收集定性和定量数据，分别分析后在解释阶段整合，以全面回答研究问题。",
                "qual_phase": {
                    "methods": ["半结构化访谈", "焦点小组"],
                    "data": "访谈录音与转录文本",
                    "analysis": "主题分析",
                },
                "quant_phase": {
                    "methods": ["问卷调查"],
                    "data": "结构化问卷数据",
                    "analysis": "描述统计 + 相关/回归分析",
                },
                "integration_points": [
                    {"phase": "设计阶段", "action": "统一抽样框架，确保两组数据来自同一人群"},
                    {"phase": "解释阶段", "action": "对比定量结果与定性主题，验证、扩展或修正结论"},
                ],
                "timeline": [
                    "第1-2周：设计定量问卷和定性访谈提纲",
                    "第3-4周：同时进行数据收集",
                    "第5-6周：分别进行定量和定性分析",
                    "第7-8周：整合两种数据，撰写报告",
                ],
                "validation_strategy": "三角验证：比较定量结果与定性发现的一致性",
            },
            "解释性顺序设计": {
                "rationale": "先收集定量数据识别总体模式，再用定性数据深入解释异常或关键发现。",
                "qual_phase": {
                    "methods": ["半结构化访谈"],
                    "data": "针对定量异常值的深度访谈",
                    "analysis": "主题分析或叙事分析",
                },
                "quant_phase": {
                    "methods": ["问卷调查"],
                    "data": "大规模问卷数据",
                    "analysis": "描述统计 + 推断统计",
                },
                "integration_points": [
                    {"phase": "设计阶段", "action": "基于定量分析结果设计定性抽样策略（目的性抽样）"},
                    {"phase": "解释阶段", "action": "用定性数据解释定量结果中的异常值和统计显著性的实际意义"},
                ],
                "timeline": [
                    "第1-2周：设计并发放问卷",
                    "第3-4周：定量数据分析，识别需要深入解释的异常模式",
                    "第5-6周：基于定量结果进行目的性抽样和定性访谈",
                    "第7-8周：定性分析并整合两种数据",
                ],
                "validation_strategy": "扩展验证：定性数据扩展并深化对定量发现的解释",
            },
            "探索性顺序设计": {
                "rationale": "先通过定性探索发现变量和假设，再设计定量工具进行检验。",
                "qual_phase": {
                    "methods": ["半结构化访谈", "参与观察"],
                    "data": "访谈转录和观察记录",
                    "analysis": "扎根理论或主题分析",
                },
                "quant_phase": {
                    "methods": ["问卷调查"],
                    "data": "基于定性发现开发的结构化问卷",
                    "analysis": "验证性因子分析 + 结构方程模型",
                },
                "integration_points": [
                    {"phase": "设计阶段", "action": "定性研究发现指导问卷题项开发"},
                    {"phase": "工具开发阶段", "action": "基于定性主题设计量表维度与题项"},
                ],
                "timeline": [
                    "第1-3周：定性探索（访谈/观察）",
                    "第4-5周：定性分析，提炼变量和假设",
                    "第6-7周：基于定性结果开发问卷并预测试",
                    "第8-10周：大规模问卷调查与定量分析",
                ],
                "validation_strategy": "构建验证：定量数据验证定性阶段构建的理论/假设",
            },
            "嵌入式设计": {
                "rationale": "以一种方法为主，另一种方法嵌入其中提供补充信息。",
                "qual_phase": {
                    "methods": ["深度访谈", "田野笔记"],
                    "data": "嵌入在主要数据收集过程中的辅助数据",
                    "analysis": "嵌入式主题分析",
                },
                "quant_phase": {
                    "methods": ["实验/准实验", "问卷"],
                    "data": "主要研究数据",
                    "analysis": "实验效果分析",
                },
                "integration_points": [
                    {"phase": "数据收集阶段", "action": "在主要数据收集中嵌入辅助方法（如实验中的访谈）"},
                    {"phase": "解释阶段", "action": "辅助数据用于解释主要方法难以触及的过程和机制"},
                ],
                "timeline": [
                    "第1-2周：设计主要研究方案",
                    "第3-6周：实施主要研究（嵌入辅助数据收集）",
                    "第7-8周：分别分析两种数据",
                    "第9-10周：撰写整合报告",
                ],
                "validation_strategy": "互补验证：辅助数据提供主要方法无法获得的解释性信息",
            },
        }

        design = designs.get(design_type, designs["聚合式设计"])

        return {
            "design_type": design_type,
            "priority": priority,
            "qual_question": qual_question,
            "quant_question": quant_question,
            **design,
        }

    @staticmethod
    def _select_mixed_design(qual_q, quant_q, priority) -> str:
        """Auto-select mixed design type."""
        qual_lower = qual_q.lower()
        quant_lower = quant_q.lower()

        # Check for embedded indicators
        embedded = "嵌入" in qual_lower + quant_lower or priority != "equal"
        if embedded:
            return "嵌入式设计"

        # If qual question is exploratory -> exploratory sequential
        exploratory = any(k in qual_lower for k in ["探索", "发现", "未知", "初步"])
        if exploratory:
            return "探索性顺序设计"

        # If qual question is explanatory -> explanatory sequential
        explanatory = any(k in qual_lower for k in ["解释", "为什么", "机制", "原因"])
        if explanatory:
            return "解释性顺序设计"

        # Check for convergent indicators in both questions
        convergent = any(k in qual_lower + quant_lower for k in ["验证", "三角", "互补", "全面"])
        if convergent:
            return "聚合式设计"

        # Default: convergent
        return "聚合式设计"

    # ------------------------------------------------------------------
    # H-3: Sampling strategy recommender
    # ------------------------------------------------------------------

    def recommend_sampling(
        self,
        research_design: str,
        population: Optional[str] = None,
        constraints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Recommend sampling strategies based on research design."""
        constraints = constraints or []
        design_lower = research_design.lower()
        pop_lower = (population or "").lower()

        # Determine if qualitative or quantitative
        is_qualitative = any(k in design_lower for k in ["质性", "定性", "访谈", "民族志", "案例", "qualitative"])
        is_quantitative = any(k in design_lower for k in ["量化", "定量", "问卷", "实验", "调查", "quantitative"])

        if not is_qualitative and not is_quantitative:
            # Try to infer from population description
            is_qualitative = any(k in pop_lower for k in ["访谈", "个案", "深度"])
            is_quantitative = any(k in pop_lower for k in ["问卷", "大样本", "随机"])

        if is_qualitative:
            strategies = self._qualitative_strategies(design_lower, pop_lower, constraints)
        elif is_quantitative:
            strategies = self._quantitative_strategies(design_lower, pop_lower, constraints)
        else:
            strategies = self._qualitative_strategies(design_lower, pop_lower, constraints) + \
                        self._quantitative_strategies(design_lower, pop_lower, constraints)

        # Sort by estimated cost (lower cost first)
        strategies.sort(key=lambda s: s.get("estimated_cost", 3))

        return {
            "research_design": research_design,
            "population": population,
            "recommended": strategies[:3],
            "is_qualitative": is_qualitative,
            "is_quantitative": is_quantitative,
        }

    @staticmethod
    def _qualitative_strategies(design, population, constraints) -> List[Dict[str, Any]]:
        """Qualitative sampling strategies."""
        has_access_issue = any(k in " ".join(constraints).lower() for k in ["难以接触", "敏感", "隐蔽"])
        needs_diversity = any(k in design for k in ["比较", "差异", "多元", "多样"])
        theory_building = any(k in design for k in ["扎根", "理论", "grounded"])

        strategies = [
            {
                "strategy": "目的性抽样",
                "description": "根据研究目的有目的地选择能提供丰富信息的个案。",
                "when_to_use": "适合大多数质性研究，尤其是当研究问题明确时。",
                "sample_size_guidance": "通常15-30人，或直至理论饱和。",
                "pros": ["能获取信息丰富的个案", "与研究问题高度相关"],
                "cons": ["不能推广到总体", "研究者主观性强"],
                "estimated_cost": 2,
            },
            {
                "strategy": "理论抽样",
                "description": "基于正在形成的理论，选择能最大化概念变异的个案。",
                "when_to_use": "扎根理论研究中，用于发展和检验理论。",
                "sample_size_guidance": "直至理论饱和（通常20-60人）。",
                "pros": ["能最大化理论发展", "系统性强"],
                "cons": ["需要迭代数据收集", "前期难以确定总样本量"],
                "estimated_cost": 3,
            },
            {
                "strategy": "滚雪球抽样",
                "description": "通过已有参与者推荐新的参与者。",
                "when_to_use": "难以接触的群体（如边缘群体、特定职业群体）。",
                "sample_size_guidance": "通常10-30人，取决于群体规模和可达性。",
                "pros": ["适合难以接触的群体", "成本低"],
                "cons": ["样本同质化风险", "依赖社会网络"],
                "estimated_cost": 1,
            },
            {
                "strategy": "最大变异抽样",
                "description": "有意选择特征差异最大的个案。",
                "when_to_use": "需要展示现象在不同情境下的多样性时。",
                "sample_size_guidance": "通常20-40人，覆盖多个变异维度。",
                "pros": ["能展示现象全貌", "增强结果丰富性"],
                "cons": ["数据量大", "分析复杂"],
                "estimated_cost": 3,
            },
            {
                "strategy": "典型案例抽样",
                "description": "选择能代表平均水平的典型个案。",
                "when_to_use": "需要描述'一般'情况或向实践者展示典型经验时。",
                "sample_size_guidance": "通常5-15个典型案例。",
                "pros": ["结果易于理解", "与实践联系紧密"],
                "cons": ["可能忽略边缘情况", "典型性判断主观"],
                "estimated_cost": 2,
            },
            {
                "strategy": "关键案例抽样",
                "description": "选择对理论或实践具有关键意义的个案。",
                "when_to_use": "需要检验理论边界条件或展示关键转折点时。",
                "sample_size_guidance": "通常1-5个关键案例（深度分析）。",
                "pros": ["能揭示关键机制", "深度极高"],
                "cons": ["推广性极弱", "案例选择至关重要"],
                "estimated_cost": 2,
            },
        ]

        # Adjust recommendations
        if has_access_issue:
            for s in strategies:
                if s["strategy"] == "滚雪球抽样":
                    s["estimated_cost"] = 0
        if theory_building:
            for s in strategies:
                if s["strategy"] == "理论抽样":
                    s["estimated_cost"] = 0
        if needs_diversity:
            for s in strategies:
                if s["strategy"] == "最大变异抽样":
                    s["estimated_cost"] = 0

        return strategies

    @staticmethod
    def _quantitative_strategies(design, population, constraints) -> List[Dict[str, Any]]:
        """Quantitative sampling strategies."""
        has_strata = any(k in " ".join(constraints).lower() for k in ["分层", "子群", "组别"])
        has_clusters = any(k in " ".join(constraints).lower() for k in ["整群", "班级", "学校", "社区"])
        resource_limited = any(k in " ".join(constraints).lower() for k in ["资源有限", "低成本", "预算", "时间紧"])

        strategies = [
            {
                "strategy": "简单随机抽样",
                "description": "总体中每个个体被抽中的概率相等。",
                "when_to_use": "总体规模小、易于获得完整名单时。",
                "sample_size_guidance": "根据效应量和检验力计算，通常N=100-500。",
                "pros": ["统计推断最简单", "偏差最小"],
                "cons": ["需要完整抽样框", "大总体成本高"],
                "estimated_cost": 2,
            },
            {
                "strategy": "分层抽样",
                "description": "先将总体按特征分层，再从每层中随机抽样。",
                "when_to_use": "总体内部差异大，需要保证各子群代表性时。",
                "sample_size_guidance": "每层样本量按比例或最优分配，总N=200-1000。",
                "pros": ["提高代表性", "可进行层间比较"],
                "cons": ["需要分层信息", "设计复杂"],
                "estimated_cost": 3,
            },
            {
                "strategy": "整群抽样",
                "description": "以群体（如班级、社区）为单位随机抽样，然后调查群内全部个体。",
                "when_to_use": "总体分散、个体难以单独接触时。",
                "sample_size_guidance": "群数≥30，群内样本量根据ICC调整。",
                "pros": ["操作方便", "成本低"],
                "cons": ["设计效应导致样本量增加", "群内同质性高"],
                "estimated_cost": 1,
            },
            {
                "strategy": "多阶段抽样",
                "description": "分多个阶段逐步缩小抽样范围（如省→学校→班级→学生）。",
                "when_to_use": "大规模调查，总体层级结构明显时。",
                "sample_size_guidance": "每阶段样本量根据设计效应调整，总N=500-5000。",
                "pros": ["适合大规模调查", "操作可行性强"],
                "cons": ["设计复杂", "误差累积"],
                "estimated_cost": 3,
            },
            {
                "strategy": "便利抽样",
                "description": "选择最容易获得的样本。",
                "when_to_use": "探索性研究、预测试、资源极度受限时。",
                "sample_size_guidance": "尽可能大（≥50），但结果需谨慎解释。",
                "pros": ["最便捷", "成本最低"],
                "cons": ["代表性差", "选择偏差大"],
                "estimated_cost": 0,
            },
        ]

        if has_strata:
            for s in strategies:
                if s["strategy"] == "分层抽样":
                    s["estimated_cost"] = 0
        if has_clusters:
            for s in strategies:
                if s["strategy"] == "整群抽样":
                    s["estimated_cost"] = 0
        if resource_limited:
            for s in strategies:
                if s["strategy"] == "便利抽样":
                    s["estimated_cost"] = 0

        return strategies
