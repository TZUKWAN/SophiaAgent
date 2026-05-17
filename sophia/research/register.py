"""Register all research method tools into the ToolRegistry.

This is the master registration file.  It receives a dict of engine instances
and creates wrapper functions with proper OpenAI-style schemas for each tool.
"""

import json
from sophia.tools.registry import ToolRegistry


def register_method_tools(registry: ToolRegistry, engines: dict):
    """Register all empirical research method tools.

    Parameters
    ----------
    registry : ToolRegistry
        The central tool registry.
    engines : dict
        Mapping of engine names to engine instances, e.g.
        {
            "statistics": StatisticalEngine(),
            "design": ResearchDesignEngine(),
            "causal": CausalEngine(),
            "survey": SurveyEngine(),
            "qualitative": QualitativeEngine(),
            "meta": MetaAnalysisEngine(),
            "computational": ComputationalEngine(),
            "ml": MLEngine(),
            "llm": LLMEngine(),
            "visualization": VisualizationEngine(workspace),
            "pipeline": ExperimentPipeline(workspace),
        }
    """
    stat = engines.get("statistics")
    if stat:
        _register_stat_tools(registry, stat)

    design = engines.get("design")
    if design:
        _register_design_tools(registry, design)

    causal = engines.get("causal")
    if causal:
        _register_causal_tools(registry, causal)

    survey = engines.get("survey")
    if survey:
        _register_survey_tools(registry, survey)

    qual = engines.get("qualitative")
    if qual:
        _register_qualitative_tools(registry, qual)

    meta = engines.get("meta")
    if meta:
        _register_meta_tools(registry, meta)

    comp = engines.get("computational")
    if comp:
        _register_computational_tools(registry, comp)

    ml = engines.get("ml")
    if ml:
        _register_ml_tools(registry, ml)

    llm = engines.get("llm")
    if llm:
        _register_llm_tools(registry, llm)

    viz = engines.get("visualization")
    if viz:
        _register_viz_tools(registry, viz)

    pipeline = engines.get("pipeline")
    if pipeline:
        _register_pipeline_tools(registry, pipeline)

    advisor = engines.get("advisor")
    if advisor:
        _register_advisor_tools(registry, advisor)

    latex_reporter = engines.get("latex_reporter")
    if latex_reporter:
        _register_latex_tools(registry, latex_reporter)


# =====================================================================
# LaTeX exporter (1 tool)
# =====================================================================

def _register_latex_tools(registry, reporter):
    def _export_latex(args):
        return reporter.export(args)
    registry.register(
        "research_export_latex",
        "Export research results to a compilable LaTeX document (.tex). Supports APA7 and Elsevier templates.",
        {
            "type": "object",
            "properties": {
                "result_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of result_ids to include in the report",
                },
                "title": {
                    "type": "string",
                    "description": "Paper title",
                },
                "authors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of author names",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Sections to include, e.g. ['abstract', 'methods', 'results', 'discussion']",
                },
                "template": {
                    "type": "string",
                    "enum": ["apa7", "elsevier", "ieee"],
                    "description": "LaTeX template",
                },
                "include_tables": {
                    "type": "boolean",
                    "description": "Include auto-generated tables",
                },
                "output_name": {
                    "type": "string",
                    "description": "Output file name (without .tex extension)",
                },
            },
            "required": ["result_ids"],
        },
        _export_latex,
    )


# =====================================================================
# Advisor (1 tool)
# =====================================================================

def _register_advisor_tools(registry, advisor):
    def _methodology_advise(args):
        return advisor.advise(args)
    registry.register(
        "methodology_advise",
        "Recommend research methods based on the study question, data characteristics, and design constraints. Returns ranked method recommendations with rationale, preconditions, and alternatives.",
        {
            "type": "object",
            "properties": {
                "research_question": {
                    "type": "string",
                    "description": "The research question or hypothesis (e.g., 'Does minimum wage increase reduce employment?')",
                },
                "data_description": {
                    "type": "object",
                    "description": "Data profile: e.g. {'N': 5000, 'type': 'panel', 'units': 100, 'periods': 5, 'variables': 10}",
                },
                "design": {
                    "type": "string",
                    "enum": ["observational", "randomized", "quasi-experimental", "survey"],
                    "description": "Study design type",
                },
                "outcome_type": {
                    "type": "string",
                    "enum": ["continuous", "binary", "count", "time", "text", "categorical"],
                    "description": "Type of outcome variable",
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints such as ['no pre-treatment data', 'small sample', 'no instrument']",
                },
            },
            "required": ["research_question"],
        },
        _methodology_advise,
    )


# =====================================================================
# Statistics (11 tools)
# =====================================================================

def _register_stat_tools(registry, engine):
    # research_describe
    def _describe(args):
        return engine.describe(args)
    registry.register(
        "research_describe",
        "Compute descriptive statistics (mean, SD, median, IQR, skewness, kurtosis, CI) for a numeric array.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "data_col": {"type": "string", "description": "Column name for data when using result_id"},
                "data": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Numeric data array",
                },
                "variable": {
                    "type": "string",
                    "description": "Variable name (optional)",
                },
            },
            "required": ["data"],
        },
        _describe,
    )

    # research_ttest
    def _ttest(args):
        return engine.ttest(args)
    registry.register(
        "research_ttest",
        "Perform t-test (independent, paired, Welch, or one-sample). Returns t, p, df, Cohen's d, and optionally BF10.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "group1_col": {"type": "string", "description": "Column name for group1 when using result_id"},
                "group2_col": {"type": "string", "description": "Column name for group2 when using result_id"},
                "group1": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "First group data",
                },
                "group2": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Second group data (not needed for one-sample)",
                },
                "paired": {"type": "boolean", "description": "Paired t-test (default false)", "default": False},
                "welch": {"type": "boolean", "description": "Welch t-test (default false)", "default": False},
                "popmean": {"type": "number", "description": "Population mean for one-sample test (optional)"},
            },
            "required": ["group1"],
        },
        _ttest,
    )

    # research_anova
    def _anova(args):
        return engine.anova(args)
    registry.register(
        "research_anova",
        "Perform ANOVA (one-way, repeated-measures, or Welch). Returns F, p, eta-squared, degrees of freedom.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "data_col": {"type": "string", "description": "Column name for data when using result_id"},
                "group_col": {"type": "string", "description": "Column name for grouping variable when using result_id"},
                "data": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "List of groups (each group is a list of numbers)",
                },
                "groups": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Group labels (optional)",
                },
                "repeated": {"type": "boolean", "description": "Repeated-measures ANOVA (default false)", "default": False},
                "type": {
                    "type": "string",
                    "enum": ["one-way", "rm", "welch"],
                    "description": "ANOVA type (default one-way)",
                    "default": "one-way",
                },
            },
            "required": ["data"],
        },
        _anova,
    )

    # research_chi_square
    def _chi_square(args):
        return engine.chi_square(args)
    registry.register(
        "research_chi_square",
        "Chi-square test (independence, goodness-of-fit, or Fisher exact). Returns chi2, p, dof, Cramer's V.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "table": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Contingency table (2-D array)",
                },
                "test": {
                    "type": "string",
                    "enum": ["independence", "goodness", "fisher"],
                    "description": "Type of chi-square test (default independence)",
                    "default": "independence",
                },
            },
            "required": ["table"],
        },
        _chi_square,
    )

    # research_nonparametric
    def _nonparametric(args):
        return engine.nonparametric(args)
    registry.register(
        "research_nonparametric",
        "Non-parametric tests: Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman. Returns test statistic, p-value, effect size.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "groups": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "List of groups",
                },
                "test": {
                    "type": "string",
                    "enum": ["mann-whitney", "wilcoxon", "kruskal", "friedman"],
                    "description": "Test type (default mann-whitney)",
                    "default": "mann-whitney",
                },
                "paired": {"type": "boolean", "description": "Paired design (default false)", "default": False},
            },
            "required": ["groups"],
        },
        _nonparametric,
    )

    # research_correlation
    def _correlation(args):
        return engine.correlation(args)
    registry.register(
        "research_correlation",
        "Compute correlation (Pearson, Spearman, or Kendall). Returns r, p, r-squared, and 95% CI.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "x_col": {"type": "string", "description": "Column name for X variable when using result_id"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "x": {"type": "array", "items": {"type": "number"}, "description": "X variable"},
                "y": {"type": "array", "items": {"type": "number"}, "description": "Y variable"},
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman", "kendall"],
                    "description": "Correlation method (default pearson)",
                    "default": "pearson",
                },
            },
            "required": ["x", "y"],
        },
        _correlation,
    )

    # research_regression
    def _regression(args):
        return engine.regression(args)
    registry.register(
        "research_regression",
        "Perform simple or multiple regression. Returns coefficients, R-squared, adjusted R-squared, F-statistic, p-values, standard errors.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "x_cols": {"type": "string", "description": "Column names for predictors when using result_id (list)"},
                "y": {"type": "array", "items": {"type": "number"}, "description": "Dependent variable"},
                "X": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "number"}},
                        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    ],
                    "description": "Predictor(s): single list or list of lists",
                },
                "x_names": {"type": "array", "items": {"type": "string"}, "description": "Predictor names (optional)"},
                "y_name": {"type": "string", "description": "Dependent variable name (optional)"},
            },
            "required": ["y", "X"],
        },
        _regression,
    )

    # research_normality
    def _normality(args):
        return engine.normality(args)
    registry.register(
        "research_normality",
        "Test normality (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling). Returns test statistic and p-value.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "data_col": {"type": "string", "description": "Column name for data when using result_id"},
                "data": {"type": "array", "items": {"type": "number"}, "description": "Numeric data array"},
                "test": {
                    "type": "string",
                    "enum": ["shapiro", "ks", "anderson", "all"],
                    "description": "Normality test (default shapiro)",
                    "default": "shapiro",
                },
            },
            "required": ["data"],
        },
        _normality,
    )

    # research_effect_size
    def _effect_size(args):
        return engine.effect_size(args)
    registry.register(
        "research_effect_size",
        "Compute effect sizes: Cohen's d, Hedges' g, eta-squared, or odds ratio with 95% CI.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "group1_col": {"type": "string", "description": "Column name for group1 when using result_id"},
                "group2_col": {"type": "string", "description": "Column name for group2 when using result_id"},
                "group1": {"type": "array", "items": {"type": "number"}, "description": "First group data"},
                "group2": {"type": "array", "items": {"type": "number"}, "description": "Second group data"},
                "metric": {
                    "type": "string",
                    "enum": ["cohens_d", "hedges_g", "eta_squared", "odds_ratio"],
                    "description": "Effect size metric (default cohens_d)",
                    "default": "cohens_d",
                },
                "table": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "2x2 table for odds ratio",
                },
            },
            "required": [],
        },
        _effect_size,
    )

    # research_bayesian
    def _bayesian(args):
        return engine.bayesian(args)
    registry.register(
        "research_bayesian",
        "Bayesian t-test (requires pingouin). Returns BF10 with interpretation (anecdotal to decisive evidence).",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "group1_col": {"type": "string", "description": "Column name for group1 when using result_id"},
                "group2_col": {"type": "string", "description": "Column name for group2 when using result_id"},
                "group1": {"type": "array", "items": {"type": "number"}, "description": "First group data"},
                "group2": {"type": "array", "items": {"type": "number"}, "description": "Second group data"},
            },
            "required": ["group1", "group2"],
        },
        _bayesian,
    )

    # research_auto_test
    def _auto_test(args):
        return engine.auto_test(args)
    registry.register(
        "research_auto_test",
        "Automatically select and run the appropriate statistical test based on data characteristics. Checks normality, decides between parametric and non-parametric tests.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "data": {
                    "type": "object",
                    "description": "Data as {group_name: [values]} dict",
                },
                "groups": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Groups as list of lists",
                },
                "paired": {"type": "boolean", "description": "Paired design (default false)", "default": False},
                "research_question": {"type": "string", "description": "Research question hint (optional)"},
            },
            "required": [],
        },
        _auto_test,
    )


# =====================================================================
# Design (5 tools)
# =====================================================================

def _register_design_tools(registry, engine):
    def _factorial_design(args):
        return engine.factorial_design(args)
    registry.register(
        "research_factorial_design",
        "Generate factorial experimental design (full, fractional, or Plackett-Burman). Returns design matrix, run count, factor names.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "factors": {"type": "integer", "description": "Number of factors (default 3)", "default": 3},
                "levels": {
                    "oneOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}],
                    "description": "Levels per factor (default 2)",
                    "default": 2,
                },
                "type": {
                    "type": "string",
                    "enum": ["full", "fractional", "plackett-burman"],
                    "description": "Design type (default full)",
                    "default": "full",
                },
                "generators": {"type": "string", "description": "Generator expressions for fractional designs (optional)"},
            },
            "required": [],
        },
        _factorial_design,
    )

    def _response_surface(args):
        return engine.response_surface(args)
    registry.register(
        "research_response_surface",
        "Generate response-surface design (Box-Behnken or Central Composite). Returns design matrix and center points.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "factors": {"type": "integer", "description": "Number of factors (default 3)", "default": 3},
                "type": {
                    "type": "string",
                    "enum": ["box-behnken", "ccf", "ccc", "cci"],
                    "description": "RSM type (default box-behnken)",
                    "default": "box-behnken",
                },
                "center": {"type": "integer", "description": "Center point replicates (default 1)", "default": 1},
            },
            "required": [],
        },
        _response_surface,
    )

    def _latin_hypercube(args):
        return engine.latin_hypercube(args)
    registry.register(
        "research_latin_hypercube",
        "Generate Latin Hypercube sample for computer experiments. Supports center, maximin, correlation criteria.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "dimensions": {"type": "integer", "description": "Number of dimensions (default 2)", "default": 2},
                "samples": {"type": "integer", "description": "Number of sample points (default 10)", "default": 10},
                "criterion": {
                    "type": "string",
                    "enum": ["center", "maximin", "centermaximin", "correlation"],
                    "description": "Sampling criterion (default maximin)",
                    "default": "maximin",
                },
            },
            "required": [],
        },
        _latin_hypercube,
    )

    def _power_analysis(args):
        return engine.power_analysis(args)
    registry.register(
        "research_power_analysis",
        "Statistical power analysis and sample-size calculation for t-test, ANOVA, correlation, chi-square, or proportion tests.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "test": {
                    "type": "string",
                    "enum": ["ttest", "anova", "correlation", "chi2", "proportion"],
                    "description": "Statistical test (default ttest)",
                    "default": "ttest",
                },
                "effect_size": {"type": "number", "description": "Standardized effect size (default 0.5)", "default": 0.5},
                "alpha": {"type": "number", "description": "Significance level (default 0.05)", "default": 0.05},
                "power": {"type": "number", "description": "Desired power (default 0.80)", "default": 0.80},
                "n": {"type": "integer", "description": "Sample size per group (compute power if given)"},
                "k_groups": {"type": "integer", "description": "Number of groups for ANOVA (default 2)", "default": 2},
                "alternative": {
                    "type": "string",
                    "enum": ["two-sided", "greater", "less"],
                    "description": "Alternative hypothesis (default two-sided)",
                    "default": "two-sided",
                },
            },
            "required": [],
        },
        _power_analysis,
    )

    def _random_assignment(args):
        return engine.random_assignment(args)
    registry.register(
        "research_random_assignment",
        "Randomly assign experimental units to groups using simple, block, or stratified randomisation.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "n": {"type": "integer", "description": "Total number of units (default 20)", "default": 20},
                "n_groups": {"type": "integer", "description": "Number of groups (default 2)", "default": 2},
                "method": {
                    "type": "string",
                    "enum": ["simple", "block", "stratified"],
                    "description": "Randomisation method (default simple)",
                    "default": "simple",
                },
                "block_size": {"type": "integer", "description": "Block size for block randomisation (optional)"},
                "strata": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}},
                    "description": "Strata for stratified randomisation (list of lists of unit indices)",
                },
                "seed": {"type": "integer", "description": "Random seed for reproducibility (optional)"},
            },
            "required": ["n"],
        },
        _random_assignment,
    )


# =====================================================================
# Causal (8 tools)
# =====================================================================

def _register_causal_tools(registry, engine):
    def _did(args):
        return engine.did(args)
    registry.register(
        "research_did",
        "Difference-in-Differences estimation. Computes treatment effect using pre/post treatment/control design.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "treat_col": {"type": "string", "description": "Column name for treatment indicator when using result_id"},
                "post_col": {"type": "string", "description": "Column name for post-treatment indicator when using result_id"},
                "time_col": {"type": "string", "description": "Column name for time variable when using result_id"},
                "entity_col": {"type": "string", "description": "Column name for entity/unit variable when using result_id"},
                "treatment_pre": {"type": "array", "items": {"type": "number"}, "description": "Treatment group pre-treatment outcomes"},
                "treatment_post": {"type": "array", "items": {"type": "number"}, "description": "Treatment group post-treatment outcomes"},
                "control_pre": {"type": "array", "items": {"type": "number"}, "description": "Control group pre-treatment outcomes"},
                "control_post": {"type": "array", "items": {"type": "number"}, "description": "Control group post-treatment outcomes"},
            },
            "required": ["treatment_pre", "treatment_post", "control_pre", "control_post"],
        },
        _did,
    )

    def _rdd(args):
        return engine.rdd(args)
    registry.register(
        "research_rdd",
        "Regression Discontinuity Design. Estimates treatment effect at a cutoff using local linear regression.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "running_col": {"type": "string", "description": "Column name for running/forcing variable when using result_id"},
                "running_variable": {"type": "array", "items": {"type": "number"}, "description": "Running/forcing variable"},
                "outcome": {"type": "array", "items": {"type": "number"}, "description": "Outcome variable"},
                "cutoff": {"type": "number", "description": "Treatment cutoff value"},
                "bandwidth": {"type": "number", "description": "Bandwidth for local estimation (optional, auto-calculated)"},
            },
            "required": ["running_variable", "outcome", "cutoff"],
        },
        _rdd,
    )

    def _iv(args):
        return engine.iv(args)
    registry.register(
        "research_iv",
        "Instrumental Variable estimation (Two-Stage Least Squares). Returns first-stage and second-stage results.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "endogenous_col": {"type": "string", "description": "Column name for endogenous predictor when using result_id"},
                "instrument_col": {"type": "string", "description": "Column name for instrument variable when using result_id"},
                "x_cols": {"type": "string", "description": "Column names for predictors when using result_id (list)"},
                "y": {"type": "array", "items": {"type": "number"}, "description": "Outcome variable"},
                "X_endogenous": {"type": "array", "items": {"type": "number"}, "description": "Endogenous predictor"},
                "Z_instrument": {"type": "array", "items": {"type": "number"}, "description": "Instrument variable"},
                "X_exogenous": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Exogenous controls (optional)",
                },
            },
            "required": ["y", "X_endogenous", "Z_instrument"],
        },
        _iv,
    )

    def _psm(args):
        return engine.psm(args)
    registry.register(
        "research_psm",
        "Propensity Score Matching. Estimates treatment effect by matching treated and control units on propensity scores.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "treat_col": {"type": "string", "description": "Column name for treatment indicator when using result_id"},
                "x_cols": {"type": "string", "description": "Column names for predictors when using result_id (list)"},
                "treatment": {"type": "array", "items": {"type": "number"}, "description": "Treatment indicator (0/1)"},
                "outcome": {"type": "array", "items": {"type": "number"}, "description": "Outcome variable"},
                "covariates": {
                    "type": "array",
                    "description": "Covariates for propensity score model (list of lists or 2D array)",
                },
                "method": {
                    "type": "string",
                    "enum": ["nearest", "caliper", "stratification"],
                    "description": "Matching method (default nearest)",
                    "default": "nearest",
                },
            },
            "required": ["treatment", "outcome", "covariates"],
        },
        _psm,
    )

    def _its(args):
        return engine.its(args)
    registry.register(
        "research_its",
        "Interrupted Time Series analysis. Estimates level and trend changes after an intervention.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "time_col": {"type": "string", "description": "Column name for time variable when using result_id"},
                "x_cols": {"type": "string", "description": "Column names for predictors when using result_id (list)"},
                "time": {"type": "array", "items": {"type": "number"}, "description": "Time points"},
                "outcome": {"type": "array", "items": {"type": "number"}, "description": "Outcome values"},
                "intervention_point": {"type": "integer", "description": "Index of intervention (0-based)"},
            },
            "required": ["time", "outcome", "intervention_point"],
        },
        _its,
    )

    def _mediation(args):
        return engine.mediation(args)
    registry.register(
        "research_mediation",
        "Mediation analysis (Baron & Kenny). Computes direct effect, indirect effect, and proportion mediated.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "x_col": {"type": "string", "description": "Column name for X variable when using result_id"},
                "mediator_col": {"type": "string", "description": "Column name for mediator variable when using result_id"},
                "X": {"type": "array", "items": {"type": "number"}, "description": "Independent variable"},
                "M": {"type": "array", "items": {"type": "number"}, "description": "Mediator variable"},
                "Y": {"type": "array", "items": {"type": "number"}, "description": "Outcome variable"},
            },
            "required": ["X", "M", "Y"],
        },
        _mediation,
    )

    def _causal_effect(args):
        return engine.causal_effect(args)
    registry.register(
        "research_causal_effect",
        "Estimate average treatment effect (ATE) from observational data using regression adjustment or inverse probability weighting.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "treat_col": {"type": "string", "description": "Column name for treatment indicator when using result_id"},
                "x_cols": {"type": "string", "description": "Column names for predictors when using result_id (list)"},
                "treatment": {"type": "array", "items": {"type": "number"}, "description": "Treatment indicator (0/1)"},
                "outcome": {"type": "array", "items": {"type": "number"}, "description": "Outcome variable"},
                "covariates": {
                    "type": "array",
                    "description": "Covariates for adjustment (optional)",
                },
                "method": {
                    "type": "string",
                    "enum": ["regression", "ipw", "simple_diff"],
                    "description": "Estimation method (default simple_diff)",
                    "default": "simple_diff",
                },
            },
            "required": ["treatment", "outcome"],
        },
        _causal_effect,
    )

    def _sensitivity(args):
        return engine.sensitivity(args)
    registry.register(
        "research_sensitivity",
        "Sensitivity analysis for causal inferences. Tests robustness of treatment effect estimates to unmeasured confounding.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "treat_col": {"type": "string", "description": "Column name for treatment indicator when using result_id"},
                "x_cols": {"type": "string", "description": "Column names for predictors when using result_id (list)"},
                "treatment": {"type": "array", "items": {"type": "number"}, "description": "Treatment indicator (0/1)"},
                "outcome": {"type": "array", "items": {"type": "number"}, "description": "Outcome variable"},
                "covariates": {"type": "array", "description": "Observed covariates (optional)"},
                "gamma_range": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Range of sensitivity parameter gamma (default [0.5, 1.0, 1.5, 2.0])",
                },
            },
            "required": ["treatment", "outcome"],
        },
        _sensitivity,
    )

    def _scm(args):
        return engine.synthetic_control(args)
    registry.register(
        "research_scm",
        "Synthetic Control Method. Constructs a weighted combination of donor units to estimate the causal effect for a treated unit.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_col": {"type": "string", "description": "Column name for Y/outcome variable when using result_id"},
                "treat_col": {"type": "string", "description": "Column name for treatment indicator when using result_id"},
                "time_col": {"type": "string", "description": "Column name for time variable when using result_id"},
                "entity_col": {"type": "string", "description": "Column name for entity/unit variable when using result_id"},
                "y": {"type": "array", "items": {"type": "number"}, "description": "Outcome values (panel)"},
                "unit": {"type": "array", "description": "Unit IDs"},
                "time": {"type": "array", "description": "Time periods"},
                "treated_unit": {"description": "ID of the treated unit"},
                "treatment_time": {"type": "number", "description": "Pre/post boundary"},
                "donor_units": {"type": "array", "description": "Donor unit IDs (optional)"},
                "covariates": {"type": "object", "description": "Additional predictors (optional)"},
                "v_method": {"type": "string", "enum": ["equal", "regression"], "default": "equal"},
                "placebo": {"type": "boolean", "description": "Run placebo tests", "default": False},
            },
            "required": ["y", "unit", "time", "treated_unit", "treatment_time"],
        },
        _scm,
    )


# =====================================================================
# Survey (5 tools)
# =====================================================================

def _register_survey_tools(registry, engine):
    def _cronbach(args):
        return engine.cronbach(args)
    registry.register(
        "research_cronbach",
        "Compute Cronbach's alpha reliability coefficient with item-total correlations and alpha-if-deleted.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "items_cols": {"type": "string", "description": "Column names for scale items when using result_id (list)"},
                "items": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Item responses (2D array: items x respondents)",
                },
                "item_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Item names (optional)",
                },
            },
            "required": ["items"],
        },
        _cronbach,
    )

    def _factor_analysis(args):
        return engine.factor_analysis(args)
    registry.register(
        "research_factor_analysis",
        "Exploratory factor analysis (principal components, ML, or minres) with varimax or oblimin rotation.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "items_cols": {"type": "string", "description": "Column names for scale items when using result_id (list)"},
                "data": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Data matrix (rows=respondents, cols=items)",
                },
                "n_factors": {"type": "integer", "description": "Number of factors (default 2)", "default": 2},
                "rotation": {
                    "type": "string",
                    "enum": ["varimax", "oblimin", "none"],
                    "description": "Rotation method (default varimax)",
                    "default": "varimax",
                },
                "method": {
                    "type": "string",
                    "enum": ["ml", "minres", "principal"],
                    "description": "Extraction method (default principal)",
                    "default": "principal",
                },
            },
            "required": ["data"],
        },
        _factor_analysis,
    )

    def _item_analysis(args):
        return engine.item_analysis(args)
    registry.register(
        "research_item_analysis",
        "Item analysis for scale development: difficulty, discrimination (item-total r), and alpha-if-deleted per item.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "items_cols": {"type": "string", "description": "Column names for scale items when using result_id (list)"},
                "total_score_col": {"type": "string", "description": "Column name for total score when using result_id"},
                "items": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Item responses (2D array: items x respondents)",
                },
                "total_score": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Total score (optional, computed if omitted)",
                },
                "item_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Item names (optional)",
                },
            },
            "required": ["items"],
        },
        _item_analysis,
    )

    def _sample_size(args):
        return engine.sample_size(args)
    registry.register(
        "research_sample_size",
        "Survey sample size calculation using Cochran's formula. Supports finite population correction and design effects.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "population": {"type": "integer", "description": "Population size (0 for infinite)", "default": 0},
                "margin_error": {"type": "number", "description": "Margin of error (default 0.05)", "default": 0.05},
                "confidence": {"type": "number", "description": "Confidence level (default 0.95)", "default": 0.95},
                "proportion": {"type": "number", "description": "Expected proportion (default 0.5)", "default": 0.5},
                "design_effect": {"type": "number", "description": "Design effect multiplier (default 1.0)", "default": 1.0},
            },
            "required": [],
        },
        _sample_size,
    )

    def _likert_analysis(args):
        return engine.likert_analysis(args)
    registry.register(
        "research_likert_analysis",
        "Likert scale analysis: frequency distribution, median, IQR, top-box/bottom-box percentages, inter-item consistency.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "items_cols": {"type": "string", "description": "Column names for scale items when using result_id (list)"},
                "data": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Likert responses (rows=respondents, cols=items)",
                },
                "scale_min": {"type": "integer", "description": "Scale minimum (default 1)", "default": 1},
                "scale_max": {"type": "integer", "description": "Scale maximum (default 5)", "default": 5},
                "item_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Item names (optional)",
                },
            },
            "required": ["data"],
        },
        _likert_analysis,
    )


# =====================================================================
# Qualitative (5 tools)
# =====================================================================

def _register_qualitative_tools(registry, engine):
    def _thematic(args):
        return engine.thematic(args)
    registry.register(
        "research_thematic",
        "Thematic analysis: extract themes and codes from qualitative text data using keyword frequency and co-occurrence.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Text documents to analyze",
                },
                "n_themes": {"type": "integer", "description": "Number of themes to extract (default 5)", "default": 5},
                "min_word_length": {"type": "integer", "description": "Minimum word length (default 4)", "default": 4},
            },
            "required": ["texts"],
        },
        _thematic,
    )

    def _content(args):
        return engine.content(args)
    registry.register(
        "research_content",
        "Content analysis: word frequency, category counts, and keyword extraction from text data.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Text documents to analyze",
                },
                "categories": {
                    "type": "object",
                    "description": "Category definitions {name: [keywords]} (optional)",
                },
                "top_n": {"type": "integer", "description": "Number of top keywords (default 20)", "default": 20},
            },
            "required": ["texts"],
        },
        _content,
    )

    def _grounded_code(args):
        return engine.grounded_code(args)
    registry.register(
        "research_grounded_code",
        "Grounded theory coding: open coding of text data to identify concepts, categories, and relationships.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Text data to code",
                },
                "existing_codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Existing codes from prior analysis (optional)",
                },
                "min_phrase_length": {"type": "integer", "description": "Minimum phrase length in words (default 2)", "default": 2},
            },
            "required": ["texts"],
        },
        _grounded_code,
    )

    def _sentiment(args):
        return engine.sentiment(args)
    registry.register(
        "research_sentiment",
        "Sentiment analysis: classify text sentiment and compute polarity scores using lexicon-based methods.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "texts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Texts to analyze",
                },
                "method": {
                    "type": "string",
                    "enum": ["lexicon", "keyword"],
                    "description": "Analysis method (default lexicon)",
                    "default": "lexicon",
                },
            },
            "required": ["texts"],
        },
        _sentiment,
    )

    def _coding_reliability(args):
        return engine.coding_reliability(args)
    registry.register(
        "research_coding_reliability",
        "Inter-coder reliability: compute Cohen's kappa, percent agreement, and Krippendorff's alpha between coders.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "coder1": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Codes from coder 1",
                },
                "coder2": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Codes from coder 2",
                },
                "coder3": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Codes from coder 3 (optional)",
                },
            },
            "required": ["coder1", "coder2"],
        },
        _coding_reliability,
    )


# =====================================================================
# Meta-analysis (5 tools)
# =====================================================================

def _register_meta_tools(registry, engine):
    def _meta_fixed(args):
        return engine.meta_fixed(args)
    registry.register(
        "research_meta_fixed",
        "Fixed-effects meta-analysis. Pools effect sizes using inverse-variance weighting. Returns pooled effect, CI, and heterogeneity.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "effects": {"type": "array", "items": {"type": "number"}, "description": "Effect sizes per study"},
                "variances": {"type": "array", "items": {"type": "number"}, "description": "Variances (SE squared) per study"},
                "study_names": {"type": "array", "items": {"type": "string"}, "description": "Study names (optional)"},
                "effect_type": {
                    "type": "string",
                    "enum": ["mean_diff", "std_mean_diff", "log_odds_ratio", "correlation"],
                    "description": "Effect type (default mean_diff)",
                    "default": "mean_diff",
                },
            },
            "required": ["effects", "variances"],
        },
        _meta_fixed,
    )

    def _meta_random(args):
        return engine.meta_random(args)
    registry.register(
        "research_meta_random",
        "Random-effects meta-analysis (DerSimonian-Laird). Accounts for between-study heterogeneity. Returns pooled effect, tau-squared, prediction interval.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "effects": {"type": "array", "items": {"type": "number"}, "description": "Effect sizes per study"},
                "variances": {"type": "array", "items": {"type": "number"}, "description": "Variances per study"},
                "study_names": {"type": "array", "items": {"type": "string"}, "description": "Study names (optional)"},
                "estimator": {
                    "type": "string",
                    "enum": ["DL", "REML", "ML"],
                    "description": "Heterogeneity estimator (default DL)",
                    "default": "DL",
                },
            },
            "required": ["effects", "variances"],
        },
        _meta_random,
    )

    def _meta_heterogeneity(args):
        return engine.meta_heterogeneity(args)
    registry.register(
        "research_meta_heterogeneity",
        "Assess heterogeneity across studies: Q-statistic, I-squared, tau-squared, and H-squared.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "effects": {"type": "array", "items": {"type": "number"}, "description": "Effect sizes per study"},
                "variances": {"type": "array", "items": {"type": "number"}, "description": "Variances per study"},
                "study_names": {"type": "array", "items": {"type": "string"}, "description": "Study names (optional)"},
            },
            "required": ["effects", "variances"],
        },
        _meta_heterogeneity,
    )

    def _meta_bias(args):
        return engine.meta_bias(args)
    registry.register(
        "research_meta_bias",
        "Publication bias assessment: Egger's test, Begg's rank correlation, fail-safe N, and trim-and-fill adjustment.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "effects": {"type": "array", "items": {"type": "number"}, "description": "Effect sizes per study"},
                "ses": {"type": "array", "items": {"type": "number"}, "description": "Standard errors per study"},
                "study_names": {"type": "array", "items": {"type": "string"}, "description": "Study names (optional)"},
            },
            "required": ["effects", "ses"],
        },
        _meta_bias,
    )

    def _meta_subgroup(args):
        return engine.meta_subgroup(args)
    registry.register(
        "research_meta_subgroup",
        "Subgroup meta-analysis: compare effect sizes across predefined subgroups using Q-between test.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "effects": {"type": "array", "items": {"type": "number"}, "description": "Effect sizes per study"},
                "variances": {"type": "array", "items": {"type": "number"}, "description": "Variances per study"},
                "subgroups": {"type": "array", "items": {"type": "string"}, "description": "Subgroup labels per study"},
                "study_names": {"type": "array", "items": {"type": "string"}, "description": "Study names (optional)"},
            },
            "required": ["effects", "variances", "subgroups"],
        },
        _meta_subgroup,
    )


# =====================================================================
# Computational (5 tools)
# =====================================================================

def _register_computational_tools(registry, engine):
    def _topic_model(args):
        return engine.topic_model(args)
    registry.register(
        "research_topic_model",
        "Topic modeling using LDA or NMF. Extracts topics with top words, document-topic distributions, and perplexity.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "texts": {"type": "array", "items": {"type": "string"}, "description": "Text documents"},
                "n_topics": {"type": "integer", "description": "Number of topics (default 5)", "default": 5},
                "method": {
                    "type": "string",
                    "enum": ["lda", "nmf"],
                    "description": "Method (default lda)",
                    "default": "lda",
                },
                "max_features": {"type": "integer", "description": "Max vocabulary features (default 1000)", "default": 1000},
                "n_top_words": {"type": "integer", "description": "Top words per topic (default 10)", "default": 10},
            },
            "required": ["texts"],
        },
        _topic_model,
    )

    def _network(args):
        return engine.network_analysis(args)
    registry.register(
        "research_network",
        "Social network analysis: degree, betweenness, closeness, eigenvector centrality, density, and community detection.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "weight": {"type": "number", "default": 1.0},
                        },
                        "required": ["source", "target"],
                    },
                    "description": "Edge list",
                },
                "directed": {"type": "boolean", "description": "Directed graph (default false)", "default": False},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to compute (default degree, betweenness, closeness, density, communities)",
                },
            },
            "required": ["edges"],
        },
        _network,
    )

    def _abm(args):
        return engine.abm_simulate(args)
    registry.register(
        "research_abm",
        "Agent-based modeling simulation: Schelling segregation, SIR epidemic, or opinion dynamics. Returns time-series data and summary.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "n_agents": {"type": "integer", "description": "Number of agents (default 100)", "default": 100},
                "steps": {"type": "integer", "description": "Simulation steps (default 50)", "default": 50},
                "agent_type": {
                    "type": "string",
                    "enum": ["schelling", "epidemic", "opinion"],
                    "description": "Model type (default schelling)",
                    "default": "schelling",
                },
                "params": {
                    "type": "object",
                    "description": "Model-specific parameters (see docs)",
                },
                "seed": {"type": "integer", "description": "Random seed (default 42)", "default": 42},
            },
            "required": [],
        },
        _abm,
    )

    def _text_classify(args):
        return engine.text_classify(args)
    registry.register(
        "research_text_classify",
        "Text classification using TF-IDF + Logistic Regression or Count + Naive Bayes. Returns accuracy, per-class metrics, confusion matrix.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "texts": {"type": "array", "items": {"type": "string"}, "description": "Text documents"},
                "labels": {"type": "array", "items": {"type": "string"}, "description": "Class labels"},
                "method": {
                    "type": "string",
                    "enum": ["tfidf_lr", "count_nb"],
                    "description": "Classification method (default tfidf_lr)",
                    "default": "tfidf_lr",
                },
                "test_size": {"type": "number", "description": "Test set proportion (default 0.2)", "default": 0.2},
            },
            "required": ["texts", "labels"],
        },
        _text_classify,
    )

    def _embedding_analysis(args):
        return engine.embedding_analysis(args)
    registry.register(
        "research_embedding_analysis",
        "Document embedding and similarity analysis using TF-IDF vectors. Returns similarity matrix, clusters, and nearest neighbors.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "texts": {"type": "array", "items": {"type": "string"}, "description": "Text documents"},
                "method": {
                    "type": "string",
                    "enum": ["tfidf"],
                    "description": "Embedding method (default tfidf)",
                    "default": "tfidf",
                },
                "n_clusters": {"type": "integer", "description": "Number of clusters (default 3)", "default": 3},
                "similarity_threshold": {"type": "number", "description": "Similarity threshold for clustering (default 0.7)", "default": 0.7},
            },
            "required": ["texts"],
        },
        _embedding_analysis,
    )


# =====================================================================
# ML (10 tools)
# =====================================================================

def _register_ml_tools(registry, engine):
    def _preprocess(args):
        return engine.preprocess(args)
    registry.register(
        "research_ml_preprocess",
        "Preprocess data for ML: standardize, normalize, encode categoricals, impute missing values, select features.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "data": {
                    "type": "array",
                    "description": "Data as list of dicts or 2D array",
                },
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "params": {"type": "object"},
                        },
                    },
                    "description": "Preprocessing operations",
                },
            },
            "required": ["data"],
        },
        _preprocess,
    )

    def _train(args):
        return engine.train(args)
    registry.register(
        "research_ml_train",
        "Train a machine learning model (logistic regression, random forest, SVM, decision tree, KNN, naive Bayes). Returns model info and metrics.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "X_train": {
                    "type": "array",
                    "description": "Training features (2D array)",
                },
                "y_train": {
                    "type": "array",
                    "description": "Training labels",
                },
                "model_type": {
                    "type": "string",
                    "enum": ["logistic_regression", "random_forest", "svm", "decision_tree", "knn", "naive_bayes"],
                    "description": "Model type (default logistic_regression)",
                    "default": "logistic_regression",
                },
                "params": {
                    "type": "object",
                    "description": "Model hyperparameters (optional)",
                },
            },
            "required": ["X_train", "y_train"],
        },
        _train,
    )

    def _evaluate(args):
        return engine.evaluate(args)
    registry.register(
        "research_ml_evaluate",
        "Evaluate ML model: accuracy, precision, recall, F1, AUC-ROC, confusion matrix, per-class metrics.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_true": {"type": "array", "description": "True labels"},
                "y_pred": {"type": "array", "description": "Predicted labels"},
                "y_proba": {"type": "array", "items": {"type": "number"}, "description": "Predicted probabilities (optional)"},
            },
            "required": ["y_true", "y_pred"],
        },
        _evaluate,
    )

    def _crossval(args):
        return engine.crossval(args)
    registry.register(
        "research_ml_crossval",
        "K-fold cross-validation. Returns fold-wise and mean metrics with standard deviations.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "X": {"type": "array", "description": "Feature matrix"},
                "y": {"type": "array", "description": "Target labels"},
                "model_type": {
                    "type": "string",
                    "description": "Model type (default logistic_regression)",
                    "default": "logistic_regression",
                },
                "n_folds": {"type": "integer", "description": "Number of folds (default 5)", "default": 5},
                "scoring": {
                    "type": "string",
                    "enum": ["accuracy", "f1", "precision", "recall", "roc_auc"],
                    "description": "Scoring metric (default accuracy)",
                    "default": "accuracy",
                },
            },
            "required": ["X", "y"],
        },
        _crossval,
    )

    def _tune(args):
        return engine.tune(args)
    registry.register(
        "research_ml_tune",
        "Hyperparameter tuning via grid search with cross-validation. Returns best parameters and score.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "X": {"type": "array", "description": "Feature matrix"},
                "y": {"type": "array", "description": "Target labels"},
                "model_type": {
                    "type": "string",
                    "description": "Model type (default logistic_regression)",
                    "default": "logistic_regression",
                },
                "param_grid": {
                    "type": "object",
                    "description": "Parameter grid {param_name: [values]}",
                },
                "n_folds": {"type": "integer", "description": "CV folds (default 5)", "default": 5},
            },
            "required": ["X", "y", "param_grid"],
        },
        _tune,
    )

    def _compare(args):
        return engine.compare(args)
    registry.register(
        "research_ml_compare",
        "Compare multiple ML models on the same dataset. Returns ranked comparison table with metrics.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "X_train": {"type": "array", "description": "Training features"},
                "y_train": {"type": "array", "description": "Training labels"},
                "X_test": {"type": "array", "description": "Test features"},
                "y_test": {"type": "array", "description": "Test labels"},
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Model types to compare",
                },
            },
            "required": ["X_train", "y_train", "X_test", "y_test", "models"],
        },
        _compare,
    )

    def _feature_importance(args):
        return engine.feature_importance(args)
    registry.register(
        "research_ml_feature_importance",
        "Compute feature importance using permutation importance or model-based coefficients. Returns ranked features.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "X": {"type": "array", "description": "Feature matrix"},
                "y": {"type": "array", "description": "Target labels"},
                "feature_names": {"type": "array", "items": {"type": "string"}, "description": "Feature names (optional)"},
                "method": {
                    "type": "string",
                    "enum": ["model_based", "permutation"],
                    "description": "Method (default model_based)",
                    "default": "model_based",
                },
                "model_type": {
                    "type": "string",
                    "description": "Model type for importance (default random_forest)",
                    "default": "random_forest",
                },
            },
            "required": ["X", "y"],
        },
        _feature_importance,
    )

    def _automl(args):
        return engine.automl(args)
    registry.register(
        "research_ml_automl",
        "Automated ML pipeline: try multiple models with default hyperparameters and return the best one.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "X_train": {"type": "array", "description": "Training features"},
                "y_train": {"type": "array", "description": "Training labels"},
                "X_test": {"type": "array", "description": "Test features"},
                "y_test": {"type": "array", "description": "Test labels"},
                "metric": {
                    "type": "string",
                    "enum": ["accuracy", "f1", "roc_auc"],
                    "description": "Optimization metric (default accuracy)",
                    "default": "accuracy",
                },
                "max_time_seconds": {"type": "integer", "description": "Time budget (default 60)", "default": 60},
            },
            "required": ["X_train", "y_train"],
        },
        _automl,
    )

    def _learning_curve(args):
        return engine.learning_curve(args)
    registry.register(
        "research_ml_learning_curve",
        "Generate learning curve data: train and validation scores across varying training set sizes.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "X": {"type": "array", "description": "Feature matrix"},
                "y": {"type": "array", "description": "Target labels"},
                "model_type": {
                    "type": "string",
                    "description": "Model type (default logistic_regression)",
                    "default": "logistic_regression",
                },
                "train_sizes": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Training set fractions (default [0.1,0.2,0.4,0.6,0.8,1.0])",
                },
                "n_folds": {"type": "integer", "description": "CV folds (default 5)", "default": 5},
            },
            "required": ["X", "y"],
        },
        _learning_curve,
    )

    def _ensemble(args):
        return engine.ensemble(args)
    registry.register(
        "research_ml_ensemble",
        "Train ensemble model: voting classifier, bagging, or stacking. Returns ensemble metrics vs individual models.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "X_train": {"type": "array", "description": "Training features"},
                "y_train": {"type": "array", "description": "Training labels"},
                "X_test": {"type": "array", "description": "Test features"},
                "y_test": {"type": "array", "description": "Test labels"},
                "method": {
                    "type": "string",
                    "enum": ["voting", "bagging", "stacking"],
                    "description": "Ensemble method (default voting)",
                    "default": "voting",
                },
                "base_models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Base model types (default ['logistic_regression','random_forest','svm'])",
                },
            },
            "required": ["X_train", "y_train"],
        },
        _ensemble,
    )


# =====================================================================
# LLM (7 tools)
# =====================================================================

def _register_llm_tools(registry, engine):
    def _prompt_test(args):
        return engine.prompt_test(args)
    registry.register(
        "research_llm_prompt_test",
        "A/B test prompt variants. Sends each variant with test inputs to LLM, compares responses, latency, and match rates.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "variants": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "template": {"type": "string"},
                        },
                        "required": ["name", "template"],
                    },
                    "description": "Prompt variants to test",
                },
                "test_inputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                            "expected": {"type": "string"},
                        },
                        "required": ["input"],
                    },
                    "description": "Test inputs",
                },
                "model": {"type": "string", "description": "LLM model name (optional)"},
            },
            "required": ["variants", "test_inputs"],
        },
        _prompt_test,
    )

    def _evaluate(args):
        return engine.evaluate(args)
    registry.register(
        "research_llm_evaluate",
        "Evaluate LLM outputs: response length, word count, sentence count, and optional reference-based similarity metrics.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "prompts": {"type": "array", "items": {"type": "string"}, "description": "Original prompts"},
                "responses": {"type": "array", "items": {"type": "string"}, "description": "LLM responses to evaluate"},
                "references": {"type": "array", "items": {"type": "string"}, "description": "Reference answers (optional)"},
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evaluation criteria (optional)",
                },
            },
            "required": ["responses"],
        },
        _evaluate,
    )

    def _llm_judge(args):
        return engine.llm_judge(args)
    registry.register(
        "research_llm_judge",
        "LLM-as-Judge evaluation: uses an LLM to rate responses on a numeric scale for specified criteria (helpfulness, accuracy, safety).",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "responses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "response": {"type": "string"},
                        },
                        "required": ["prompt", "response"],
                    },
                    "description": "Items to judge",
                },
                "criteria": {
                    "oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
                    "description": "Evaluation criteria (default helpfulness)",
                    "default": "helpfulness",
                },
                "scale": {
                    "type": "string",
                    "enum": ["1-5", "1-10"],
                    "description": "Rating scale (default 1-5)",
                    "default": "1-5",
                },
                "judge_prompt": {"type": "string", "description": "Custom judge prompt template (optional)"},
                "reference": {"type": "string", "description": "Reference answer (optional)"},
                "model": {"type": "string", "description": "Judge model name (optional)"},
            },
            "required": ["responses"],
        },
        _llm_judge,
    )

    def _rag_eval(args):
        return engine.rag_eval(args)
    registry.register(
        "research_llm_rag_eval",
        "RAG evaluation: faithfulness (response-context alignment), relevance (context-query alignment), completeness (response-reference alignment).",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "queries": {"type": "array", "items": {"type": "string"}, "description": "User queries"},
                "contexts": {"type": "array", "items": {"type": "string"}, "description": "Retrieved contexts"},
                "responses": {"type": "array", "items": {"type": "string"}, "description": "Generated responses"},
                "references": {"type": "array", "items": {"type": "string"}, "description": "Reference answers"},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to compute (default faithfulness, relevance, completeness)",
                },
            },
            "required": ["queries", "contexts", "responses"],
        },
        _rag_eval,
    )

    def _benchmark(args):
        return engine.benchmark(args)
    registry.register(
        "research_llm_benchmark",
        "Run benchmark evaluation: call LLM on dataset, compare with expected outputs, compute exact match and contains-match accuracy.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "dataset": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                            "expected": {"type": "string"},
                        },
                        "required": ["input", "expected"],
                    },
                    "description": "Benchmark dataset",
                },
                "max_samples": {"type": "integer", "description": "Max samples to evaluate (optional)"},
            },
            "required": ["dataset"],
        },
        _benchmark,
    )

    def _quality_score(args):
        return engine.quality_score(args)
    registry.register(
        "research_llm_quality_score",
        "Output quality scoring: BLEU, ROUGE (unigram/bigram recall), Jaccard similarity, and Levenshtein edit distance ratio.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "generated": {"type": "array", "items": {"type": "string"}, "description": "Generated texts"},
                "references": {"type": "array", "items": {"type": "string"}, "description": "Reference texts"},
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics (default bleu, rouge, jaccard, edit_distance)",
                },
            },
            "required": ["generated", "references"],
        },
        _quality_score,
    )

    def _prompt_generate(args):
        return engine.prompt_generate(args)
    registry.register(
        "research_llm_prompt_generate",
        "Generate prompt variants for testing: rephrase, add constraints, or change role. Uses LLM if available, otherwise heuristic transforms.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "base_prompt": {"type": "string", "description": "Base prompt to create variants from"},
                "n_variants": {"type": "integer", "description": "Number of variants (default 3)", "default": 3},
                "variation_type": {
                    "type": "string",
                    "enum": ["rephrase", "add_constraints", "change_role"],
                    "description": "Variation style (default rephrase)",
                    "default": "rephrase",
                },
            },
            "required": ["base_prompt"],
        },
        _prompt_generate,
    )


# =====================================================================
# Visualization (10 tools)
# =====================================================================

def _register_viz_tools(registry, engine):
    def _plot(args):
        return engine.plot(args)
    registry.register(
        "research_plot",
        "Generate statistical plots: box, violin, histogram, bar, scatter, QQ, line. Saves to workspace figures directory.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "data": {
                    "type": "array",
                    "description": "Data array or list of arrays",
                },
                "type": {
                    "type": "string",
                    "enum": ["box", "violin", "hist", "bar", "scatter", "qq", "line"],
                    "description": "Plot type (default hist)",
                    "default": "hist",
                },
                "labels": {"type": "array", "items": {"type": "string"}, "description": "Labels for groups"},
                "title": {"type": "string", "description": "Plot title"},
                "x_label": {"type": "string", "description": "X-axis label"},
                "y_label": {"type": "string", "description": "Y-axis label"},
                "filename": {"type": "string", "description": "Output filename (default auto-generated)"},
                "bins": {
                    "oneOf": [{"type": "string"}, {"type": "integer"}],
                    "description": "Histogram bins (default auto)",
                    "default": "auto",
                },
            },
            "required": ["data"],
        },
        _plot,
    )

    def _forest_plot(args):
        return engine.forest_plot(args)
    registry.register(
        "research_forest_plot",
        "Generate forest plot for meta-analysis. Shows individual study effects with confidence intervals and optional pooled estimate.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "studies": {"type": "array", "items": {"type": "string"}, "description": "Study names"},
                "effects": {"type": "array", "items": {"type": "number"}, "description": "Effect sizes"},
                "cis_low": {"type": "array", "items": {"type": "number"}, "description": "Lower CI bounds"},
                "cis_high": {"type": "array", "items": {"type": "number"}, "description": "Upper CI bounds"},
                "x_label": {"type": "string", "description": "X-axis label (default Effect Size)", "default": "Effect Size"},
                "title": {"type": "string", "description": "Plot title (default Forest Plot)", "default": "Forest Plot"},
                "filename": {"type": "string", "description": "Output filename"},
                "pooled_effect": {"type": "number", "description": "Pooled effect estimate (optional)"},
                "pooled_ci_low": {"type": "number", "description": "Pooled CI lower bound (optional)"},
                "pooled_ci_high": {"type": "number", "description": "Pooled CI upper bound (optional)"},
            },
            "required": ["studies", "effects", "cis_low", "cis_high"],
        },
        _forest_plot,
    )

    def _funnel_plot(args):
        return engine.funnel_plot(args)
    registry.register(
        "research_funnel_plot",
        "Generate funnel plot for publication bias assessment. Plots effect sizes against standard errors.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "effects": {"type": "array", "items": {"type": "number"}, "description": "Effect sizes"},
                "se": {"type": "array", "items": {"type": "number"}, "description": "Standard errors"},
                "x_label": {"type": "string", "description": "X-axis label", "default": "Effect Size"},
                "y_label": {"type": "string", "description": "Y-axis label", "default": "Standard Error"},
                "title": {"type": "string", "description": "Plot title", "default": "Funnel Plot"},
                "filename": {"type": "string", "description": "Output filename"},
            },
            "required": ["effects", "se"],
        },
        _funnel_plot,
    )

    def _network_plot(args):
        return engine.network_plot(args)
    registry.register(
        "research_network_plot",
        "Network graph visualization with spring, circular, or Kamada-Kawai layouts.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "nodes": {"type": "array", "items": {"type": "string"}, "description": "Node names"},
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "weight": {"type": "number"},
                        },
                        "required": ["source", "target"],
                    },
                    "description": "Edge list",
                },
                "layout": {
                    "type": "string",
                    "enum": ["spring", "circular", "kamada"],
                    "description": "Layout algorithm (default spring)",
                    "default": "spring",
                },
                "title": {"type": "string", "description": "Plot title", "default": "Network Plot"},
                "filename": {"type": "string", "description": "Output filename"},
            },
            "required": ["nodes", "edges"],
        },
        _network_plot,
    )

    def _heatmap(args):
        return engine.heatmap(args)
    registry.register(
        "research_heatmap",
        "Correlation or data heatmap with annotations. Supports custom colormaps.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Data matrix",
                },
                "x_labels": {"type": "array", "items": {"type": "string"}, "description": "Column labels"},
                "y_labels": {"type": "array", "items": {"type": "string"}, "description": "Row labels"},
                "title": {"type": "string", "description": "Plot title", "default": "Heatmap"},
                "filename": {"type": "string", "description": "Output filename"},
                "annot": {"type": "boolean", "description": "Show values (default true)", "default": True},
                "cmap": {"type": "string", "description": "Colormap (default RdBu_r)", "default": "RdBu_r"},
            },
            "required": ["matrix"],
        },
        _heatmap,
    )

    def _roc_curve(args):
        return engine.roc_curve(args)
    registry.register(
        "research_roc_curve",
        "ROC curve plot with AUC. Shows true positive rate vs false positive rate.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "y_true": {"type": "array", "items": {"type": "number"}, "description": "True binary labels"},
                "y_score": {"type": "array", "items": {"type": "number"}, "description": "Predicted scores/probabilities"},
                "title": {"type": "string", "description": "Plot title", "default": "ROC Curve"},
                "filename": {"type": "string", "description": "Output filename"},
                "auc": {"type": "number", "description": "AUC value (optional, computed if omitted)"},
            },
            "required": ["y_true", "y_score"],
        },
        _roc_curve,
    )

    def _confusion_matrix(args):
        return engine.confusion_matrix(args)
    registry.register(
        "research_confusion_matrix",
        "Confusion matrix heatmap for classification results.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Confusion matrix",
                },
                "labels": {"type": "array", "items": {"type": "string"}, "description": "Class labels"},
                "title": {"type": "string", "description": "Plot title", "default": "Confusion Matrix"},
                "filename": {"type": "string", "description": "Output filename"},
            },
            "required": ["matrix"],
        },
        _confusion_matrix,
    )

    def _did_plot(args):
        return engine.did_plot(args)
    registry.register(
        "research_did_plot",
        "Difference-in-Differences parallel trends plot with treatment, control, and intervention line.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "time": {"type": "array", "items": {"type": "number"}, "description": "Time points"},
                "treatment": {"type": "array", "items": {"type": "number"}, "description": "Treatment group outcomes"},
                "control": {"type": "array", "items": {"type": "number"}, "description": "Control group outcomes"},
                "intervention_time": {"type": "number", "description": "Intervention time point"},
                "title": {"type": "string", "description": "Plot title", "default": "Difference-in-Differences"},
                "filename": {"type": "string", "description": "Output filename"},
            },
            "required": ["time", "treatment", "control", "intervention_time"],
        },
        _did_plot,
    )

    def _experiment_dashboard(args):
        return engine.experiment_dashboard(args)
    registry.register(
        "research_experiment_dashboard",
        "Multi-run experiment comparison dashboard. Bar charts comparing metrics across experiment runs.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "runs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "metrics": {"type": "object"},
                        },
                        "required": ["name", "metrics"],
                    },
                    "description": "Experiment runs with name and metrics",
                },
                "title": {"type": "string", "description": "Dashboard title", "default": "Experiment Dashboard"},
                "filename": {"type": "string", "description": "Output filename"},
            },
            "required": ["runs"],
        },
        _experiment_dashboard,
    )

    def _effect_size_plot(args):
        return engine.effect_size_plot(args)
    registry.register(
        "research_effect_size_plot",
        "Effect size visualization: horizontal error bars showing effect sizes with confidence intervals.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "effects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "number"},
                            "ci_low": {"type": "number"},
                            "ci_high": {"type": "number"},
                        },
                        "required": ["name", "value"],
                    },
                    "description": "Effect sizes with CIs",
                },
                "title": {"type": "string", "description": "Plot title", "default": "Effect Sizes"},
                "filename": {"type": "string", "description": "Output filename"},
            },
            "required": ["effects"],
        },
        _effect_size_plot,
    )


# =====================================================================
# Pipeline (6 tools)
# =====================================================================

def _register_pipeline_tools(registry, engine):
    def _load_data(args):
        return engine.load_data(args)
    registry.register(
        "research_load_data",
        "Load data from CSV, Excel, or JSON files within the workspace. Returns shape, columns, dtypes, and preview.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "path": {"type": "string", "description": "File path relative to workspace"},
                "format": {
                    "type": "string",
                    "enum": ["csv", "excel", "json", "auto"],
                    "description": "File format (default auto)",
                    "default": "auto",
                },
                "sheet": {
                    "oneOf": [{"type": "string"}, {"type": "integer"}],
                    "description": "Excel sheet name or index (default 0)",
                    "default": 0,
                },
                "columns": {"type": "array", "items": {"type": "string"}, "description": "Select specific columns (optional)"},
            },
            "required": ["path"],
        },
        _load_data,
    )

    def _validate_data(args):
        return engine.validate_data(args)
    registry.register(
        "research_validate_data",
        "Validate data quality: check missing values, outliers (IQR method), data types, and distributions.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "data": {
                    "type": "array",
                    "description": "Data as list of dicts",
                },
                "path": {"type": "string", "description": "File path (alternative to data)"},
                "checks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Checks to run (default missing, outliers, types, distribution)",
                },
            },
            "required": [],
        },
        _validate_data,
    )

    def _transform(args):
        return engine.transform(args)
    registry.register(
        "research_transform",
        "Transform data: standardize, normalize, encode, impute, filter, select, or rename columns.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "data": {
                    "type": "array",
                    "description": "Data as list of dicts",
                },
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["standardize", "normalize", "encode", "impute", "filter", "select", "rename"],
                            },
                            "params": {"type": "object"},
                        },
                        "required": ["type"],
                    },
                    "description": "Transformation operations",
                },
            },
            "required": ["data"],
        },
        _transform,
    )

    def _save_results(args):
        return engine.save_results(args)
    registry.register(
        "research_save_results",
        "Save results to JSON or CSV file in the workspace cache directory.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "data": {"description": "Data to save"},
                "path": {"type": "string", "description": "Output path (default results.json)", "default": "results.json"},
                "format": {
                    "type": "string",
                    "enum": ["json", "csv"],
                    "description": "Output format (default json)",
                    "default": "json",
                },
            },
            "required": ["data"],
        },
        _save_results,
    )

    def _export_report(args):
        return engine.export_report(args)
    registry.register(
        "research_export_report",
        "Export a research report in Markdown or HTML format with multiple sections.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "title": {"type": "string", "description": "Report title (default Research Report)", "default": "Research Report"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["heading", "content"],
                    },
                    "description": "Report sections",
                },
                "path": {"type": "string", "description": "Output path (default report.md)", "default": "report.md"},
                "format": {
                    "type": "string",
                    "enum": ["markdown", "html"],
                    "description": "Report format (default markdown)",
                    "default": "markdown",
                },
            },
            "required": [],
        },
        _export_report,
    )

    def _snapshot(args):
        return engine.snapshot(args)
    registry.register(
        "research_snapshot",
        "Create an experiment snapshot with data hashes and metadata for reproducibility.",
        {
            "type": "object",
            "properties": {
                "result_id": {"type": "string", "description": "Optional previous result ID to use as input data"},
                "label": {"type": "string", "description": "Snapshot label (default unnamed)", "default": "unnamed"},
                "data_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Data file paths to hash (optional)",
                },
                "code": {"type": "string", "description": "Associated code or script (optional)"},
            },
            "required": [],
        },
        _snapshot,
    )
