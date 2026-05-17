"""Method catalog: persistent index of all research methods."""
import json
import sqlite3
import uuid
from typing import Dict, List, Optional


METHOD_CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS method_catalog (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT DEFAULT '',
    keywords TEXT DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'known',
    tool_name TEXT,
    dependencies TEXT DEFAULT '[]',
    handler_code TEXT,
    tool_schema TEXT,
    source TEXT NOT NULL DEFAULT 'builtin',
    discovery_context TEXT,
    search_sources TEXT DEFAULT '[]',
    verified INTEGER NOT NULL DEFAULT 0,
    install_attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class MethodCatalog:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(METHOD_CATALOG_SCHEMA)
        self._seed_builtin_methods()

    def _seed_builtin_methods(self):
        """Pre-populate all built-in methods with status='installed'."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM method_catalog").fetchone()[0]
            if count > 0:
                return
            # Seed all 77 built-in tools
            builtin = [
                # Statistics
                ("research_describe", "Descriptive Statistics", "statistics", "Compute mean, SD, median, IQR, skewness, kurtosis, CI"),
                ("research_ttest", "T-Test", "statistics", "Independent, paired, Welch t-test with effect size"),
                ("research_anova", "ANOVA", "statistics", "One-way, repeated measures, Welch ANOVA"),
                ("research_chi_square", "Chi-Square Test", "statistics", "Independence, goodness-of-fit, Fisher exact"),
                ("research_nonparametric", "Non-Parametric Tests", "statistics", "Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman"),
                ("research_correlation", "Correlation Analysis", "statistics", "Pearson, Spearman, Kendall correlation"),
                ("research_regression", "Regression Analysis", "statistics", "Linear and multiple regression"),
                ("research_normality", "Normality Test", "statistics", "Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling"),
                ("research_effect_size", "Effect Size", "statistics", "Cohen's d, Hedges' g, eta-squared, odds ratio"),
                ("research_bayesian", "Bayesian T-Test", "statistics", "Bayesian t-test with BF10"),
                ("research_auto_test", "Auto Test Selection", "statistics", "Automatically select appropriate statistical test"),
                # Design
                ("research_factorial_design", "Factorial Design", "design", "Full, fractional, Plackett-Burman designs"),
                ("research_response_surface", "Response Surface Design", "design", "Box-Behnken, central composite designs"),
                ("research_latin_hypercube", "Latin Hypercube Sampling", "design", "LHS with various criteria"),
                ("research_power_analysis", "Power Analysis", "design", "Statistical power and sample size calculation"),
                ("research_random_assignment", "Random Assignment", "design", "Simple, block, stratified randomization"),
                # Causal
                ("research_did", "Difference-in-Differences", "causal", "DiD estimation with event study"),
                ("research_rdd", "Regression Discontinuity", "causal", "RDD with local polynomial regression"),
                ("research_iv", "Instrumental Variables", "causal", "2SLS with first-stage F and Hausman test"),
                ("research_psm", "Propensity Score Matching", "causal", "Nearest neighbor, stratification, IPW"),
                ("research_its", "Interrupted Time Series", "causal", "Segmented regression for policy evaluation"),
                ("research_mediation", "Mediation Analysis", "causal", "Baron-Kenny with bootstrap CI"),
                ("research_causal_effect", "Causal Effect Estimation", "causal", "ATE via OLS, IPW, AIPW"),
                ("research_sensitivity", "Sensitivity Analysis", "causal", "Oster bounds, Rosenbaum bounds"),
                # Survey
                ("research_cronbach", "Cronbach's Alpha", "survey", "Internal consistency reliability"),
                ("research_factor_analysis", "Factor Analysis", "survey", "EFA with varimax/oblimin rotation"),
                ("research_item_analysis", "Item Analysis", "survey", "Difficulty, discrimination, alpha-if-deleted"),
                ("research_sample_size", "Sample Size Calculation", "survey", "Cochran formula with design effect"),
                ("research_likert_analysis", "Likert Scale Analysis", "survey", "Frequency, median, top-box analysis"),
                # Qualitative
                ("research_thematic", "Thematic Analysis", "qualitative", "Automated thematic coding"),
                ("research_content", "Content Analysis", "qualitative", "Keyword frequency and co-occurrence"),
                ("research_grounded_code", "Grounded Theory Coding", "qualitative", "Open, axial, selective coding"),
                ("research_sentiment", "Sentiment Analysis", "qualitative", "VADER sentiment for interview coding"),
                ("research_coding_reliability", "Coding Reliability", "qualitative", "Cohen's Kappa inter-coder reliability"),
                # Meta-Analysis
                ("research_meta_fixed", "Fixed-Effect Meta-Analysis", "meta", "Inverse-variance weighted pooling"),
                ("research_meta_random", "Random-Effects Meta-Analysis", "meta", "DerSimonian-Laird with tau2, I2"),
                ("research_meta_heterogeneity", "Heterogeneity Analysis", "meta", "Q, I2, tau2, H statistics"),
                ("research_meta_bias", "Publication Bias Test", "meta", "Egger, Begg, fail-safe N"),
                ("research_meta_subgroup", "Subgroup Analysis", "meta", "Subgroup meta-analysis with Q decomposition"),
                # Computational
                ("research_topic_model", "Topic Modeling", "computational", "LDA, NMF topic extraction"),
                ("research_network", "Network Analysis", "computational", "Centrality, communities, density"),
                ("research_abm", "Agent-Based Modeling", "computational", "Schelling, SIR, opinion dynamics"),
                ("research_text_classify", "Text Classification", "computational", "TF-IDF + logistic/naive bayes"),
                ("research_embedding", "Embedding Analysis", "computational", "TF-IDF similarity and clustering"),
                # ML
                ("research_ml_preprocess", "ML Preprocessing", "ml", "Standardize, normalize, PCA, feature selection"),
                ("research_ml_train", "ML Training", "ml", "Train sklearn/xgboost/lightgbm models"),
                ("research_ml_evaluate", "ML Evaluation", "ml", "Accuracy, F1, AUC, MSE, R2 metrics"),
                ("research_ml_crossval", "Cross-Validation", "ml", "K-fold, stratified cross-validation"),
                ("research_ml_tune", "Hyperparameter Tuning", "ml", "Grid, random, Optuna search"),
                ("research_ml_compare", "Model Comparison", "ml", "Multi-model CV comparison with t-test"),
                ("research_ml_feature_importance", "Feature Importance", "ml", "Builtin, SHAP, permutation importance"),
                ("research_ml_automl", "AutoML", "ml", "FLAML automated machine learning"),
                ("research_ml_learning_curve", "Learning Curve", "ml", "Train/validation score vs sample size"),
                ("research_ml_ensemble", "Ensemble Methods", "ml", "Voting, bagging, stacking ensembles"),
                # LLM
                ("research_llm_prompt_test", "Prompt A/B Test", "llm", "Test and compare prompt variants"),
                ("research_llm_evaluate", "LLM Evaluation", "llm", "Evaluate LLM output quality"),
                ("research_llm_judge", "LLM-as-Judge", "llm", "LLM-based quality scoring"),
                ("research_llm_rag_eval", "RAG Evaluation", "llm", "Faithfulness, relevance, completeness"),
                ("research_llm_benchmark", "LLM Benchmark", "llm", "Dataset-based accuracy evaluation"),
                ("research_llm_quality_score", "Quality Scoring", "llm", "BLEU, ROUGE, Jaccard, edit distance"),
                ("research_llm_prompt_generate", "Prompt Generation", "llm", "Generate prompt variants"),
                # Visualization
                ("research_plot", "Statistical Plot", "visualization", "Box, violin, hist, bar, scatter, QQ, line"),
                ("research_forest_plot", "Forest Plot", "visualization", "Meta-analysis forest plot"),
                ("research_funnel_plot", "Funnel Plot", "visualization", "Publication bias funnel plot"),
                ("research_network_plot", "Network Plot", "visualization", "Graph visualization"),
                ("research_heatmap", "Heatmap", "visualization", "Correlation/data heatmap"),
                ("research_roc_curve", "ROC Curve", "visualization", "Receiver operating characteristic"),
                ("research_confusion_matrix", "Confusion Matrix", "visualization", "Classification heatmap"),
                ("research_did_plot", "DiD Plot", "visualization", "Parallel trends visualization"),
                ("research_experiment_dashboard", "Experiment Dashboard", "visualization", "Multi-run comparison"),
                ("research_effect_size_plot", "Effect Size Plot", "visualization", "Effect size with CI visualization"),
                # Pipeline
                ("research_load_data", "Data Loading", "pipeline", "CSV, Excel, JSON data loading"),
                ("research_validate_data", "Data Validation", "pipeline", "Missing values, outliers, type checks"),
                ("research_transform", "Data Transformation", "pipeline", "Standardize, encode, impute"),
                ("research_save_results", "Save Results", "pipeline", "JSON, CSV result export"),
                ("research_export_report", "Export Report", "pipeline", "Markdown, HTML report generation"),
                ("research_snapshot", "Experiment Snapshot", "pipeline", "Data+code environment snapshot"),
            ]
            for tool_name, name, category, desc in builtin:
                method_id = tool_name.replace("research_", "")
                conn.execute(
                    "INSERT OR IGNORE INTO method_catalog "
                    "(id, name, category, description, status, tool_name, source, verified) "
                    "VALUES (?, ?, ?, ?, 'installed', ?, 'builtin', 1)",
                    (method_id, name, category, desc, tool_name),
                )

    def add(self, method: dict) -> str:
        """Add a new method. Returns method_id."""
        method_id = method.get("id") or str(uuid.uuid4())[:12]
        name = method.get("name", "")
        category = method.get("category", "uncategorized")
        description = method.get("description", "")
        keywords = json.dumps(method.get("keywords", []))
        status = method.get("status", "known")
        tool_name = method.get("tool_name")
        dependencies = json.dumps(method.get("dependencies", []))
        handler_code = method.get("handler_code")
        tool_schema = json.dumps(method.get("tool_schema")) if method.get("tool_schema") else None
        source = method.get("source", "user")
        discovery_context = method.get("discovery_context")
        search_sources = json.dumps(method.get("search_sources", []))
        verified = int(method.get("verified", False))

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO method_catalog "
                "(id, name, category, description, keywords, status, tool_name, "
                "dependencies, handler_code, tool_schema, source, discovery_context, "
                "search_sources, verified, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "  datetime('now'), datetime('now'))",
                (method_id, name, category, description, keywords, status, tool_name,
                 dependencies, handler_code, tool_schema, source, discovery_context,
                 search_sources, verified),
            )
        return method_id

    def get(self, method_id: str) -> Optional[dict]:
        """Get method by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM method_catalog WHERE id = ?", (method_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def get_by_tool(self, tool_name: str) -> Optional[dict]:
        """Get method by its tool_name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM method_catalog WHERE tool_name = ?", (tool_name,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def search(self, query: str, category: str = None) -> List[dict]:
        """Search methods by keyword/name/description."""
        with self._connect() as conn:
            pattern = f"%{query}%"
            if category:
                rows = conn.execute(
                    "SELECT * FROM method_catalog "
                    "WHERE (name LIKE ? OR description LIKE ? OR keywords LIKE ?) "
                    "AND category = ? "
                    "ORDER BY updated_at DESC",
                    (pattern, pattern, pattern, category),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM method_catalog "
                    "WHERE name LIKE ? OR description LIKE ? OR keywords LIKE ? "
                    "ORDER BY updated_at DESC",
                    (pattern, pattern, pattern),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def update(self, method_id: str, **kwargs) -> bool:
        """Update method fields."""
        if not kwargs:
            return False

        allowed = {
            "name", "category", "description", "keywords", "status",
            "tool_name", "dependencies", "handler_code", "tool_schema",
            "source", "discovery_context", "search_sources", "verified",
            "install_attempts", "last_error",
        }

        sets = []
        vals = []
        for key, val in kwargs.items():
            if key not in allowed:
                continue
            if key in ("keywords", "dependencies", "search_sources", "tool_schema"):
                val = json.dumps(val) if not isinstance(val, str) else val
            if key == "verified":
                val = int(val)
            sets.append(f"{key} = ?")
            vals.append(val)

        if not sets:
            return False

        sets.append("updated_at = datetime('now')")
        vals.append(method_id)

        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE method_catalog SET {', '.join(sets)} WHERE id = ?",
                vals,
            )
            return cursor.rowcount > 0

    def list_methods(self, category: str = None, status: str = None, source: str = None) -> List[dict]:
        """List methods with optional filters."""
        conditions = []
        params = []
        if category:
            conditions.append("category = ?")
            params.append(category)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if source:
            conditions.append("source = ?")
            params.append(source)

        where = " AND ".join(conditions)
        sql = "SELECT * FROM method_catalog"
        if where:
            sql += f" WHERE {where}"
        sql += " ORDER BY category, name"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def activate_all(self, registry) -> int:
        """Load all status='installed' methods that have handler_code into the registry."""
        from sophia.research.discovery.sandbox import HandlerSandbox, SandboxViolation

        count = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM method_catalog "
                "WHERE status = 'installed' AND handler_code IS NOT NULL "
                "AND tool_schema IS NOT NULL AND tool_name IS NOT NULL"
            ).fetchall()

        for row in rows:
            method = self._row_to_dict(row)
            try:
                handler_code = method["handler_code"]
                tool_schema = method["tool_schema"]
                tool_name = method["tool_name"]

                # Validate and execute handler code in sandbox
                local_ns = HandlerSandbox.exec_safe(
                    handler_code,
                    {"json": json, "traceback": __import__("traceback"), "__builtins__": __builtins__},
                )

                # Find the handle function
                handler_fn = local_ns.get("handle")
                if handler_fn is None or not callable(handler_fn):
                    continue

                # Register with the tool registry
                registry.register(
                    tool_name,
                    tool_schema.get("description", method.get("description", "")),
                    tool_schema.get("parameters", {}),
                    handler_fn,
                )
                count += 1
            except SandboxViolation:
                # Skip methods that violate sandbox policy
                continue
            except Exception:
                # Skip methods that fail to activate
                continue

        return count

    def register_skill(self, skill_id: str, skill_name: str, workflow: List[dict]) -> int:
        """Register tools from a skill workflow into the catalog.

        Each tool referenced in the workflow is linked to the skill.
        Returns the number of tools registered.
        """
        count = 0
        with self._connect() as conn:
            for step in workflow:
                tool_name = step.get("tool")
                if not tool_name:
                    continue
                # Check if already in catalog
                row = conn.execute(
                    "SELECT id, status, discovery_context FROM method_catalog WHERE tool_name = ?", (tool_name,)
                ).fetchone()
                if row:
                    # Link existing method to this skill
                    ctx = {}
                    if row["discovery_context"]:
                        try:
                            ctx = json.loads(row["discovery_context"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    ctx["skill_id"] = skill_id
                    conn.execute(
                        "UPDATE method_catalog SET status = 'skill_linked', discovery_context = ? WHERE id = ?",
                        (json.dumps(ctx, ensure_ascii=False), row["id"]),
                    )
                else:
                    # Add as a new skill-linked method
                    method_id = tool_name.replace("research_", "")
                    conn.execute(
                        "INSERT OR IGNORE INTO method_catalog "
                        "(id, name, category, description, status, tool_name, source, discovery_context) "
                        "VALUES (?, ?, 'general', ?, 'skill_linked', ?, 'skill', ?)",
                        (method_id, f"Skill: {skill_name}", f"Linked from skill {skill_id}", tool_name, skill_id),
                    )
                count += 1
        return count

    def get_stats(self) -> dict:
        """Return counts by category and status."""
        stats = {"by_category": {}, "by_status": {}, "total": 0}

        with self._connect() as conn:
            # By category
            rows = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM method_catalog GROUP BY category"
            ).fetchall()
            for r in rows:
                stats["by_category"][r["category"]] = r["cnt"]

            # By status
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM method_catalog GROUP BY status"
            ).fetchall()
            for r in rows:
                stats["by_status"][r["status"]] = r["cnt"]

            # Total
            total = conn.execute("SELECT COUNT(*) FROM method_catalog").fetchone()[0]
            stats["total"] = total

        return stats

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict, deserializing JSON fields."""
        d = dict(row)
        for field in ("keywords", "dependencies", "search_sources"):
            val = d.get(field)
            if isinstance(val, str):
                try:
                    d[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        # tool_schema is stored as JSON string
        val = d.get("tool_schema")
        if isinstance(val, str):
            try:
                d["tool_schema"] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                d["tool_schema"] = None
        return d
