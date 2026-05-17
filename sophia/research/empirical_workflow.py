"""End-to-end empirical research workflow orchestration.

This module turns the scattered research tools into a single auditable
workflow.  It is inspired by the public Awesome-Agent-Skills empirical
playbooks, but implemented as Sophia-native logic: workspace guarded inputs,
ResultStore lineage, explicit missing-input reports, and no fabricated results.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sophia.research.workspace_guard import WorkspaceGuard


@dataclass
class EmpiricalStage:
    """A single stage in the empirical workflow."""

    stage_id: str
    title: str
    purpose: str
    required_inputs: List[str] = field(default_factory=list)
    recommended_tools: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    quality_gates: List[str] = field(default_factory=list)
    status: str = "planned"
    notes: List[str] = field(default_factory=list)


@dataclass
class EmpiricalWorkflowPlan:
    """Serializable plan for a full empirical analysis."""

    research_question: str
    mode: str
    ready_to_run: bool
    missing_inputs: List[str]
    stages: List[EmpiricalStage]
    recommended_methods: List[Dict[str, Any]] = field(default_factory=list)
    capability_audit: Dict[str, Any] = field(default_factory=dict)
    preserved_sophia_features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class EmpiricalWorkflowEngine:
    """Plan and run a transparent empirical workflow.

    The engine intentionally separates "plan" from "run".  If a request lacks
    real data or design variables, it returns a concrete workflow and missing
    inputs instead of inventing analyses.
    """

    _MODE_KEYWORDS = {
        "epi": [
            "epidemiology",
            "public health",
            "target trial",
            "iptw",
            "tmle",
            "survival",
            "cox",
            "strobe",
            "流行病",
            "公共卫生",
            "队列",
        ],
        "ml_causal": [
            "dml",
            "double machine learning",
            "causal forest",
            "cate",
            "policy learning",
            "uplift",
            "因果机器学习",
            "异质性处理效应",
        ],
    }

    _CAPABILITY_CATALOG = [
        {
            "capability": "explicit_8_step_empirical_pipeline",
            "status": "built_in",
            "tools": ["empirical_workflow_plan", "empirical_workflow_run"],
        },
        {
            "capability": "data_contract_and_sample_log",
            "status": "built_in",
            "tools": ["empirical_workflow_run", "research_validate_data"],
        },
        {
            "capability": "table_1_descriptives",
            "status": "built_in",
            "tools": ["empirical_workflow_run", "research_describe"],
        },
        {
            "capability": "progressive_regression_table",
            "status": "built_in_basic",
            "tools": ["empirical_workflow_run"],
            "note": "Uses NumPy OLS when inputs are supplied; pyfixest/stargazer remain optional for richer exports.",
        },
        {
            "capability": "diagnostics_missing_outlier_vif_corr",
            "status": "built_in_basic",
            "tools": ["empirical_workflow_run", "research_validate_data"],
        },
        {
            "capability": "did_iv_rdd_psm_scm_mediation_sensitivity",
            "status": "sophia_existing",
            "tools": [
                "research_did",
                "research_iv",
                "research_rdd",
                "research_psm",
                "research_scm",
                "research_mediation",
                "research_sensitivity",
            ],
        },
        {
            "capability": "pyfixest_hdfe_and_event_study",
            "status": "optional_dependency",
            "package": "pyfixest",
        },
        {
            "capability": "rdrobust_rdd_bandwidth_density",
            "status": "optional_dependency",
            "package": "rdrobust",
        },
        {
            "capability": "rddensity_mccrary_density",
            "status": "optional_dependency",
            "package": "rddensity",
        },
        {
            "capability": "econml_dml_causal_forest",
            "status": "optional_dependency",
            "package": "econml",
        },
        {
            "capability": "doubleml_debiased_ml",
            "status": "optional_dependency",
            "package": "doubleml",
        },
        {
            "capability": "causalml_meta_learners_matching",
            "status": "optional_dependency",
            "package": "causalml",
        },
        {
            "capability": "fairlearn_policy_fairness_audit",
            "status": "optional_dependency",
            "package": "fairlearn",
        },
        {
            "capability": "mapie_conformal_uncertainty",
            "status": "optional_dependency",
            "package": "mapie",
        },
        {
            "capability": "lifelines_survival",
            "status": "optional_dependency",
            "package": "lifelines",
        },
        {
            "capability": "publication_docx_latex_export",
            "status": "sophia_existing",
            "tools": ["research_export_report", "research_export_latex", "doc_export_docx"],
        },
    ]

    def __init__(
        self,
        workspace: str,
        *,
        store: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        advisor: Optional[Any] = None,
    ):
        self.workspace = os.path.realpath(workspace)
        self.guard = WorkspaceGuard(self.workspace)
        self.store = store
        self.pipeline = pipeline
        self.advisor = advisor

    def capability_audit(self, args: Optional[dict] = None) -> str:
        """Return Sophia empirical capabilities and optional dependency status."""
        capabilities = []
        missing_optional = []
        for item in self._CAPABILITY_CATALOG:
            entry = dict(item)
            package = entry.get("package")
            if package:
                available = importlib.util.find_spec(package) is not None
                entry["available"] = available
                if not available:
                    missing_optional.append(package)
            capabilities.append(entry)
        return json.dumps(
            {
                "capabilities": capabilities,
                "missing_optional_packages": sorted(set(missing_optional)),
                "note": (
                    "Built-in stages never fabricate missing estimator output. "
                    "Optional packages unlock richer specialized estimators."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    def plan(self, args: dict) -> str:
        """Create a full empirical workflow plan without executing analysis."""
        plan = self._build_plan(args, include_data_profile=True)
        return json.dumps(self._plan_to_dict(plan), ensure_ascii=False, indent=2, default=str)

    def run(self, args: dict) -> str:
        """Run the workflow stages that have real inputs available."""
        plan = self._build_plan(args, include_data_profile=True)
        result: Dict[str, Any] = {
            "plan": self._plan_to_dict(plan),
            "executed": False,
            "stage_outputs": {},
            "artifacts": [],
            "warnings": list(plan.warnings),
        }
        if not plan.ready_to_run:
            result["message"] = (
                "Workflow planned but not executed because required real inputs are missing."
            )
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)

        df_info = self._load_dataframe(args)
        if df_info.get("error"):
            result["warnings"].append(df_info["error"])
            result["message"] = "Data could not be loaded; no analysis was executed."
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)

        df: pd.DataFrame = df_info["df"]
        data_result_id = df_info.get("result_id")
        outputs = result["stage_outputs"]
        outputs["scope"] = self._scope_output(args, plan)
        outputs["data_contract"] = self._data_contract(df, args, data_result_id)
        outputs["data_quality"] = self._data_quality(df)
        outputs["descriptives"] = self._descriptives(df, args)
        outputs["diagnostics"] = self._diagnostics(df, args)
        outputs["estimation"] = self._baseline_estimation(df, args)
        outputs["robustness"] = self._robustness_plan(args, outputs["estimation"])
        outputs["further_analysis"] = self._further_analysis_plan(args)
        outputs["reporting"] = self._reporting_contract(args, outputs)

        result["executed"] = True
        result["result_id"] = self._store_result(result, args)
        if result["result_id"]:
            result["artifacts"].append({"kind": "result_store", "result_id": result["result_id"]})
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)

    def _build_plan(self, args: dict, *, include_data_profile: bool = False) -> EmpiricalWorkflowPlan:
        rq = str(args.get("research_question") or args.get("question") or "").strip()
        mode = self._infer_mode(args, rq)
        data_path = args.get("data_path") or args.get("path")
        result_id = args.get("result_id")
        missing = []
        if not rq:
            missing.append("research_question")
        if not data_path and not result_id and not args.get("data"):
            missing.append("data_path_or_result_id")

        outcome = args.get("outcome") or args.get("y") or args.get("outcome_col")
        treatment = args.get("treatment") or args.get("x") or args.get("treatment_col")
        if args.get("goal", "causal").lower() in {"causal", "impact", "effect"}:
            if not outcome:
                missing.append("outcome")
            if not treatment:
                missing.append("treatment")

        data_profile: Dict[str, Any] = {}
        if include_data_profile and (data_path or result_id or args.get("data")):
            data_profile = self._safe_data_profile(args)

        recommendations = self._recommend_methods(args, rq, data_profile)
        capability = json.loads(self.capability_audit({}))
        stages = self._default_stages(mode, args)

        warnings = []
        if data_profile.get("error"):
            warnings.append(data_profile["error"])
        if capability.get("missing_optional_packages"):
            warnings.append(
                "Optional empirical packages not installed: "
                + ", ".join(capability["missing_optional_packages"])
            )

        return EmpiricalWorkflowPlan(
            research_question=rq,
            mode=mode,
            ready_to_run=len(missing) == 0,
            missing_inputs=missing,
            stages=stages,
            recommended_methods=recommendations,
            capability_audit=capability,
            preserved_sophia_features=[
                "ResultStore lineage",
                "WorkspaceGuard path safety",
                "methodology_advise recommendations",
                "research_* causal/statistical tools",
                "LaTeX/Word/Markdown export path",
                "no fabricated data or citations",
            ],
            warnings=warnings,
        )

    def _default_stages(self, mode: str, args: dict) -> List[EmpiricalStage]:
        estimator_tools = ["methodology_advise", "research_regression"]
        if mode == "epi":
            estimator_tools += ["research_sensitivity"]
        elif mode == "ml_causal":
            estimator_tools += ["research_ml_train", "research_ml_evaluate"]
        else:
            estimator_tools += [
                "research_did",
                "research_iv",
                "research_rdd",
                "research_psm",
                "research_scm",
            ]

        return [
            EmpiricalStage(
                "scope",
                "Pre-analysis plan",
                "Clarify research question, estimand, hypotheses, unit of analysis, and identification logic.",
                ["research_question"],
                ["empirical_workflow_plan", "methodology_advise"],
                ["pap.json", "estimand summary", "assumption checklist"],
                ["Research question is explicit", "Outcome/treatment are not guessed"],
            ),
            EmpiricalStage(
                "data_contract",
                "Sample log and data contract",
                "Load real data, record shape, columns, row exclusions, file hashes, and required variable availability.",
                ["data_path_or_result_id"],
                ["research_load_data", "research_snapshot"],
                ["data_contract.json", "sample_log"],
                ["No silent row drops", "All required columns exist"],
            ),
            EmpiricalStage(
                "data_quality",
                "Data cleaning and variable construction",
                "Audit missingness, outliers, dtypes, transformations, lags, fixed-effect keys, and coding decisions.",
                ["outcome", "treatment"],
                ["research_validate_data", "research_transform"],
                ["cleaning_report", "variable_codebook"],
                ["Transformations are explicit", "Missing values are not filled silently"],
            ),
            EmpiricalStage(
                "descriptives",
                "Descriptive statistics and Table 1",
                "Produce summary statistics, group balance if treatment is present, correlations, and basic distributions.",
                ["data_path_or_result_id"],
                ["research_describe", "research_plot", "research_heatmap"],
                ["table1_descriptives", "balance_summary", "correlation_summary"],
                ["N is reported", "Treated/control balance is visible when applicable"],
            ),
            EmpiricalStage(
                "diagnostics",
                "Diagnostics and identification checks",
                "Check normality, multicollinearity, missingness, support/overlap, panel keys, and design-specific assumptions.",
                ["outcome"],
                ["research_normality", "research_correlation", "research_did_plot"],
                ["diagnostic_log", "assumption_warnings"],
                ["Assumptions are checked before model interpretation"],
            ),
            EmpiricalStage(
                "estimation",
                "Baseline empirical modeling",
                "Run the best feasible estimator family and progressive baseline specifications when variables are available.",
                ["outcome", "treatment"],
                estimator_tools,
                ["main_results_table", "effect_size_summary"],
                ["Estimator matches design", "SE/CI/N are reported"],
            ),
            EmpiricalStage(
                "robustness",
                "Robustness and sensitivity battery",
                "Plan or run alternative samples, controls, functional forms, placebo checks, clustering, and sensitivity tests.",
                ["baseline_result"],
                ["research_sensitivity", "research_snapshot"],
                ["robustness_matrix", "specification_curve_plan"],
                ["Skipped checks explain why", "No cherry-picked specification"],
            ),
            EmpiricalStage(
                "further_analysis",
                "Mechanism, heterogeneity, and extensions",
                "Plan or run mediator, subgroup, moderation, and CATE analyses when supported by data.",
                ["baseline_result"],
                ["research_mediation", "research_effect_size_plot"],
                ["mechanism_plan", "heterogeneity_plan"],
                ["Subgroups are pre-specified or clearly exploratory"],
            ),
            EmpiricalStage(
                "reporting",
                "Publication-ready reporting",
                "Assemble outputs into reproducible tables, figures, report text, and export-ready artifacts.",
                ["stage_outputs"],
                ["research_export_report", "research_export_latex", "doc_export_docx"],
                ["report.md", "tables", "figures", "replication_manifest"],
                ["Every missing deliverable has a stated reason"],
            ),
        ]

    def _infer_mode(self, args: dict, rq: str) -> str:
        explicit = str(args.get("mode", "")).strip().lower()
        if explicit in {"econ", "applied_econ", "epi", "ml_causal"}:
            return explicit
        text = " ".join([rq, str(args.get("design", "")), str(args.get("constraints", ""))]).lower()
        for mode, keywords in self._MODE_KEYWORDS.items():
            if any(k in text for k in keywords):
                return mode
        return "applied_econ"

    def _recommend_methods(self, args: dict, rq: str, data_profile: dict) -> List[Dict[str, Any]]:
        if self.advisor is None or not rq:
            return []
        data_description = dict(args.get("data_description") or {})
        if data_profile and not data_profile.get("error"):
            data_description.update({
                "N": data_profile.get("rows", 0),
                "variables": data_profile.get("columns", 0),
                "type": data_profile.get("data_type", "tabular"),
                "units": data_profile.get("units", 0),
                "periods": data_profile.get("periods", 0),
            })
        try:
            raw = self.advisor.advise({
                "research_question": rq,
                "data_description": data_description,
                "design": args.get("design", "observational"),
                "outcome_type": args.get("outcome_type", "continuous"),
                "constraints": args.get("constraints", []),
            })
            return json.loads(raw).get("recommended_methods", [])
        except Exception as exc:  # noqa: BLE001
            return [{"error": f"methodology_advise failed: {exc}"}]

    def _load_dataframe(self, args: dict) -> Dict[str, Any]:
        if args.get("result_id") and self.store is not None:
            try:
                return {"df": self.store.get_dataframe(args["result_id"]), "result_id": args["result_id"]}
            except Exception as exc:  # noqa: BLE001
                return {"error": f"Failed to load result_id {args['result_id']}: {exc}"}
        if args.get("data") is not None:
            try:
                return {"df": pd.DataFrame(args["data"])}
            except Exception as exc:  # noqa: BLE001
                return {"error": f"Failed to materialize inline data: {exc}"}
        path = args.get("data_path") or args.get("path")
        if not path:
            return {"error": "No data_path, data, or result_id provided."}
        if self.pipeline is not None:
            loaded = json.loads(self.pipeline.load_data({"path": path, "format": args.get("format", "auto")}))
            if "error" in loaded:
                return {"error": loaded["error"]}
            rid = loaded.get("result_id")
            if rid and self.store is not None:
                return {"df": self.store.get_dataframe(rid), "result_id": rid, "preview": loaded}
        try:
            resolved = self.guard.resolve_read(path)
            ext = os.path.splitext(resolved)[1].lower()
            if ext in {".xlsx", ".xls"}:
                df = pd.read_excel(resolved, sheet_name=args.get("sheet", 0))
            elif ext == ".json":
                df = pd.read_json(resolved)
            else:
                df = pd.read_csv(resolved)
            return {"df": df}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Failed to load data: {exc}"}

    def _safe_data_profile(self, args: dict) -> Dict[str, Any]:
        loaded = self._load_dataframe(args)
        if loaded.get("error"):
            return {"error": loaded["error"]}
        df = loaded["df"]
        unit = args.get("unit") or args.get("unit_col")
        time = args.get("time") or args.get("time_col")
        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(map(str, df.columns)),
            "numeric_columns": list(map(str, df.select_dtypes(include=[np.number]).columns)),
            "data_type": "panel" if unit in df.columns and time in df.columns else "tabular",
            "units": int(df[unit].nunique()) if unit in df.columns else 0,
            "periods": int(df[time].nunique()) if time in df.columns else 0,
        }

    def _scope_output(self, args: dict, plan: EmpiricalWorkflowPlan) -> Dict[str, Any]:
        return {
            "research_question": plan.research_question,
            "mode": plan.mode,
            "outcome": args.get("outcome") or args.get("y") or args.get("outcome_col"),
            "treatment": args.get("treatment") or args.get("x") or args.get("treatment_col"),
            "design": args.get("design", "observational"),
            "recommended_methods": plan.recommended_methods[:3],
        }

    def _data_contract(self, df: pd.DataFrame, args: dict, result_id: Optional[str]) -> Dict[str, Any]:
        required = [
            args.get("outcome") or args.get("y") or args.get("outcome_col"),
            args.get("treatment") or args.get("x") or args.get("treatment_col"),
            args.get("unit") or args.get("unit_col"),
            args.get("time") or args.get("time_col"),
        ]
        required = [c for c in required if c]
        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(map(str, df.columns)),
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
            "required_columns": required,
            "missing_required_columns": [c for c in required if c not in df.columns],
            "result_id": result_id,
        }

    def _data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        missing = {
            str(c): {"count": int(df[c].isna().sum()), "percent": round(float(df[c].isna().mean() * 100), 2)}
            for c in df.columns
        }
        numeric = df.select_dtypes(include=[np.number])
        outliers = {}
        for c in numeric.columns:
            s = numeric[c].dropna()
            if s.empty:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers[str(c)] = int(((s < lo) | (s > hi)).sum())
        return {
            "missing": missing,
            "outlier_counts_iqr": outliers,
            "issue_count": sum(1 for v in missing.values() if v["count"] > 0)
            + sum(1 for v in outliers.values() if v > 0),
        }

    def _descriptives(self, df: pd.DataFrame, args: dict) -> Dict[str, Any]:
        numeric = df.select_dtypes(include=[np.number])
        summary = {}
        for c in numeric.columns:
            s = numeric[c].dropna()
            if s.empty:
                continue
            summary[str(c)] = {
                "n": int(s.size),
                "mean": round(float(s.mean()), 6),
                "sd": round(float(s.std()), 6) if s.size > 1 else None,
                "min": round(float(s.min()), 6),
                "p25": round(float(s.quantile(0.25)), 6),
                "median": round(float(s.median()), 6),
                "p75": round(float(s.quantile(0.75)), 6),
                "max": round(float(s.max()), 6),
            }
        treatment = args.get("treatment") or args.get("x") or args.get("treatment_col")
        balance = {}
        if treatment in df.columns and df[treatment].nunique(dropna=True) <= 10:
            for c in numeric.columns:
                if c == treatment:
                    continue
                grouped = df.groupby(treatment)[c].agg(["count", "mean", "std"]).reset_index()
                balance[str(c)] = json.loads(grouped.to_json(orient="records", default_handler=str))
        return {"table1": summary, "balance_by_treatment": balance}

    def _diagnostics(self, df: pd.DataFrame, args: dict) -> Dict[str, Any]:
        numeric = df.select_dtypes(include=[np.number])
        corr_warnings = []
        if numeric.shape[1] >= 2:
            corr = numeric.corr(numeric_only=True).abs()
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i + 1:]:
                    val = corr.loc[c1, c2]
                    if pd.notna(val) and val >= 0.9:
                        corr_warnings.append({"variables": [str(c1), str(c2)], "abs_corr": round(float(val), 4)})
        outcome = args.get("outcome") or args.get("y") or args.get("outcome_col")
        outcome_summary = None
        if outcome in df.columns and pd.api.types.is_numeric_dtype(df[outcome]):
            s = df[outcome].dropna()
            outcome_summary = {
                "n": int(s.size),
                "skew": round(float(s.skew()), 6) if s.size > 2 else None,
                "kurtosis": round(float(s.kurtosis()), 6) if s.size > 3 else None,
            }
        return {
            "high_correlation_pairs": corr_warnings,
            "outcome_distribution": outcome_summary,
            "notes": [
                "Full heteroskedasticity/autocorrelation tests require model residuals.",
                "DID/event-study diagnostics require unit, time, treatment timing, and pre-period data.",
            ],
        }

    def _baseline_estimation(self, df: pd.DataFrame, args: dict) -> Dict[str, Any]:
        y_col = args.get("outcome") or args.get("y") or args.get("outcome_col")
        t_col = args.get("treatment") or args.get("x") or args.get("treatment_col")
        covariates = args.get("covariates") or args.get("controls") or []
        if isinstance(covariates, str):
            covariates = [c.strip() for c in covariates.split(",") if c.strip()]
        needed = [y_col, t_col] + list(covariates)
        missing = [c for c in needed if not c or c not in df.columns]
        if missing:
            return {
                "status": "skipped",
                "reason": "Required estimation columns are missing.",
                "missing_columns": missing,
            }
        model_specs = [
            {"name": "M1", "x_cols": [t_col]},
            {"name": "M2", "x_cols": [t_col] + list(covariates)},
        ]
        rows = []
        for spec in model_specs:
            fit = self._ols(df, y_col, spec["x_cols"], focal=t_col)
            fit["model"] = spec["name"]
            fit["x_cols"] = spec["x_cols"]
            rows.append(fit)
        return {
            "status": "completed",
            "estimator": "numpy_ols",
            "note": (
                "This built-in baseline is a transparent OLS fallback. "
                "Use specialized Sophia causal tools for DID/IV/RD/PSM/SCM when design inputs are present."
            ),
            "models": rows,
        }

    def _ols(self, df: pd.DataFrame, y_col: str, x_cols: List[str], *, focal: str) -> Dict[str, Any]:
        data = df[[y_col] + x_cols].dropna()
        if len(data) <= len(x_cols) + 1:
            return {"status": "skipped", "reason": "Insufficient complete cases", "n": int(len(data))}
        y = data[y_col].astype(float).to_numpy()
        x = data[x_cols].astype(float).to_numpy()
        x = np.column_stack([np.ones(len(x)), x])
        names = ["Intercept"] + x_cols
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            fitted = x @ beta
            resid = y - fitted
            dof = len(y) - x.shape[1]
            sigma2 = float((resid @ resid) / dof) if dof > 0 else float("nan")
            cov = sigma2 * np.linalg.pinv(x.T @ x)
            se = np.sqrt(np.diag(cov))
            tvals = beta / se
            sst = float(((y - y.mean()) ** 2).sum())
            ssr = float((resid ** 2).sum())
            r2 = 1 - ssr / sst if sst else None
            idx = names.index(focal) if focal in names else None
            return {
                "status": "completed",
                "n": int(len(y)),
                "coefficients": {
                    name: {
                        "coef": round(float(beta[i]), 6),
                        "se": round(float(se[i]), 6) if math.isfinite(float(se[i])) else None,
                        "t": round(float(tvals[i]), 6) if math.isfinite(float(tvals[i])) else None,
                    }
                    for i, name in enumerate(names)
                },
                "focal_effect": {
                    "variable": focal,
                    "coef": round(float(beta[idx]), 6) if idx is not None else None,
                    "se": round(float(se[idx]), 6) if idx is not None and math.isfinite(float(se[idx])) else None,
                },
                "r_squared": round(float(r2), 6) if r2 is not None else None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc), "n": int(len(data))}

    def _robustness_plan(self, args: dict, estimation: dict) -> Dict[str, Any]:
        checks = [
            "alternative covariate sets",
            "complete-case versus missingness-sensitive sample",
            "outlier trimming/winsorization sensitivity",
            "alternative functional forms",
            "placebo outcome or placebo treatment timing when available",
            "cluster/robust standard error comparison when panel keys exist",
        ]
        return {
            "status": "planned" if estimation.get("status") == "completed" else "blocked",
            "checks": checks,
            "blocked_reason": None if estimation.get("status") == "completed" else "Baseline estimation did not complete.",
        }

    def _further_analysis_plan(self, args: dict) -> Dict[str, Any]:
        return {
            "mechanism": args.get("mediators") or "not supplied",
            "heterogeneity": args.get("subgroups") or args.get("heterogeneity") or "not supplied",
            "moderation": args.get("moderators") or "not supplied",
            "note": "These analyses are exploratory unless variables were specified before estimation.",
        }

    def _reporting_contract(self, args: dict, outputs: dict) -> Dict[str, Any]:
        return {
            "required_tables": [
                "Table 1 descriptive/balance table",
                "Main results table",
                "Robustness table",
                "Mechanism/heterogeneity table when variables exist",
            ],
            "required_figures": [
                "Distribution/trend figure when variables support it",
                "Coefficient or effect-size figure",
                "Design-specific identification figure when applicable",
            ],
            "missing_deliverables_policy": "Any skipped table or figure must include a concrete reason.",
            "result_store_ready": bool(self.store is not None),
        }

    def _store_result(self, result: Dict[str, Any], args: dict) -> Optional[str]:
        if self.store is None:
            return None
        try:
            parents = [args["result_id"]] if args.get("result_id") and self.store.exists(args["result_id"]) else []
            return self.store.store(
                result,
                kind="result",
                tool="empirical_workflow_run",
                params={k: v for k, v in args.items() if k != "data"},
                parents=parents,
            )
        except Exception:
            return None

    @staticmethod
    def _plan_to_dict(plan: EmpiricalWorkflowPlan) -> Dict[str, Any]:
        out = asdict(plan)
        out["stages"] = [asdict(stage) for stage in plan.stages]
        return out
