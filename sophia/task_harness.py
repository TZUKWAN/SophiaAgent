"""Deterministic task-quality harness prompts for SophiaAgent."""
from __future__ import annotations

from typing import Optional

EMPIRICAL_TERMS = [
    "实证",
    "回归",
    "因果",
    "稳健性",
    "可信性",
    "显著性",
    "内生性",
    "异质性",
    "机制检验",
    "中介效应",
    "调节效应",
    "双重差分",
    "断点回归",
    "工具变量",
    "倾向得分",
    "合成控制",
    "面板数据",
    "变量",
    "样本",
    "empirical",
    "regression",
    "causal",
    "causality",
    "robustness",
    "credibility",
    "validity",
    "identification",
    "endogeneity",
    "heterogeneity",
    "mechanism",
    "did",
    "iv",
    "rdd",
    "psm",
    "synthetic control",
    "panel data",
]

COMPLEX_TERMS = [
    "全面",
    "详细",
    "深入",
    "完整",
    "系统",
    "计划",
    "执行",
    "研究",
    "分析",
    "论文",
    "报告",
    "综述",
    "比较",
    "优化",
    "修复",
    "实现",
    "complete",
    "comprehensive",
    "detailed",
    "research",
    "analyze",
    "analysis",
    "paper",
    "report",
    "review",
    "compare",
    "implement",
    "fix",
]

SKILL_TERMS = [
    "重复",
    "固定流程",
    "模板",
    "技能",
    "复用",
    "workflow",
    "repeat",
    "template",
    "skill",
    "reuse",
]


def is_empirical_request(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in EMPIRICAL_TERMS)


def is_complex_task(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in COMPLEX_TERMS) or is_empirical_request(text)


def may_need_skill(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in SKILL_TERMS)


def build_task_harness_prompt(
    user_message: str,
    *,
    workspace_has_evidence: bool = False,
    empirical_preflight: Optional[str] = None,
) -> str:
    """Build a deterministic operating contract for non-trivial tasks."""
    if not is_complex_task(user_message):
        return ""

    lines = [
        "[Sophia task harness]",
        "This is a controlled execution task. Do not answer with a loose draft.",
        "",
        "Mandatory execution loop:",
        "1. Decompose the task into atomic steps before substantive output. Each step must "
        "have an action, required inputs, expected output, verification method, and done "
        "condition. For complex tasks, prefer 10-30 small steps over a vague 3-step plan.",
        "2. Execute exactly one step at a time. Do not mark a step complete until its done "
        "condition has concrete evidence.",
        "3. For each step, decide the required tool or skill. If a tool can verify, read, "
        "calculate, export, or inspect something, use the tool instead of guessing.",
        "4. Keep an evidence ledger linking each important "
        "claim to workspace evidence, tool output, or an explicit limitation.",
        "5. Run quality checks before finalizing. If a check fails, repair the output "
        "and check again. Do not stop at a checklist of unresolved work.",
        "6. Never fabricate data, citations, files, tool output, or completed actions.",
        "7. Final output must include concrete deliverables or artifact paths when the "
        "user asked for files.",
        "8. Include a completion proof: completed steps, tools used, artifacts created, "
        "quality gates passed, and any irreducible blocked items with reasons.",
        "",
        "Tool and skill policy:",
        "- Use `goal_create` for multi-step goals when available.",
        "- Use `skill_list` before `skill_execute` when the task may match an installed "
        "reusable workflow.",
        "- Use `skill_create` only after a real reusable workflow has been executed, not "
        "as a substitute for doing the current task.",
        "- If a tool fails, try the next reliable internal tool or fallback workflow and "
        "record the concrete failure. Do not ask the user to solve routine tool failure.",
        "",
        "Hook and loop policy:",
        "- Treat agent, tool, swarm, goal, context, and loop hooks as the execution trace.",
        "- Long, multi-stage, blocked, or recurring tasks must keep a goal/loop active until "
        "the user's real requested deliverable is completed or a verified external blocker "
        "is recorded.",
        "- Do not report a task as completed because a plan exists. Completion requires "
        "executed steps and verified outputs.",
        "",
        "Context continuity policy:",
        "- Maintain a handoff packet while working: current objective, decomposition, "
        "completed steps, pending steps, decisions, evidence ledger, tool results, "
        "artifacts, failures, retries, quality gates, and next action.",
        "- If the conversation is compressed or resumed in a new session, continue from "
        "the handoff packet without asking the user to restate prior context.",
    ]

    if workspace_has_evidence:
        lines.extend([
            "",
            "Workspace policy:",
            "- Local workspace evidence has priority over web search.",
            "- Use all relevant workspace material already surfaced by the workspace scanner.",
            "- Do not cite sources that are not in workspace evidence or verified tool output.",
        ])

    if is_empirical_request(user_message):
        lines.extend([
            "",
            "Empirical execution contract:",
            "- First use `empirical_workflow_plan` or the preflight plan below.",
            "- If real data, outcome, treatment/key variables, and design inputs are available, "
            "run `empirical_workflow_run` and then use recommended specialized `research_*` "
            "tools for the design.",
            "- Flexibly choose empirical modules from the actual design: descriptive/Table 1, "
            "OLS/GLM, panel/fixed effects, DID, IV, RDD, PSM, SCM, mediation, moderation, "
            "heterogeneity, sensitivity, ML, DML, causal forest, survey, qualitative, or "
            "meta-analysis. Use only modules that match real inputs and assumptions.",
            "- If inputs are missing, produce a blocked-but-auditable empirical plan with exact "
            "missing inputs. Do not invent a sample, coefficient, p-value, standard error, "
            "table, or robustness result.",
            "- Credibility checks are mandatory: data contract, missingness, outliers, variable "
            "coding, sample construction, model assumptions, identification assumptions, "
            "baseline specification, robustness checks, sensitivity checks, and skipped-check "
            "reasons.",
            "- Real analysis requires real data lineage: file path or result_id, row count, "
            "column names, variable definitions, transformations, dropped rows, estimator, "
            "standard errors, uncertainty, and reproducible artifact paths.",
            "- Report N, variable definitions, estimator, uncertainty, effect size, robustness "
            "status, and limitations whenever real estimation is executed.",
            "- If the user asks for a paper, convert real empirical outputs into a real paper "
            "with methods, results, tables, figures, credibility checks, limitations, and "
            "the requested export format. Do not write an empirical paper from fake numbers.",
        ])
        if empirical_preflight:
            lines.extend([
                "",
                "[Empirical preflight plan generated by Sophia]",
                empirical_preflight,
            ])

    if may_need_skill(user_message):
        lines.extend([
            "",
            "Reusable workflow policy:",
            "- Inspect installed skills before inventing a new workflow.",
            "- Execute matching skills only when their steps fit the current task.",
            "- If no skill matches, continue with direct tools and create a skill only after "
            "the real workflow has succeeded.",
        ])

    return "\n".join(lines)
