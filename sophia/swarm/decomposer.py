"""Task decomposition for automatic swarm execution."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Callable, Dict, List, Optional

from sophia.swarm.models import AgentSpec, Stage, SwarmDecision, SwarmPlan
from sophia.swarm.roles import RoleTemplate, RoleTemplateBank

logger = logging.getLogger(__name__)

TASK_PATTERNS: Dict[str, Dict] = {
    "literature_review": {
        "keywords": ["文献综述", "综述", "研究现状", "文献回顾", "文献梳理"],
        "workflow": "pipeline",
        "stages": [
            {"parallel": True, "roles": ["literature_searcher", "methodologist"]},
            {"parallel": False, "roles": ["writer"]},
            {"parallel": False, "roles": ["reviewer"]},
        ],
        "coordinator": "整合检索、方法和评审意见，形成严谨的文献综述答复。",
    },
    "paper_writing": {
        "keywords": ["写论文", "撰写论文", "论文写作", "写一篇", "学术论文"],
        "workflow": "mixed",
        "stages": [
            {"parallel": True, "roles": ["literature_searcher", "methodologist"]},
            {"parallel": False, "roles": ["writer"]},
            {"parallel": True, "roles": ["reviewer", "critic"]},
            {"parallel": False, "roles": ["synthesizer"]},
        ],
        "coordinator": "合并各专家结果，输出一份结构完整、证据边界清楚的写作成果。",
    },
    "data_analysis": {
        "keywords": ["分析数据", "数据分析", "统计分析", "回归分析", "可视化"],
        "workflow": "mixed",
        "stages": [
            {"parallel": True, "roles": ["data_analyst", "methodologist"]},
            {"parallel": False, "roles": ["critic"]},
            {"parallel": False, "roles": ["writer"]},
        ],
        "coordinator": "整合方法、结果和局限，形成可复核的数据分析答复。",
    },
    "comparative_study": {
        "keywords": ["对比", "比较", "差异", "异同"],
        "workflow": "mixed",
        "stages": [
            {"parallel": True, "roles": ["literature_searcher", "data_analyst"]},
            {"parallel": False, "roles": ["critic"]},
            {"parallel": False, "roles": ["synthesizer"]},
        ],
        "coordinator": "综合各维度比较结果，明确相同点、差异点、原因和不确定性。",
    },
    "research_design": {
        "keywords": ["研究设计", "研究方案", "实验设计", "方法设计"],
        "workflow": "pipeline",
        "stages": [
            {"parallel": False, "roles": ["methodologist"]},
            {"parallel": False, "roles": ["critic"]},
            {"parallel": False, "roles": ["writer"]},
            {"parallel": False, "roles": ["reviewer"]},
        ],
        "coordinator": "输出完整研究设计方案，并保留方法前提、风险和修订建议。",
    },
}

DECOMPOSITION_PROMPT = """Decompose this request into a 2-5 agent swarm plan.

Return only JSON:
{
  "workflow": "parallel|pipeline|mixed",
  "stages": [
    {"stage_id": "stage_1", "parallel": true,
     "agents": [{"role_id": "writer", "task_prompt": "Chinese task"}]}
  ],
  "coordinator_prompt": "Chinese synthesis instructions"
}

Available roles: literature_searcher, data_analyst, writer, reviewer, methodologist,
critic, synthesizer, citation_manager.

Task:
{message}
"""


def _match_task_type(message: str) -> Optional[str]:
    for task_type, config in TASK_PATTERNS.items():
        if any(keyword in message for keyword in config["keywords"]):
            return task_type
    return None


def _agent_id(role_id: str) -> str:
    return f"{role_id}_{uuid.uuid4().hex[:4]}"


def _build_task_prompt(role: RoleTemplate, original_query: str, stage_index: int) -> str:
    return (
        f"原始用户需求：{original_query}\n\n"
        f"你的角色：{role.name}\n"
        f"角色职责：{role.description}\n\n"
        f"专属要求：{role.system_prompt}\n\n"
        f"当前执行阶段：第 {stage_index} 阶段。\n"
        "请只完成与你角色相关的真实子任务。不得编造事实、数据、文献或工具结果。"
        "如果信息不足，请明确说明缺口和下一步需要什么。输出要结构化，方便后续专家读取。"
    )


class TaskDecomposer:
    """Builds executable swarm plans from a user request."""

    def __init__(
        self,
        llm_call: Optional[Callable[[str], str]] = None,
        role_bank: Optional[RoleTemplateBank] = None,
        use_llm: bool = True,
    ):
        self.llm_call = llm_call
        self.role_bank = role_bank or RoleTemplateBank()
        self.use_llm = use_llm

    def decompose(self, message: str, decision: SwarmDecision) -> SwarmPlan:
        rule_plan = self._rule_decompose(message)
        if rule_plan:
            return rule_plan

        if self.use_llm and self.llm_call:
            llm_plan = self._llm_decompose(message)
            if llm_plan:
                return llm_plan

        return self._fallback_plan(message, decision)

    def _rule_decompose(self, message: str) -> Optional[SwarmPlan]:
        task_type = _match_task_type(message)
        if not task_type:
            return None
        config = TASK_PATTERNS[task_type]
        return self._build_plan_from_stage_config(
            message=message,
            workflow=config["workflow"],
            stage_configs=config["stages"],
            coordinator_prompt=config["coordinator"],
        )

    def _llm_decompose(self, message: str) -> Optional[SwarmPlan]:
        try:
            raw = self.llm_call(DECOMPOSITION_PROMPT.replace("{message}", message))
            match = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            stages: List[Stage] = []
            for index, stage_data in enumerate(data.get("stages", []), start=1):
                agents: List[AgentSpec] = []
                for agent_data in stage_data.get("agents", []):
                    role_id = str(agent_data.get("role_id", "")).strip()
                    role = self.role_bank.get(role_id)
                    if not role:
                        continue
                    agents.append(
                        AgentSpec(
                            agent_id=_agent_id(role_id),
                            role_id=role_id,
                            task_prompt=str(agent_data.get("task_prompt") or _build_task_prompt(role, message, index)),
                            tools=list(role.allowed_tools if role.needs_tools else []),
                            system_prompt=role.system_prompt,
                        )
                    )
                if agents:
                    stages.append(
                        Stage(
                            stage_id=str(stage_data.get("stage_id") or f"stage_{index}"),
                            parallel=bool(stage_data.get("parallel", True)),
                            agents=agents,
                            depends_on=[stages[-1].stage_id] if stages else [],
                        )
                    )
            if stages:
                return SwarmPlan(
                    workflow=str(data.get("workflow") or "mixed"),
                    stages=stages,
                    coordinator_prompt=str(data.get("coordinator_prompt") or "综合各专家结果形成最终答复。"),
                    original_query=message,
                )
        except Exception as exc:
            logger.warning("LLM decomposition failed: %s", exc)
        return None

    def _fallback_plan(self, message: str, decision: SwarmDecision) -> SwarmPlan:
        role_ids = decision.recommended_roles or [
            role.role_id for role in self.role_bank.match_for_task(message, limit=4)
        ]
        if not role_ids:
            role_ids = ["writer", "critic"]
        if "reviewer" not in role_ids and len(role_ids) < 5:
            role_ids.append("reviewer")
        roles = [self.role_bank.get(role_id) for role_id in role_ids[:5]]
        roles = [role for role in roles if role is not None]
        stage_configs = [{"parallel": True, "roles": [role.role_id for role in roles]}]
        return self._build_plan_from_stage_config(
            message=message,
            workflow=decision.workflow or "parallel",
            stage_configs=stage_configs,
            coordinator_prompt="整合自动分配的专家结果，形成一份可靠、清晰、完整的最终答复。",
        )

    def _build_plan_from_stage_config(
        self,
        message: str,
        workflow: str,
        stage_configs: List[Dict],
        coordinator_prompt: str,
    ) -> SwarmPlan:
        stages: List[Stage] = []
        for index, stage_config in enumerate(stage_configs, start=1):
            agents: List[AgentSpec] = []
            for role_id in stage_config["roles"]:
                role = self.role_bank.get(role_id)
                if not role:
                    continue
                agents.append(
                    AgentSpec(
                        agent_id=_agent_id(role_id),
                        role_id=role_id,
                        task_prompt=_build_task_prompt(role, message, index),
                        tools=list(role.allowed_tools if role.needs_tools else []),
                        system_prompt=role.system_prompt,
                    )
                )
            if agents:
                stages.append(
                    Stage(
                        stage_id=f"stage_{index}",
                        parallel=bool(stage_config.get("parallel", True)),
                        agents=agents,
                        depends_on=[stages[-1].stage_id] if stages else [],
                    )
                )
        return SwarmPlan(
            workflow=workflow,
            stages=stages,
            coordinator_prompt=coordinator_prompt,
            original_query=message,
        )
