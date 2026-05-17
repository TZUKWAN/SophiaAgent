"""Automatic task analysis for SophiaAgent swarm startup."""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from sophia.swarm.models import SwarmDecision

logger = logging.getLogger(__name__)

SIMPLE_PATTERNS = [
    r"^\s*(你好|您好|在吗|hi|hello|hey|thanks|谢谢|再见|bye)[。！!？?\s]*$",
    r"^\s*(现在几点|date|pwd|ls|dir)\s*$",
]

SINGLE_TOOL_KEYWORDS = [
    "读取文件",
    "列出文件",
    "查看目录",
    "描述统计",
    "加载数据",
    "打开文件",
]

ROLE_KEYWORDS = {
    "literature_searcher": ["文献", "综述", "研究现状", "引用", "参考文献", "检索"],
    "data_analyst": ["数据", "统计", "回归", "因果", "可视化", "模型"],
    "writer": ["写", "撰写", "论文", "报告", "专著", "大纲", "章节"],
    "reviewer": ["评审", "审查", "修改", "润色", "质量", "问题"],
    "methodologist": ["方法", "研究设计", "实验设计", "调查", "访谈", "方案"],
    "critic": ["批判", "逻辑", "漏洞", "矛盾", "风险"],
    "citation_manager": ["格式", "APA", "GB/T", "参考文献"],
}

SWARM_INTENSIFIERS = [
    "全面",
    "系统",
    "深入",
    "完整",
    "全流程",
    "多角度",
    "分别",
    "同时",
    "并行",
    "对比",
    "比较",
]

ANALYSIS_PROMPT = """Analyze whether this user request should be handled by multiple specialized agents.

Return only valid JSON:
{
  "need_swarm": true,
  "reason": "brief Chinese reason",
  "estimated_roles": 3,
  "recommended_roles": ["literature_searcher", "writer", "reviewer"],
  "workflow": "parallel"
}

Use a swarm only when the request has multiple domains, multiple stages, parallelizable work,
or quality review needs. Do not use a swarm for greetings, single file operations, or one simple tool call.

User request:
{message}
"""


def _unique_roles(roles: list[str]) -> list[str]:
    seen = set()
    result = []
    for role in roles:
        if role not in seen:
            seen.add(role)
            result.append(role)
    return result


def _rule_analyze(message: str) -> Optional[SwarmDecision]:
    text = message.strip()
    if not text:
        return SwarmDecision(False, "空消息不启动蜂群", confidence=1.0)

    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return SwarmDecision(False, "匹配简单问候或单句命令", confidence=0.95)

    if any(keyword in text for keyword in SINGLE_TOOL_KEYWORDS) and len(text) < 50:
        return SwarmDecision(False, "匹配单次工具调用任务", confidence=0.85)

    matched_roles = []
    for role_id, keywords in ROLE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            matched_roles.append(role_id)

    if len(text) < 12 and not matched_roles:
        return SwarmDecision(False, "消息很短，按简单任务处理", confidence=0.85)

    intensifier_count = sum(1 for keyword in SWARM_INTENSIFIERS if keyword in text)
    explicit_multi_stage = any(token in text for token in ["先", "然后", "最后", "全流程", "从", "到"])

    if len(matched_roles) >= 3:
        workflow = "mixed" if explicit_multi_stage else "parallel"
        return SwarmDecision(
            True,
            f"检测到多个专业领域：{', '.join(matched_roles[:5])}",
            estimated_roles=min(len(matched_roles), 5),
            confidence=0.85,
            recommended_roles=_unique_roles(matched_roles)[:5],
            workflow=workflow,
        )

    if len(matched_roles) >= 2 and (intensifier_count >= 1 or explicit_multi_stage):
        return SwarmDecision(
            True,
            "任务涉及多角色协作并具有复杂度强化词或阶段依赖",
            estimated_roles=min(len(matched_roles) + 1, 5),
            confidence=0.8,
            recommended_roles=_unique_roles(matched_roles + ["reviewer"])[:5],
            workflow="mixed" if explicit_multi_stage else "parallel",
        )

    if intensifier_count >= 2 and len(text) > 40:
        roles = _unique_roles(matched_roles + ["writer", "critic"])
        return SwarmDecision(
            True,
            "任务要求全面/系统/多角度处理，适合自动拆分",
            estimated_roles=min(max(len(roles), 2), 5),
            confidence=0.72,
            recommended_roles=roles[:5],
            workflow="parallel",
        )

    if len(matched_roles) == 1:
        return None

    return SwarmDecision(False, "未检测到需要多智能体协作的特征", confidence=0.7)


class TaskAnalyzer:
    """Two-layer task analyzer: deterministic rules first, optional LLM second."""

    def __init__(self, llm_call: Optional[Callable[[str], str]] = None, use_llm: bool = True):
        self.llm_call = llm_call
        self.use_llm = use_llm

    def analyze(self, message: str) -> SwarmDecision:
        rule_decision = _rule_analyze(message)
        if rule_decision is not None:
            logger.info("Swarm rule decision: %s", rule_decision.to_dict())
            return rule_decision

        if self.use_llm and self.llm_call:
            decision = self._llm_analyze(message)
            logger.info("Swarm LLM decision: %s", decision.to_dict())
            return decision

        return SwarmDecision(False, "规则层不确定且没有可用 LLM 分析器，保守不启动蜂群", confidence=0.5)

    def quick_check(self, message: str) -> bool:
        decision = _rule_analyze(message)
        if decision is not None:
            return decision.need_swarm
        return False

    def _llm_analyze(self, message: str) -> SwarmDecision:
        try:
            raw = self.llm_call(ANALYSIS_PROMPT.replace("{message}", message))
            match = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if not match:
                raise ValueError("LLM did not return JSON")
            data = json.loads(match.group())
            roles = data.get("recommended_roles") or []
            return SwarmDecision(
                need_swarm=bool(data.get("need_swarm")),
                reason=str(data.get("reason") or "LLM 判断结果"),
                estimated_roles=max(0, min(int(data.get("estimated_roles") or len(roles)), 5)),
                confidence=0.8,
                recommended_roles=_unique_roles([str(role) for role in roles])[:5],
                workflow=str(data.get("workflow") or "parallel"),
            )
        except Exception as exc:
            logger.warning("LLM swarm analysis failed: %s", exc)
            return SwarmDecision(False, f"LLM 分析失败：{exc}", confidence=0.3)
