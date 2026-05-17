"""Synthesize multiple swarm agent outputs into one user-facing response."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from sophia.swarm.models import AgentResult, SwarmPlan

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """你是 SophiaAgent 的蜂群协调汇总器。

原始用户需求：
{query}

协调要求：
{coordinator_prompt}

各专家输出：
{agent_outputs}

请生成一份统一最终回复。要求：
1. 只采纳专家输出中真实出现的内容，不编造数据、文献或完成状态。
2. 对事实、数据、文献、方法论分别按对应专家优先级处理。
3. 如果专家之间存在冲突或信息不足，明确标注。
4. 用中文回答，除非用户明确要求其他语言。
"""


class ResultSynthesizer:
    """Merge, de-duplicate, and resolve swarm outputs."""

    def __init__(self, llm_call: Optional[Callable[[str], str]] = None):
        self.llm_call = llm_call

    def synthesize(self, results: List[AgentResult], plan: SwarmPlan, use_llm: bool = True) -> str:
        completed = [result for result in results if result.status == "completed" and result.content]
        failed = [result for result in results if result.status in {"failed", "timeout"}]

        if not completed:
            return self._format_failure(failed)

        if len(completed) == 1 and not failed:
            return completed[0].content

        if use_llm and self.llm_call:
            try:
                return self._llm_synthesize(completed, failed, plan)
            except Exception as exc:
                logger.warning("Swarm synthesis LLM failed: %s", exc)

        return self._structured_concat(completed, failed, plan)

    def _llm_synthesize(
        self,
        completed: List[AgentResult],
        failed: List[AgentResult],
        plan: SwarmPlan,
    ) -> str:
        outputs = []
        for result in completed:
            outputs.append(f"[{result.role_id} / {result.agent_id}]\n{result.content[:4000]}")
        if failed:
            outputs.append(
                "[执行失败的专家]\n"
                + "\n".join(f"- {r.role_id}/{r.agent_id}: {r.error or r.status}" for r in failed)
            )
        prompt = SYNTHESIS_PROMPT.format(
            query=plan.original_query,
            coordinator_prompt=plan.coordinator_prompt,
            agent_outputs="\n\n---\n\n".join(outputs),
        )
        return self.llm_call(prompt)

    def _structured_concat(
        self,
        completed: List[AgentResult],
        failed: List[AgentResult],
        plan: SwarmPlan,
    ) -> str:
        lines = ["基于蜂群中各专家的输出，综合结果如下：", ""]
        for result in completed:
            lines.extend([f"## {result.role_id}", result.content.strip(), ""])
        if failed:
            lines.append("## 未完成或失败的子任务")
            for result in failed:
                lines.append(f"- {result.role_id}/{result.agent_id}: {result.error or result.status}")
            lines.append("")
        if plan.coordinator_prompt:
            lines.extend(["## 汇总说明", plan.coordinator_prompt])
        return "\n".join(lines).strip()

    def _format_failure(self, failed: List[AgentResult]) -> str:
        if not failed:
            return "蜂群执行没有返回有效结果。"
        lines = ["蜂群执行失败，未生成可用结果："]
        for result in failed:
            lines.append(f"- {result.role_id}/{result.agent_id}: {result.error or result.status}")
        return "\n".join(lines)

    def detect_conflicts(self, results: List[AgentResult]) -> List[dict]:
        """Small heuristic conflict detector for tests and diagnostics."""
        import re

        conflicts = []
        completed = [result for result in results if result.status == "completed"]
        for index, left in enumerate(completed):
            left_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", left.content))
            for right in completed[index + 1 :]:
                right_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", right.content))
                shared = left_numbers & right_numbers
                if len(shared) >= 3:
                    conflicts.append(
                        {
                            "between": [left.role_id, right.role_id],
                            "shared_numbers": sorted(shared)[:5],
                            "severity": "info",
                        }
                    )
        return conflicts
