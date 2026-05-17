"""Context compression for SophiaAgent.

Token-aware compression with 64K limit (configurable).
Auto-triggers when approaching the limit, preserving recent messages
and key research artifacts.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Smart compression of long conversations to prevent context overflow.

    Trigger: when estimated tokens exceed trigger_threshold (default 75% of max).
    Strategy: summarize older messages, keep recent N pairs intact.
    """

    DEFAULT_MAX_TOKENS = 64000
    DEFAULT_TRIGGER_RATIO = 0.65  # 41.6K for 64K limit; leaves ~22K for 146 tool schemas + response
    DEFAULT_KEEP_RECENT = 4       # keep 4 user-assistant pairs = 8 messages

    def __init__(
        self,
        hooks: HookManager = None,
        max_tokens: int = None,
        trigger_ratio: float = None,
        keep_recent: int = None,
    ):
        self.hooks = hooks
        self.max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS
        self.trigger_tokens = int(self.max_tokens * (trigger_ratio or self.DEFAULT_TRIGGER_RATIO))
        self.keep_recent = keep_recent or self.DEFAULT_KEEP_RECENT

    # ------------------------------------------------------------------
    # Hook compatibility
    # ------------------------------------------------------------------

    def on_pre_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook handler: compress messages before LLM call."""
        messages = context.get("messages")
        if messages and self.should_compress(messages):
            compressed = self.compress(messages)
            context["messages"] = compressed
        return context

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total tokens in messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total += self._count_content_tokens(content)
            # Tool calls / results also have overhead
            if msg.get("tool_calls"):
                total += 200  # rough overhead per tool call block
        return total

    @staticmethod
    def _count_content_tokens(text: str) -> int:
        """Token count with tiktoken if available, else char heuristic."""
        if not text:
            return 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Heuristic: CJK ~0.5 tokens/char, ASCII ~0.25 tokens/char
            cjk = sum(1 for c in text if "一" <= c <= "鿿")
            return int(cjk * 0.5 + (len(text) - cjk) * 0.25)

    # ------------------------------------------------------------------
    # Compression entry point
    # ------------------------------------------------------------------

    def should_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return self.estimate_tokens(messages) > self.trigger_tokens

    def maybe_compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress if over threshold, otherwise return as-is."""
        token_count = self.estimate_tokens(messages)
        if token_count <= self.trigger_tokens:
            return messages

        logger.info(
            "Context compression triggered: %d tokens > %d trigger threshold",
            token_count,
            self.trigger_tokens,
        )
        compressed = self.compress(messages)
        new_count = self.estimate_tokens(compressed)
        logger.info(
            "Compressed %d messages (%d tokens) -> %d messages (%d tokens)",
            len(messages),
            token_count,
            len(compressed),
            new_count,
        )
        return compressed

    def compress(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compress by summarizing older messages."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        keep_count = self.keep_recent * 2  # user + assistant pairs
        if len(non_system) <= keep_count:
            return messages

        to_summarize = non_system[:-keep_count]
        recent = non_system[-keep_count:]

        summary_text = self._summarize(to_summarize)

        result = list(system_msgs)
        result.append({
            "role": "system",
            "content": f"[Earlier Conversation Summary]\n{summary_text}",
        })
        result.extend(recent)

        if self.hooks:
            self.hooks.emit(
                HookEvent.CONTEXT_COMPRESS,
                {
                    "original_count": len(messages),
                    "compressed_count": len(result),
                    "summarized_count": len(to_summarize),
                    "original_tokens": self.estimate_tokens(messages),
                    "compressed_tokens": self.estimate_tokens(result),
                },
            )

        return result

    # ------------------------------------------------------------------
    # Smart summarization
    # ------------------------------------------------------------------

    def _summarize(self, messages: List[Dict[str, Any]]) -> str:
        """Extract key research artifacts from old messages."""
        parts: List[str] = []

        research_question = ""
        methodology = ""
        findings: List[str] = []
        data_refs: List[str] = []
        goals: List[str] = []
        completed_steps: List[str] = []
        pending_steps: List[str] = []
        artifacts: List[str] = []
        tool_results: List[str] = []
        failures: List[str] = []
        quality_gates: List[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue

            if role == "user":
                if any(kw in content.lower() for kw in ("research", "analysis", "paper", "empirical")):
                    if len(content) > len(research_question):
                        research_question = content[:400]
                if any(kw in content for kw in ("研究", "分析", "问题", "假设")):
                    if len(content) > len(research_question):
                        research_question = content[:400]

            elif role == "assistant":
                # Extract substantive conclusions
                for line in content.split("\n"):
                    line = line.strip()
                    if len(line) < 20:
                        continue
                    if any(kw in line for kw in ("结论", "结果", "显著", "p=", "effect", "finding")):
                        if line not in findings:
                            findings.append(line[:350])
                    if "methodology_advise" in line:
                        # Try to extract recommended method
                        pass

            elif role == "tool":
                try:
                    data = json.loads(content)
                    if data.get("result_id"):
                        data_refs.append(data["result_id"])
                        tool_results.append(f"result_id={data['result_id']}")
                    if data.get("apa"):
                        findings.append(data["apa"][:400])
                    if data.get("method_id"):
                        methodology = data["method_id"]
                    if data.get("recommended_methods"):
                        methods = data["recommended_methods"]
                        if methods:
                            methodology = methods[0].get("method_id", methodology)
                    if data.get("artifacts"):
                        artifacts.extend(str(item)[:350] for item in data["artifacts"][:10])
                    if data.get("warnings"):
                        failures.extend(str(item)[:350] for item in data["warnings"][:10])
                    if data.get("stage_outputs"):
                        tool_results.append("stage_outputs captured")
                    if data.get("executed") is not None:
                        tool_results.append(f"executed={data.get('executed')}")
                except (json.JSONDecodeError, TypeError):
                    # Non-JSON tool result
                    if len(content) > 30 and len(content) < 500:
                        findings.append(content[:300])

        parts.append(
            "Continuity handoff packet: preserve objective, completed work, pending work, "
            "tool evidence, artifacts, failures, quality gates, and next action."
        )
        if research_question:
            parts.append(f"User's research question: {research_question}")
        if methodology:
            parts.append(f"Methodology selected: {methodology}")
        if data_refs:
            parts.append(f"Data / result IDs: {', '.join(data_refs[:10])}")
        if completed_steps:
            parts.append("Completed steps:\n" + "\n".join(f"- {x}" for x in completed_steps[-8:]))
        if pending_steps:
            parts.append("Pending or blocked steps:\n" + "\n".join(f"- {x}" for x in pending_steps[-8:]))
        if artifacts:
            parts.append("Artifacts and exports:\n" + "\n".join(f"- {x}" for x in artifacts[-10:]))
        if tool_results:
            parts.append("Tool execution facts:\n" + "\n".join(f"- {x}" for x in tool_results[-10:]))
        if failures:
            parts.append("Warnings / blockers:\n" + "\n".join(f"- {x}" for x in failures[-10:]))
        if quality_gates:
            parts.append("Quality and credibility gates:\n" + "\n".join(f"- {x}" for x in quality_gates[-8:]))
        if findings:
            unique_findings = []
            seen = set()
            for f in findings[-8:]:
                key = f[:60]
                if key not in seen:
                    seen.add(key)
                    unique_findings.append(f)
            parts.append("Key findings:\n" + "\n".join(f"- {f}" for f in unique_findings))
        if goals:
            parts.append(f"Goals: {', '.join(goals)}")

        return "\n\n".join(parts) if parts else "Earlier discussion summarized."
