"""Autopilot: autonomous orchestration layer for SophiaAgent.

Three layers:
1. Intent Router    -- rule-based, zero-latency prompt augmentation
2. Execution Monitor -- hook-driven pattern detection
3. System Prompt    -- LLM-operating manual embedded in system prompt
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ── Layer 1: Intent Router ───────────────────────────────────────

class AutopilotRouter:
    """Detect user intent and augment messages with system hints."""

    RESEARCH_KEYWORDS = [
        # Chinese
        "研究", "分析", "比较", "影响", "效应", "政策", "调查", "问卷", "访谈",
        "因果", "回归", "显著", "相关", "差异", "检验", "实证", "评估", "推断",
        "计量", "统计", "样本", "变量", "控制", "处理", "干预", "实验",
        # English
        "research", "analyze", "analysis", "compare", "impact", "effect",
        "policy", "survey", "questionnaire", "interview", "causal",
        "regression", "significant", "correlation", "difference", "test",
        "empirical", "evaluate", "inference", "econometric", "statistical",
        "sample", "variable", "control", "treatment", "intervention", "experiment",
        "did", "psm", "iv", "rdd", "synthetic control", "meta-analysis",
    ]

    REPETITIVE_KEYWORDS = [
        "每天", "每周", "定期", "批量", "自动化", "定时", "循环",
        "每次都要", "能不能记住", "固定流程", "模板", "复用",
        "daily", "weekly", "schedule", "batch", "automate", "loop",
        "every time", "template", "reuse", "remember",
    ]

    @classmethod
    def is_research_intent(cls, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in cls.RESEARCH_KEYWORDS)

    @classmethod
    def is_repetitive_intent(cls, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in cls.REPETITIVE_KEYWORDS)

    @classmethod
    def augment_messages(
        cls,
        messages: List[Dict[str, Any]],
        user_message: str,
    ) -> List[Dict[str, Any]]:
        """Inject system hints based on detected intent."""
        hints = []

        if cls.is_research_intent(user_message):
            hints.append(
                "[Autopilot] User is asking a research question. "
                "For empirical, quantitative, causal, or data-analysis work, first call "
                "empirical_workflow_plan; if real data and required variables are available, "
                "call empirical_workflow_run; then use methodology_advise and the recommended "
                "specialized research tools to synthesize real results. Run credibility checks: "
                "data contract, diagnostics, identification assumptions, robustness, sensitivity, "
                "skipped-check reasons, and artifact export. Do not fabricate any result. "
                "If the workflow repeats, consider creating a skill."
            )

        if cls.is_repetitive_intent(user_message):
            hints.append(
                "[Autopilot] User mentions repetitive/scheduled work. "
                "Consider using loop_create or skill_create to automate."
            )

        if hints:
            messages = list(messages)
            # Merge hints into the last system message to preserve message ordering
            # (some models require system messages at the beginning).
            merged = False
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "system":
                    messages[i] = {
                        "role": "system",
                        "content": messages[i].get("content", "") + "\n\n" + " ".join(hints),
                    }
                    merged = True
                    break
            if not merged:
                messages.insert(0, {"role": "system", "content": " ".join(hints)})

        return messages


# ── Layer 2: Execution Monitor ───────────────────────────────────

class ExecutionMonitor:
    """Watch execution patterns and trigger auto-actions via hooks."""

    def __init__(
        self,
        skill_factory=None,
        learning_manager=None,
        goal_manager=None,
    ):
        self.skill_factory = skill_factory
        self.learning_manager = learning_manager
        self.goal_manager = goal_manager
        self._tool_sequence: List[Dict[str, Any]] = []
        self._sequence_window = 20
        self._auto_discover_cooldown = 0  # timestamp

    def on_tool_post_dispatch(self, context: Dict[str, Any]):
        """Hook handler for tool.post_dispatch."""
        tool_name = context.get("tool", "")
        if not tool_name:
            return

        self._tool_sequence.append({
            "tool": tool_name,
            "timestamp": context.get("timestamp", 0),
            "success": not context.get("error"),
        })

        if len(self._tool_sequence) > self._sequence_window:
            self._tool_sequence = self._tool_sequence[-self._sequence_window:]

        self._maybe_auto_discover_skill()

    def _maybe_auto_discover_skill(self):
        """If a 3+ step sequence repeats 3+ times, auto-discover skill."""
        import time

        # Cooldown: avoid spamming auto-discover
        if time.time() - self._auto_discover_cooldown < 300:
            return

        if len(self._tool_sequence) < 6:
            return

        seq = [t["tool"] for t in self._tool_sequence]
        for length in range(3, min(8, len(seq) // 2 + 1)):
            recent = tuple(seq[-length:])
            count = sum(
                1
                for i in range(len(seq) - length + 1)
                if tuple(seq[i : i + length]) == recent
            )
            if count >= 3:
                logger.info(
                    "Autopilot: detected repeated workflow %s (%d times), "
                    "auto-discovering skill",
                    recent,
                    count,
                )
                self._auto_discover_cooldown = time.time()
                try:
                    if self.skill_factory and self.skill_factory.learning_manager:
                        result = self.skill_factory.auto_generate_from_logs(top_n=1)
                        if result:
                            logger.info(
                                "Autopilot: auto-discovered skill %s",
                                result[0].get("skill_id"),
                            )
                except Exception as e:
                    logger.warning("Autopilot: auto skill discovery failed: %s", e)
                break

    def on_goal_completed(self, context: Dict[str, Any]):
        """Hook handler for goal.completed."""
        if self.learning_manager:
            try:
                self.learning_manager.on_goal_completed(context)
            except Exception:
                pass


# ── Layer 3: System Prompt Augmentation ──────────────────────────

AUTOPILOT_SYSTEM_APPENDIX = """

## Autopilot Operating Manual

You have access to a full research tool suite. When the user asks a research question, follow this default workflow automatically:

1. **Advise** -- Call `methodology_advise` with the user's research question and data description to get ranked method recommendations.
2. **Execute** -- Based on the recommendation, call the appropriate research tools in sequence (e.g., `research_load_data` → `research_did` → `research_plot` → `research_export_report`).
3. **Verify** -- Run credibility checks before interpretation: data contract, missingness, outliers, sample construction, assumptions, robustness, sensitivity, and skipped-check reasons.
4. **Synthesize** -- Provide an APA-style interpretation of the results. Include effect sizes, confidence intervals, practical significance, limitations, and artifact paths. Never invent missing empirical results.

### When to use internal mechanisms
- **Goal**: If the task has multiple steps or takes multiple turns, call `goal_create` first.
- **Loop**: If the user mentions daily/weekly/scheduled work, call `loop_create`.
- **Skill**: If you detect yourself doing the same 3+ tool sequence for this user, call `skill_create` to save it as a reusable template.
- **Skill Evolution**: If a skill exists but keeps failing, call `skill_evolve` to auto-tune it.
- **Methodology**: Always call `methodology_advise` before running analysis on new data.

### Context Compression
The system automatically compresses old conversation history when approaching the context limit. Recent messages and all tool results are preserved. You do not need to ask the user to start a new conversation.
"""


def build_autopilot_system_prompt(base_prompt: str) -> str:
    return base_prompt + AUTOPILOT_SYSTEM_APPENDIX


# ── Orchestrator ─────────────────────────────────────────────────

class AutopilotOrchestrator:
    """Top-level autopilot combining all layers."""

    def __init__(self, agent):
        self.agent = agent
        self.router = AutopilotRouter()
        self.monitor = ExecutionMonitor(
            skill_factory=getattr(agent, "skill_factory", None),
            learning_manager=getattr(agent, "learning", None),
            goal_manager=getattr(agent, "goals", None),
        )

    def before_run(
        self,
        user_message: str,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Augment messages before sending to LLM."""
        return self.router.augment_messages(messages, user_message)

    def register_hooks(self, hooks):
        """Register execution monitor hooks."""
        hooks.register(
            "tool.post_dispatch",
            self.monitor.on_tool_post_dispatch,
            priority=80,
            name="autopilot_monitor",
        )
        hooks.register(
            "goal.completed",
            self.monitor.on_goal_completed,
            priority=40,
            name="autopilot_goal",
        )
