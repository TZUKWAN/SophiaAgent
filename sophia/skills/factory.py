"""SkillFactory: Convert execution patterns and manual workflows into installable skills.

A *skill* is a reusable, versioned workflow that the agent can trigger
automatically (by intent matching) or on explicit user request.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from sophia.skills.pattern_miner import ExecutionPatternMiner, WorkflowPattern

logger = logging.getLogger(__name__)


class SkillFactory:
    """Generate, install, and execute skills from workflow patterns."""

    def __init__(self, skill_manager, learning_manager=None, miner: Optional[ExecutionPatternMiner] = None, catalog=None):
        self.skill_manager = skill_manager
        self.learning_manager = learning_manager
        self.miner = miner or ExecutionPatternMiner()
        self.catalog = catalog  # Optional MethodCatalog for discovery linkage

    # ------------------------------------------------------------------
    # Auto-generation from logs
    # ------------------------------------------------------------------

    def auto_generate_from_logs(self, top_n: int = 5) -> List[Dict]:
        """Scan execution logs and automatically generate skills from top patterns.

        Returns:
            List of installed skill definitions.
        """
        if self.learning_manager is None:
            logger.warning("No learning_manager attached; cannot auto-generate skills.")
            return []

        log = self.learning_manager.get_structured_log()
        patterns = self.miner.mine(log)
        installed = []
        for pattern in patterns[:top_n]:
            skill_def = self._pattern_to_skill_def(pattern)
            sid = self.skill_manager.install(skill_def)
            installed.append({"skill_id": sid, "name": skill_def["name"], "auto": True})
            logger.info("Auto-generated skill '%s' (id=%s)", skill_def["name"], sid)
            if self.catalog is not None:
                try:
                    self.catalog.register_skill(sid, skill_def["name"], skill_def.get("workflow", []))
                except Exception as e:
                    logger.warning("Failed to register skill '%s' in catalog: %s", sid, e)
        return installed

    # ------------------------------------------------------------------
    # Explicit save from a manually provided workflow
    # ------------------------------------------------------------------

    def create_skill(
        self,
        name: str,
        workflow: List[Dict[str, Any]],
        trigger: Optional[Dict[str, Any]] = None,
        description: str = "",
        category: str = "general",
        author: str = "user",
    ) -> str:
        """Explicitly create a skill from a user-provided workflow.

        Args:
            name: human-readable skill name.
            workflow: list of step dicts, each with at least {"tool": "...", "params": {...}}.
            trigger: optional trigger conditions (intent, keywords, category, data_type).
            description: long-form description.
            category: skill category.
            author: who created it.

        Returns:
            The installed skill_id.
        """
        skill_def = self._build_skill_def(
            name=name,
            workflow=workflow,
            trigger=trigger or {},
            description=description,
            category=category,
            author=author,
            auto_generated=False,
        )
        sid = self.skill_manager.install(skill_def)
        logger.info("User-created skill '%s' (id=%s)", name, sid)
        if self.catalog is not None:
            try:
                self.catalog.register_skill(sid, name, workflow)
            except Exception as e:
                logger.warning("Failed to register skill '%s' in catalog: %s", sid, e)
        return sid

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_skill(
        self,
        skill_id: str,
        initial_args: Optional[Dict[str, Any]] = None,
        *,
        registry=None,
        result_store=None,
    ) -> Dict[str, Any]:
        """Execute a skill's workflow step by step.

        Args:
            skill_id: skill to run.
            initial_args: arguments passed to the first step (and available to all).
            registry: ToolRegistry for dispatching tools.
            result_store: ResultStore for lineage tracking.

        Returns:
            Dict with step results, final output, and metadata.
        """
        skill = self.skill_manager.get_skill(skill_id)
        if not skill:
            return {"success": False, "error": f"Skill not found: {skill_id}"}

        workflow = skill.get("workflow", [])
        if not workflow:
            return {"success": False, "error": "Skill has no workflow steps"}

        step_results = []
        context = dict(initial_args or {})
        start_time = time.time()

        for idx, step in enumerate(workflow):
            tool_name = step.get("tool")
            params = dict(step.get("params", {}))

            # Merge context variables (e.g. previous result_ids)
            merged_params = self._merge_params(params, context)

            if registry is None:
                step_results.append({
                    "step": idx,
                    "tool": tool_name,
                    "status": "skipped",
                    "reason": "No registry available",
                })
                continue

            try:
                result_raw = registry.dispatch(tool_name, merged_params)
                result = json.loads(result_raw) if isinstance(result_raw, str) else result_raw

                # Detect error responses from registry.dispatch
                if isinstance(result, dict) and result.get("error"):
                    step_results.append({
                        "step": idx,
                        "tool": tool_name,
                        "status": "error",
                        "error": result["error"],
                    })
                    return self._finalize_execution(
                        skill_id, skill, step_results, idx, False,
                        f"Step {idx} ({tool_name}) failed: {result['error']}",
                        context, start_time, len(workflow),
                    )

                step_results.append({
                    "step": idx,
                    "tool": tool_name,
                    "status": "success",
                    "result_preview": str(result)[:500],
                })
                # Update context with result_id if present
                if isinstance(result, dict) and result.get("result_id"):
                    context[f"{tool_name}_result_id"] = result["result_id"]
                # Also expose the raw result under a slot name
                slot = step.get("result_slot", f"step_{idx}_result")
                context[slot] = result
            except Exception as exc:
                step_results.append({
                    "step": idx,
                    "tool": tool_name,
                    "status": "error",
                    "error": str(exc),
                })
                return self._finalize_execution(
                    skill_id, skill, step_results, idx, False,
                    f"Step {idx} ({tool_name}) failed: {exc}",
                    context, start_time, len(workflow),
                )

        return self._finalize_execution(
            skill_id, skill, step_results, len(workflow), True, "",
            context, start_time, len(workflow),
        )

    def _finalize_execution(
        self,
        skill_id: str,
        skill: Dict[str, Any],
        step_results: List[Dict],
        completed_steps: int,
        success: bool,
        error: str,
        context: Dict[str, Any],
        start_time: float,
        total_steps: int,
    ) -> Dict[str, Any]:
        """Finalize execution result, record history, and return."""
        elapsed = time.time() - start_time
        result = {
            "success": success,
            "skill_id": skill_id,
            "skill_name": skill.get("name"),
            "step_results": step_results,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
            "context": {k: v for k, v in context.items() if not k.endswith("_result")},
        }
        # Record execution history for evolution
        try:
            self.skill_manager.record_execution(skill_id, result)
        except Exception as e:
            logger.warning("Failed to record execution history for %s: %s", skill_id, e)
        return result

    # ------------------------------------------------------------------
    # Evolution: auto-tune and version iteration
    # ------------------------------------------------------------------

    def auto_evolve(self, skill_id: str, min_executions: int = 5,
                    success_rate_threshold: float = 0.7) -> Dict[str, Any]:
        """Analyze execution history and evolve the skill if needed.

        Evolution rules:
        1. If success_rate < threshold and enough executions: try to simplify
           workflow by removing steps that consistently fail.
        2. If success_rate >= threshold: bump version and mark as stable.
        3. Update trigger keywords based on successful execution context.

        Returns:
            Dict describing what changed.
        """
        stats = self.skill_manager.get_skill_stats(skill_id)
        if not stats:
            return {"success": False, "error": f"Skill not found: {skill_id}"}

        history = self.skill_manager.get_execution_history(skill_id, limit=50)
        if len(history) < min_executions:
            return {
                "success": True,
                "action": "noop",
                "reason": f"Only {len(history)} executions, need {min_executions}",
            }

        success_rate = stats["success_rate"]
        skill = self.skill_manager.get_skill(skill_id)
        workflow = skill.get("workflow", [])
        trigger = skill.get("trigger", {})

        changes = []

        # Rule 1: Low success rate -> identify and remove failing steps
        if success_rate < success_rate_threshold:
            failing_steps = self._identify_failing_steps(history, len(workflow))
            if failing_steps:
                new_workflow = [s for i, s in enumerate(workflow) if i not in failing_steps]
                if new_workflow and len(new_workflow) < len(workflow):
                    self.skill_manager.evolve_skill(skill_id, new_workflow=new_workflow)
                    changes.append(f"Removed failing steps: {sorted(failing_steps)}")
                    workflow = new_workflow

        # Rule 2: High success rate -> bump version
        if success_rate >= success_rate_threshold and stats["execution_count"] >= min_executions:
            # Only bump if current version hasn't been bumped for this milestone
            current_version = skill.get("version", "1.0")
            self.skill_manager.evolve_skill(skill_id, new_workflow=workflow, new_trigger=trigger)
            new_skill = self.skill_manager.get_skill(skill_id)
            if new_skill and new_skill.get("version") != current_version:
                changes.append(f"Bumped version {current_version} -> {new_skill['version']}")

        # Rule 3: Enrich trigger keywords from successful execution context
        successful_contexts = [
            h["context"] for h in history
            if h.get("success") and isinstance(h.get("context"), dict)
        ]
        if successful_contexts:
            new_keywords = self._extract_keywords_from_contexts(successful_contexts)
            old_keywords = set(trigger.get("keywords", []))
            merged_keywords = sorted(old_keywords | new_keywords)
            if merged_keywords != sorted(old_keywords):
                trigger["keywords"] = merged_keywords
                self.skill_manager.evolve_skill(skill_id, new_workflow=workflow, new_trigger=trigger)
                changes.append(f"Updated trigger keywords: {merged_keywords}")

        return {
            "success": True,
            "skill_id": skill_id,
            "action": "evolved" if changes else "noop",
            "changes": changes,
            "stats": stats,
        }

    @staticmethod
    def _identify_failing_steps(history: List[Dict], total_steps: int) -> set:
        """Identify step indices that consistently fail across executions."""
        step_failures = {i: 0 for i in range(total_steps)}
        step_attempts = {i: 0 for i in range(total_steps)}

        for h in history:
            for sr in h.get("step_results", []):
                step_idx = sr.get("step", 0)
                if step_idx in step_attempts:
                    step_attempts[step_idx] += 1
                    if sr.get("status") == "error":
                        step_failures[step_idx] += 1

        # Steps that fail > 80% of the time and have at least 3 attempts
        failing = set()
        for i in range(total_steps):
            if step_attempts[i] >= 3 and step_failures[i] / step_attempts[i] > 0.8:
                failing.add(i)
        return failing

    @staticmethod
    def _extract_keywords_from_contexts(contexts: List[Dict]) -> set:
        """Extract potential trigger keywords from successful execution contexts."""
        keywords = set()
        for ctx in contexts:
            for key in ctx.keys():
                # Use top-level keys as potential keywords
                if len(key) > 2 and not key.endswith("_result"):
                    keywords.add(key.lower().replace("_", " "))
        return keywords

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pattern_to_skill_def(self, pattern: WorkflowPattern) -> Dict[str, Any]:
        """Convert a mined WorkflowPattern into a skill definition."""
        name = "Auto: " + " → ".join(pattern.sequence)
        workflow = []
        for idx, tool in enumerate(pattern.sequence):
            step_params = pattern.params_template.get(f"step_{idx}", {})
            workflow.append({
                "tool": tool,
                "params": step_params,
                "result_slot": f"step_{idx}_result",
            })

        trigger = {
            "keywords": pattern.trigger_keywords[:10],
            "category": self._infer_category(pattern.sequence),
        }

        return self._build_skill_def(
            name=name,
            workflow=workflow,
            trigger=trigger,
            description=f"Auto-generated skill from {pattern.frequency} successful executions (success_rate={pattern.success_rate:.2f})",
            category="auto_mined",
            author="sophia_learning",
            auto_generated=True,
            success_rate=pattern.success_rate,
            execution_count=pattern.total_sessions,
            avg_score=pattern.avg_score,
        )

    @staticmethod
    def _build_skill_def(
        *,
        name: str,
        workflow: List[Dict[str, Any]],
        trigger: Dict[str, Any],
        description: str,
        category: str,
        author: str,
        auto_generated: bool,
        success_rate: float = 0.0,
        execution_count: int = 0,
        avg_score: float = 0.0,
    ) -> Dict[str, Any]:
        skill_id = name.lower().replace(" ", "_").replace(":", "").replace("→", "to")[:50]
        # Ensure uniqueness by appending short hash if needed
        skill_id = f"{skill_id}_{str(uuid.uuid4())[:6]}"

        # Build tool_schemas for SkillManager compatibility
        tool_schemas = []
        handler_code_parts = ["import json\n\ndef handle(args):\n"]
        handler_code_parts.append(f'    """{description}"""\n')
        handler_code_parts.append("    results = []\n")

        for idx, step in enumerate(workflow):
            tool = step["tool"]
            params = step.get("params", {})
            tool_schemas.append({
                "name": tool,
                "description": f"Step {idx + 1}: {tool}",
                "parameters": {"type": "object", "properties": {}},
            })
            # Build a simple sequential handler that delegates to each tool
            handler_code_parts.append(f"    # Step {idx + 1}: {tool}\n")
            handler_code_parts.append(f"    step{idx}_params = {json.dumps(params, ensure_ascii=False)}\n")
            handler_code_parts.append(f"    results.append({{'step': {idx}, 'tool': '{tool}', 'params': step{idx}_params}})\n")

        handler_code_parts.append("    return json.dumps({'skill_workflow': results, 'status': 'success'}, ensure_ascii=False)\n")

        return {
            "id": skill_id,
            "name": name,
            "version": "1.0",
            "description": description,
            "author": author,
            "category": category,
            "workflow": workflow,
            "trigger": trigger,
            "success_rate": success_rate,
            "execution_count": execution_count,
            "avg_score": avg_score,
            "auto_generated": auto_generated,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # Legacy compatibility with SkillManager schema
            "tool_schemas": tool_schemas,
            "handler_code": "".join(handler_code_parts),
        }

    @staticmethod
    def _merge_params(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Replace placeholder values in params with context variables."""
        merged = dict(params)
        for key, val in merged.items():
            if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                var_name = val[2:-1]
                if var_name in context:
                    merged[key] = context[var_name]
        return merged

    @staticmethod
    def _infer_category(sequence: List[str]) -> str:
        """Infer skill category from the tools in the workflow."""
        category_map = {
            "research_did": "causal",
            "research_psm": "causal",
            "research_iv": "causal",
            "research_rdd": "causal",
            "research_its": "causal",
            "research_synthetic_control": "causal",
            "research_ttest": "statistics",
            "research_anova": "statistics",
            "research_regression": "statistics",
            "research_correlation": "statistics",
            "research_ml_train": "ml",
            "research_ml_evaluate": "ml",
            "research_thematic": "qualitative",
            "research_content": "qualitative",
            "research_meta_fixed": "meta",
            "research_meta_random": "meta",
        }
        for tool in sequence:
            if tool in category_map:
                return category_map[tool]
        return "general"
