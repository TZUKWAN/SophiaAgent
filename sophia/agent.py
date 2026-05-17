"""Core SophiaAgent conversation loop.

Integrates all mechanisms: Hook, Goal, SubAgent, Loop, Memory, Context,
Credentials, Recovery, Guardrails, Scheduler, Kanban, Plugins, Security,
Skills, Learning, Browser, Snapshot, Trajectory.
"""

import logging
from typing import Any, Dict, List, Optional

from sophia.autopilot import AutopilotOrchestrator
from sophia.config import Config
from sophia.context import ContextCompressor
from sophia.credentials import CredentialPool
from sophia.document_delivery import requested_output_format, save_generated_docx
from sophia.experiment import ExperimentManager, register_experiment_tools
from sophia.goal import GoalManager, register_goal_tools
from sophia.guardrails import ToolGuardrails
from sophia.hooks import HookEvent, HookManager
from sophia.kanban import KanbanBoard, register_kanban_tools
from sophia.learning import LearningManager
from sophia.loop import LoopManager, register_loop_tools
from sophia.memory import MemoryManager, register_memory_tools
from sophia.paper_quality import (
    append_quality_report_if_needed,
    build_paper_generation_contract,
    build_reference_priority_notice,
    inspect_generated_paper,
    is_paper_generation_request,
)
from sophia.paper_requirements import check_paper_requirements
from sophia.plugins import PluginManager
from sophia.prompts.system import build_system_prompt
from sophia.providers import create_provider
from sophia.providers.base import BaseProvider
from sophia.recovery import RecoveryManager
from sophia.research.advisor import MethodologyAdvisor
from sophia.research.causal import CausalEngine
from sophia.research.computational import ComputationalEngine
from sophia.research.design import ResearchDesignEngine
from sophia.research.discovery.dependency_manager import DependencyManager
from sophia.research.discovery.method_builder import MethodBuilder

# Self-evolving discovery system
from sophia.research.discovery.method_catalog import MethodCatalog
from sophia.research.discovery.method_searcher import MethodSearcher
from sophia.research.discovery.register import register_discovery_tools
from sophia.research.empirical_workflow import EmpiricalWorkflowEngine
from sophia.research.latex_exporter import LaTeXReporter
from sophia.research.llm import LLMEngine
from sophia.research.meta_analysis import MetaAnalysisEngine
from sophia.research.ml import MLEngine
from sophia.research.pipeline import ExperimentPipeline
from sophia.research.qualitative import QualitativeEngine
from sophia.research.register import register_method_tools
from sophia.research.result_store import ResultStore
from sophia.research.seed import GlobalSeed

# Research method engines
from sophia.research.statistics import StatisticalEngine
from sophia.research.survey import SurveyEngine
from sophia.research.visualization import VisualizationEngine
from sophia.research.workspace_guard import WorkspaceGuard
from sophia.scheduler import CronScheduler
from sophia.security import SecurityManager
from sophia.skills import SkillManager
from sophia.skills.factory import SkillFactory
from sophia.snapshot import SnapshotManager
from sophia.subagent import SubAgentManager, register_subagent_tools
from sophia.swarm import FilteredToolRegistry, SwarmOrchestrator
from sophia.task_harness import build_task_harness_prompt, is_empirical_request
from sophia.tools.analysis import register_analysis_tools
from sophia.tools.citation import register_citation_tools
from sophia.tools.data_collection import register_data_collection_tools
from sophia.tools.files import register_file_tools
from sophia.tools.registry import ToolRegistry
from sophia.tools.research import register_research_tools
from sophia.tools.review import register_review_tools
from sophia.tools.swarm import register_swarm_tools
from sophia.tools.web import register_web_tools
from sophia.tools.writing import register_writing_tools
from sophia.trajectory import TrajectoryRecorder
from sophia.workspace_context import (
    asks_for_paper_document,
    collect_workspace_context,
    iter_workspace_context_events,
    save_generated_markdown,
)

logger = logging.getLogger(__name__)


class SophiaAgent:
    """Core conversation agent for humanities and social science research."""

    def __init__(self, config: Config):
        self.config = config
        self.provider: BaseProvider = create_provider(config)
        self.tools = ToolRegistry()
        self.max_turns = config.model.max_turns
        self.workspace = config.session.workspace

        # Initialize all mechanisms
        self.hooks = HookManager()
        self.tools.set_hooks(self.hooks)

        # P0: Core
        self.goals = GoalManager(config.session.db_path, self.hooks)
        self.swarm_orchestrator = SwarmOrchestrator(
            run_fn=self._run_internal,
            llm_call_fn=self._llm_text,
            hooks=self.hooks,
            max_workers=getattr(config.loop, "max_concurrent", 3),
        )
        self.subagents = SubAgentManager(
            self._run_subagent, self.hooks, config.session.db_path,
            orchestrator=self.swarm_orchestrator,
        )
        self.loops = LoopManager(
            self._run_loop_tick, self.hooks, config.session.db_path,
        )

        # P1: Stability
        self.memory = MemoryManager(config.session.db_path, self.hooks)
        self.context_compressor = ContextCompressor(
            self.hooks,
            max_tokens=getattr(config.context, "max_tokens", None),
            trigger_ratio=getattr(config.context, "compress_threshold", None),
            keep_recent=getattr(config.context, "keep_recent", None),
        )
        self.autopilot = AutopilotOrchestrator(self)
        self.autopilot.register_hooks(self.hooks)
        self.credentials = CredentialPool(config.session.db_path, self.hooks)
        self.recovery = RecoveryManager(self.hooks, self.credentials)
        self.guardrails = ToolGuardrails(
            self.hooks,
            max_consecutive_calls=config.guardrail.max_consecutive_calls,
            max_calls_per_minute=config.guardrail.max_calls_per_minute,
        )

        # P2: Engineering
        self.scheduler = CronScheduler(
            self._run_loop_tick, self.hooks, config.session.db_path,
        )
        self.kanban = KanbanBoard(config.session.db_path)
        self.plugins = PluginManager()
        self.security = SecurityManager()
        self.skills = SkillManager(config.session.db_path)
        self.skill_factory = SkillFactory(
            skill_manager=self.skills,
            learning_manager=None,  # set below after learning is created
        )

        # P3: Advanced
        self.learning = LearningManager(self.hooks, self.memory)
        self.skill_factory.learning_manager = self.learning
        self.snapshot = SnapshotManager(config.session.db_path, self.workspace)
        self.trajectory = TrajectoryRecorder(self.hooks, self.workspace)
        self.experiments = ExperimentManager(config.session.db_path, self.hooks)

        # Research method engines
        self.workspace_guard = WorkspaceGuard(self.workspace)
        self.result_store = ResultStore(self.workspace)
        self._session_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # Apply configured seed (if any) so engine RNGs use a single source.
        configured_seed = getattr(config.session, "seed", None)
        if configured_seed is not None:
            GlobalSeed.set(configured_seed)
        self.stat_engine = StatisticalEngine(store=self.result_store, guard=self.workspace_guard)
        self.design_engine = ResearchDesignEngine(store=self.result_store, guard=self.workspace_guard)
        self.causal_engine = CausalEngine(store=self.result_store, guard=self.workspace_guard)
        self.survey_engine = SurveyEngine(store=self.result_store, guard=self.workspace_guard)
        self.qual_engine = QualitativeEngine(self.provider, store=self.result_store, guard=self.workspace_guard)
        self.meta_engine = MetaAnalysisEngine(store=self.result_store, guard=self.workspace_guard)
        self.comp_engine = ComputationalEngine()
        self.ml_engine = MLEngine()
        self.llm_engine = LLMEngine(self.provider, config, store=self.result_store, guard=self.workspace_guard)
        self.viz_engine = VisualizationEngine(self.workspace, store=self.result_store, guard=self.workspace_guard)
        self.pipeline = ExperimentPipeline(self.workspace, store=self.result_store)

        # Self-evolving discovery system
        self.method_catalog = MethodCatalog(config.session.db_path)
        self.advisor = MethodologyAdvisor(catalog=self.method_catalog)
        self.empirical_workflow = EmpiricalWorkflowEngine(
            self.workspace,
            store=self.result_store,
            pipeline=self.pipeline,
            advisor=self.advisor,
        )
        self.latex_reporter = LaTeXReporter(self.result_store)
        self.skill_factory.catalog = self.method_catalog
        self.dep_manager = DependencyManager()
        self.method_searcher = MethodSearcher(self.method_catalog, self.provider)
        self.method_builder = MethodBuilder(self.method_catalog, self.provider)

        # Register all tools and hooks
        self._register_tools()
        self._register_hooks()

        logger.info(
            "SophiaAgent initialized: model=%s, workspace=%s, tools=%d",
            config.model.name, self.workspace, len(self.tools.list_tools()),
        )

    def _register_tools(self):
        register_file_tools(self.tools, self.workspace)
        register_research_tools(self.tools)
        register_citation_tools(self.tools, self.workspace)
        register_writing_tools(self.tools, self.workspace, self.result_store)
        register_analysis_tools(self.tools, self.workspace)
        register_web_tools(self.tools)
        register_review_tools(self.tools, self.workspace)

        # Data collection (macro / finance / scrape / news)
        try:
            register_data_collection_tools(self.tools, self.result_store, self.workspace_guard)
        except Exception as e:
            logger.warning("Data collection tools registration failed: %s", e)

        # P0
        register_goal_tools(self.tools, self.goals)
        register_swarm_tools(self.tools, self.swarm_orchestrator)
        register_subagent_tools(self.tools, self.subagents)
        register_loop_tools(self.tools, self.loops)
        # P1
        register_memory_tools(self.tools, self.memory)
        # P2
        register_kanban_tools(self.tools, self.kanban)
        # Experiment
        register_experiment_tools(self.tools, self.experiments)

        # Research method engines (77 tools)
        register_method_tools(self.tools, {
            "statistics": self.stat_engine,
            "design": self.design_engine,
            "causal": self.causal_engine,
            "survey": self.survey_engine,
            "qualitative": self.qual_engine,
            "meta": self.meta_engine,
            "computational": self.comp_engine,
            "ml": self.ml_engine,
            "llm": self.llm_engine,
            "visualization": self.viz_engine,
            "pipeline": self.pipeline,
            "advisor": self.advisor,
            "empirical_workflow": self.empirical_workflow,
            "latex_reporter": self.latex_reporter,
        })

        # Self-evolving discovery system (5 tools)
        register_discovery_tools(self.tools, {
            "catalog": self.method_catalog,
            "searcher": self.method_searcher,
            "builder": self.method_builder,
            "dep_manager": self.dep_manager,
        })

        # Skill system tools
        self._register_skill_tools()

    def _register_hooks(self):
        self.hooks.register(
            "tool.pre_dispatch", self.guardrails.check_hook,
            priority=10, name="guardrails",
        )
        self.hooks.register(
            "tool.error", self.recovery.on_tool_error,
            priority=50, name="recovery",
        )
        self.hooks.register(
            "tool.post_dispatch",
            lambda ctx: self.learning.record_execution("tool.post_dispatch", ctx),
            priority=90, name="learning_record",
        )
        self.hooks.register(
            "goal.completed", self.learning.on_goal_completed,
            priority=50, name="learning_goal",
        )

    def _register_skill_tools(self):
        """Register skill management tools."""
        import json

        def _skill_create(args):
            name = args.get("name", "").strip()
            workflow = args.get("workflow", [])
            if not name or not workflow:
                return json.dumps({"success": False, "error": "name and workflow required"}, ensure_ascii=False)
            sid = self.skill_factory.create_skill(
                name=name,
                workflow=workflow,
                trigger=args.get("trigger"),
                description=args.get("description", ""),
                category=args.get("category", "general"),
            )
            return json.dumps({"success": True, "skill_id": sid}, ensure_ascii=False)

        self.tools.register(
            "skill_create",
            "Create a reusable skill from a workflow definition",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "workflow": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string"},
                                "params": {"type": "object"},
                            },
                        },
                    },
                    "trigger": {"type": "object"},
                    "description": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["name", "workflow"],
            },
            _skill_create,
        )

        def _skill_execute(args):
            skill_id = args.get("skill_id", "").strip()
            if not skill_id:
                return json.dumps({"success": False, "error": "skill_id required"}, ensure_ascii=False)
            result = self.skill_factory.execute_skill(
                skill_id,
                initial_args=args.get("args"),
                registry=self.tools,
                result_store=self.result_store,
            )
            return json.dumps(result, ensure_ascii=False)

        self.tools.register(
            "skill_execute",
            "Execute a previously created skill workflow",
            {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "args": {"type": "object"},
                },
                "required": ["skill_id"],
            },
            _skill_execute,
        )

        def _skill_list(args):
            skills = self.skills.list_skills(category=args.get("category"))
            return json.dumps({"count": len(skills), "skills": skills}, ensure_ascii=False)

        self.tools.register(
            "skill_list",
            "List all available skills",
            {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                },
            },
            _skill_list,
        )

        def _skill_auto_discover(args):
            top_n = args.get("top_n", 3)
            installed = self.skill_factory.auto_generate_from_logs(top_n=top_n)
            return json.dumps({"success": True, "generated": installed}, ensure_ascii=False)

        self.tools.register(
            "skill_auto_discover",
            "Automatically discover and create skills from execution history",
            {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer", "default": 3},
                },
            },
            _skill_auto_discover,
        )

        def _skill_evolve(args):
            skill_id = args.get("skill_id", "").strip()
            if not skill_id:
                return json.dumps({"success": False, "error": "skill_id required"}, ensure_ascii=False)
            result = self.skill_factory.auto_evolve(
                skill_id,
                min_executions=args.get("min_executions", 5),
                success_rate_threshold=args.get("success_rate_threshold", 0.7),
            )
            return json.dumps(result, ensure_ascii=False)

        self.tools.register(
            "skill_evolve",
            "Evolve a skill based on its execution history (auto-tune and version bump)",
            {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "min_executions": {"type": "integer", "default": 5},
                    "success_rate_threshold": {"type": "number", "default": 0.7},
                },
                "required": ["skill_id"],
            },
            _skill_evolve,
        )

    def _llm_text(self, prompt: str) -> str:
        response = self.provider.chat([{"role": "user", "content": prompt}], tools=None)
        return response.content or ""

    def _run_subagent(self, prompt: str, allowed_tools=None) -> str:
        return self._run_internal(prompt, allowed_tools=allowed_tools, allow_swarm=False)

    def _run_loop_tick(self, prompt: str) -> str:
        return self._run_internal(prompt, allow_swarm=False)

    def _run_internal(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        allow_swarm: bool = False,
    ) -> str:
        return self.run(
            user_message,
            history=history,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            allow_swarm=allow_swarm,
        )

    def _inject_workspace_context(self, user_message: str, workspace_context) -> str:
        block = workspace_context.to_prompt_block(user_message=user_message)
        paper_contract = build_paper_generation_contract(user_message)
        reference_notice = build_reference_priority_notice(
            user_message,
            workspace_has_evidence=workspace_context.has_evidence,
        )
        task_harness = build_task_harness_prompt(
            user_message,
            workspace_has_evidence=workspace_context.has_evidence,
            empirical_preflight=self._build_empirical_preflight(user_message),
        )
        if not block:
            parts = [user_message]
            if task_harness:
                parts.append(task_harness)
            if reference_notice:
                parts.append(reference_notice)
            if paper_contract:
                parts.append(paper_contract)
            return "\n\n".join(parts)
        requirements = [
            "【强制执行约束】",
            "1. 已读取的工作空间材料是本次回答的主要证据来源。",
            "2. 文中引用只能来自工作空间材料或真实工具返回结果，严禁编造引用。",
            "3. 如果工作空间材料不足，必须明确标注不足，不得用看似真实的作者年份补齐。",
        ]
        if asks_for_paper_document(user_message):
            requirements.extend([
                "4. 按用户要求生成完整论文正文，严格控制标题层级；不要自行添加三级标题。",
                "5. 论文完成后，系统会自动保存 Markdown 文档；正文中仍需给出完整内容。",
            ])
        if asks_for_paper_document(user_message):
            requirements.append(
                "6. If the user requests Word/DOCX, the final saved deliverable "
                "must be Word/DOCX, not Markdown."
            )
        parts = [user_message, block, "\n".join(requirements)]
        if reference_notice:
            parts.append(reference_notice)
        if task_harness:
            parts.append(task_harness)
        if paper_contract:
            parts.append(paper_contract)
        return "\n\n".join(parts)

    def _build_empirical_preflight(self, user_message: str) -> str:
        if not is_empirical_request(user_message):
            return ""
        try:
            return self.empirical_workflow.plan({"research_question": user_message})
        except Exception as exc:
            logger.warning("Empirical preflight failed: %s", exc)
            return ""

    def _append_generated_document_path(self, user_message: str, final_text: str) -> str:
        final_text = append_quality_report_if_needed(user_message, final_text)
        output_format = requested_output_format(user_message)
        if output_format == "docx":
            path = save_generated_docx(self.workspace, user_message, final_text)
            if not path:
                return final_text
            return f"{final_text}\n\n---\n已自动生成 Word 文档：{path}"

        path = save_generated_markdown(self.workspace, user_message, final_text)
        if not path:
            return final_text
        return f"{final_text}\n\n---\n已自动生成 Markdown 文档：{path}"

    def _maybe_repair_paper_output(
        self,
        user_message: str,
        final_text: str,
        messages: List[Dict[str, Any]],
        registry,
        tool_schemas,
    ) -> str:
        if not is_paper_generation_request(user_message) or not final_text.strip():
            return final_text
        report = inspect_generated_paper(final_text)
        if report.passed:
            return final_text

        repair_prompt = (
            "[Sophia internal quality repair]\n"
            "The previous paper draft failed mandatory quality gates. Continue repairing it now. "
            "Do not ask the user to fix routine quality problems. Expand the body, add verified "
            "references, add meaningful tables and figures, and preserve the user's requested "
            "structure and export format. Never fabricate citations, data, or empirical results.\n\n"
            f"Failed checks: {report.issues}\n\n"
            "Previous draft:\n"
            f"{final_text}"
        )
        repair_messages = list(messages)
        repair_messages.append({"role": "user", "content": repair_prompt})
        try:
            repaired = self.provider.chat(repair_messages, tools=tool_schemas)
        except Exception as exc:
            logger.warning("Paper repair pass failed: %s", exc)
            return final_text

        if repaired.tool_calls:
            repair_messages.append(repaired.to_dict())
            for tc in repaired.tool_calls:
                try:
                    result = registry.dispatch(tc.name, tc.arguments)
                except Exception as exc:
                    result = f"Tool failed during repair: {exc}"
                repair_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            try:
                repaired = self.provider.chat(repair_messages, tools=tool_schemas)
            except Exception as exc:
                logger.warning("Paper repair synthesis failed: %s", exc)
                return final_text

        return repaired.content or final_text

    def run(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        allow_swarm: bool = True,
    ) -> str:
        if system_prompt is None:
            system_prompt = build_system_prompt(self.workspace)

        requirement_check = check_paper_requirements(user_message)
        if requirement_check.requires_clarification:
            return requirement_check.message

        workspace_context = collect_workspace_context(self.workspace, user_message)
        effective_user_message = self._inject_workspace_context(user_message, workspace_context)

        if allow_swarm and allowed_tools is None:
            decision = self.swarm_orchestrator.analyze(effective_user_message)
            if decision.need_swarm:
                try:
                    final_text = self.swarm_orchestrator.execute(
                        decision,
                        effective_user_message,
                        history=history,
                        system_prompt=system_prompt,
                    )
                    return self._append_generated_document_path(user_message, final_text)
                except Exception as exc:
                    logger.warning("Swarm execution failed; falling back to single-agent run: %s", exc)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": effective_user_message})

        # Autopilot: augment messages based on intent
        messages = self.autopilot.before_run(effective_user_message, messages)

        # Auto-compress context if approaching token limit
        messages = self.context_compressor.maybe_compress(messages)

        self.hooks.emit(HookEvent.AGENT_PRE_RUN, {"messages": messages})

        registry = FilteredToolRegistry(self.tools, allowed_tools) if allowed_tools is not None else self.tools
        tool_schemas = registry.get_schemas()
        if not tool_schemas:
            tool_schemas = None

        for turn in range(self.max_turns):
            logger.debug("Turn %d/%d", turn + 1, self.max_turns)

            response = self.provider.chat(
                messages=messages,
                tools=tool_schemas,
            )

            messages.append(response.to_dict())

            if not response.tool_calls:
                self.hooks.emit(HookEvent.AGENT_POST_RUN, {"messages": messages})
                final_text = self._maybe_repair_paper_output(
                    user_message,
                    response.content or "",
                    messages,
                    registry,
                    tool_schemas,
                )
                return self._append_generated_document_path(user_message, final_text)

            for tc in response.tool_calls:
                logger.info("Tool call: %s(%s)", tc.name, tc.arguments)
                result = registry.dispatch(tc.name, tc.arguments)
                logger.debug("Tool result (%d chars): %s", len(result), result[:200])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        self.hooks.emit(HookEvent.AGENT_POST_RUN, {"messages": messages})
        final_text = self._maybe_repair_paper_output(
            user_message,
            messages[-1].get("content", "") if messages else "",
            messages,
            registry,
            tool_schemas,
        )
        return self._append_generated_document_path(user_message, final_text)

    def chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if history is None:
            history = []

        response_text = self.run(
            user_message=user_message,
            history=list(history),
        )

        return {
            "response": response_text,
            "history": history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response_text},
            ],
        }

    def run_stream(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        allow_swarm: bool = True,
    ):
        if system_prompt is None:
            system_prompt = build_system_prompt(self.workspace)

        if history is None:
            history = []

        requirement_check = check_paper_requirements(user_message)
        if requirement_check.requires_clarification:
            yield {"type": "token", "content": requirement_check.message}
            yield {
                "type": "done",
                "response": requirement_check.message,
                "history": history + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": requirement_check.message},
                ],
                "usage": dict(self._session_tokens),
            }
            return

        workspace_context = None
        for event in iter_workspace_context_events(self.workspace, user_message):
            if event["type"] == "workspace_context_complete":
                workspace_context = event["context"]
                if workspace_context.requested:
                    yield {
                        "type": "workspace_scan_done",
                        "read_files": event.get("read_files", len(workspace_context.evidences)),
                        "skipped_files": event.get("skipped_files", len(workspace_context.skipped)),
                        "total_files": event.get("total_files", workspace_context.total_candidates),
                    }
                break
            yield event
        if workspace_context is None:
            workspace_context = collect_workspace_context(self.workspace, user_message)
        effective_user_message = self._inject_workspace_context(user_message, workspace_context)
        if allow_swarm and allowed_tools is None:
            decision = self.swarm_orchestrator.analyze(effective_user_message)
            if decision.need_swarm:
                final_parts = []
                try:
                    for event in self.swarm_orchestrator.execute_stream(
                        decision,
                        effective_user_message,
                        history=history,
                        system_prompt=system_prompt,
                    ):
                        if event.get("type") == "token":
                            final_parts.append(event.get("content", ""))
                        yield event
                    final_text = "".join(final_parts)
                    final_text = self._append_generated_document_path(user_message, final_text)
                    if final_text != "".join(final_parts):
                        saved_note = final_text[len("".join(final_parts)):]
                        yield {"type": "token", "content": saved_note}
                    yield {
                        "type": "done",
                        "response": final_text,
                        "history": history + [
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": final_text},
                        ],
                        "usage": dict(self._session_tokens),
                    }
                    return
                except Exception as exc:
                    logger.warning("Swarm stream failed; falling back to single-agent stream: %s", exc)
                    yield {
                        "type": "swarm_error",
                        "error": "Swarm failed and Sophia is continuing with the main agent.",
                        "detail": str(exc),
                    }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": effective_user_message})

        # Autopilot: augment messages based on intent
        messages = self.autopilot.before_run(effective_user_message, messages)

        # Auto-compress context if approaching token limit
        messages = self.context_compressor.maybe_compress(messages)

        self.hooks.emit(HookEvent.AGENT_PRE_STREAM, {"messages": messages})

        registry = FilteredToolRegistry(self.tools, allowed_tools) if allowed_tools is not None else self.tools
        tool_schemas = registry.get_schemas()
        if not tool_schemas:
            tool_schemas = None

        for turn in range(self.max_turns):
            full_text = []
            gen = self.provider.chat_stream(messages, tools=tool_schemas)

            try:
                while True:
                    try:
                        chunk = next(gen)
                        full_text.append(chunk)
                        yield {"type": "token", "content": chunk}
                    except StopIteration as e:
                        response = e.value
                        break
            except Exception:
                response = self.provider.chat(messages, tools=tool_schemas)
                if response.content:
                    full_text.append(response.content)
                    yield {"type": "token", "content": response.content}

            response.content = "".join(full_text) or response.content
            messages.append(response.to_dict())

            if response.usage:
                for k in self._session_tokens:
                    self._session_tokens[k] += response.usage.get(k, 0)

            if not response.tool_calls:
                final_text = self._maybe_repair_paper_output(
                    user_message,
                    response.content or "",
                    messages,
                    registry,
                    tool_schemas,
                )
                final_text = self._append_generated_document_path(user_message, final_text)
                if final_text != (response.content or ""):
                    saved_note = final_text[len(response.content or ""):]
                    yield {"type": "token", "content": saved_note}
                self.hooks.emit(HookEvent.AGENT_POST_STREAM, {"messages": messages})
                yield {
                    "type": "done",
                    "response": final_text,
                    "history": history + [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": final_text},
                    ],
                    "usage": dict(self._session_tokens),
                }
                return

            for tc in response.tool_calls:
                yield {"type": "tool_call", "name": tc.name, "arguments": tc.arguments}
                result = registry.dispatch(tc.name, tc.arguments)
                yield {"type": "tool_result", "name": tc.name, "result": result[:500]}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        self.hooks.emit(HookEvent.AGENT_POST_STREAM, {"messages": messages})
        final_text = self._maybe_repair_paper_output(
            user_message,
            messages[-1].get("content", ""),
            messages,
            registry,
            tool_schemas,
        )
        final_text = self._append_generated_document_path(user_message, final_text)
        if final_text != messages[-1].get("content", ""):
            saved_note = final_text[len(messages[-1].get("content", "")):]
            yield {"type": "token", "content": saved_note}
        yield {
            "type": "done",
            "response": final_text,
            "history": history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final_text},
            ],
            "usage": dict(self._session_tokens),
        }

    def reset_session_tokens(self):
        """Reset cumulative token counters (e.g. on session resume)."""
        self._session_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def reconfigure(self, config: Config):
        """Hot-swap provider and workspace at runtime."""
        self.config = config
        self.provider = create_provider(config)
        self.workspace = config.session.workspace
        self.workspace_guard = WorkspaceGuard(self.workspace)
        self.result_store = ResultStore(self.workspace)
        logger.info("Agent reconfigured: model=%s, workspace=%s", config.model.name, self.workspace)
