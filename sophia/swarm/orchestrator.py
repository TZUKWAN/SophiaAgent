"""Automatic swarm orchestrator for SophiaAgent."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from inspect import signature
from typing import Any, Callable, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager
from sophia.swarm.analyzer import TaskAnalyzer
from sophia.swarm.bus import SwarmBus
from sophia.swarm.decomposer import TaskDecomposer
from sophia.swarm.models import AgentResult, AgentSpec, Stage, SwarmDecision, SwarmExecutionRecord
from sophia.swarm.roles import RoleTemplateBank
from sophia.swarm.synthesizer import ResultSynthesizer

logger = logging.getLogger(__name__)


class FilteredToolRegistry:
    """A per-call registry wrapper that exposes only whitelisted tools."""

    def __init__(self, parent_registry, allowed_tools: Optional[List[str]]):
        self._parent = parent_registry
        self._allowed = set(allowed_tools or [])

    def dispatch(self, name: str, args: Dict[str, Any]) -> str:
        if self._allowed and name not in self._allowed:
            return json.dumps(
                {
                    "error": f"Tool '{name}' is not allowed for this sub-agent.",
                    "allowed_tools": sorted(self._allowed),
                },
                ensure_ascii=False,
            )
        return self._parent.dispatch(name, args)

    def get_schemas(self):
        schemas = self._parent.get_schemas()
        if not self._allowed:
            return schemas
        return [
            schema
            for schema in schemas
            if schema.get("function", {}).get("name") in self._allowed
        ]

    def list_tools(self) -> List[str]:
        if not self._allowed:
            return self._parent.list_tools()
        return sorted(self._allowed)


class SwarmOrchestrator:
    """Analyze, plan, execute, and synthesize automatic sub-agent swarms."""

    def __init__(
        self,
        run_fn: Callable[..., str],
        llm_call_fn: Optional[Callable[[str], str]] = None,
        hooks: Optional[HookManager] = None,
        max_workers: int = 4,
    ):
        self._run_fn = run_fn
        self._llm_call_fn = llm_call_fn
        self.hooks = hooks or HookManager()
        self.max_workers = max_workers
        self.role_bank = RoleTemplateBank()
        self.analyzer = TaskAnalyzer(llm_call=llm_call_fn, use_llm=llm_call_fn is not None)
        self.decomposer = TaskDecomposer(
            llm_call=llm_call_fn,
            role_bank=self.role_bank,
            use_llm=llm_call_fn is not None,
        )
        self.synthesizer = ResultSynthesizer(llm_call=llm_call_fn)
        self._lock = threading.RLock()
        self._executions: Dict[str, SwarmExecutionRecord] = {}
        self._run_fn_accepts_keywords = self._detect_keyword_run_fn(run_fn)

    def analyze(self, user_message: str) -> SwarmDecision:
        decision = self.analyzer.analyze(user_message)
        self.hooks.emit(
            HookEvent.SWARM_ANALYZED,
            {"message": user_message, "decision": decision.to_dict()},
        )
        return decision

    def execute(
        self,
        decision: SwarmDecision,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        session_id: str = "default",
    ) -> str:
        if not decision.need_swarm:
            return self._call_run_fn(
                user_message,
                history=history,
                system_prompt=system_prompt,
                allowed_tools=None,
            )

        record, bus = self._prepare_execution(decision, user_message, session_id)
        all_results: List[AgentResult] = []
        record.status = "running"

        try:
            for stage in record.plan.stages:
                self.hooks.emit(
                    HookEvent.SWARM_STAGE_START,
                    {
                        "execution_id": record.execution_id,
                        "stage_id": stage.stage_id,
                        "agent_count": len(stage.agents),
                    },
                )
                stage_results = self._execute_stage(stage, bus, history, system_prompt)
                all_results.extend(stage_results)
                self.hooks.emit(
                    HookEvent.SWARM_STAGE_END,
                    {
                        "execution_id": record.execution_id,
                        "stage_id": stage.stage_id,
                        "results": [result.to_dict() for result in stage_results],
                    },
                )

            final = self.synthesizer.synthesize(all_results, record.plan)
            record.results = all_results
            record.final_synthesis = final
            record.status = "completed"
            record.completed_at = datetime.now()
            record.total_tokens = self._aggregate_tokens(all_results)
            self.hooks.emit(
                HookEvent.SWARM_SYNTHESIZED,
                {
                    "execution_id": record.execution_id,
                    "agent_count": len(all_results),
                    "final_length": len(final),
                },
            )
            return final
        except Exception:
            record.status = "failed"
            record.completed_at = datetime.now()
            raise

    def execute_stream(
        self,
        decision: SwarmDecision,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        session_id: str = "default",
    ):
        if not decision.need_swarm:
            yield {"type": "swarm_skip", "reason": decision.reason}
            final = self._call_run_fn(
                user_message,
                history=history,
                system_prompt=system_prompt,
                allowed_tools=None,
            )
            yield {"type": "token", "content": final}
            return

        yield {"type": "swarm_analyze", "need_swarm": True, "reason": decision.reason}
        record, bus = self._prepare_execution(decision, user_message, session_id)
        yield {
            "type": "swarm_plan",
            "execution_id": record.execution_id,
            "workflow": record.plan.workflow,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "parallel": stage.parallel,
                    "agents": [
                        {"agent_id": agent.agent_id, "role_id": agent.role_id}
                        for agent in stage.agents
                    ],
                }
                for stage in record.plan.stages
            ],
        }

        all_results: List[AgentResult] = []
        record.status = "running"
        for stage in record.plan.stages:
            yield {"type": "swarm_stage_start", "stage_id": stage.stage_id}
            stage_results = self._execute_stage(stage, bus, history, system_prompt)
            all_results.extend(stage_results)
            for result in stage_results:
                yield {
                    "type": "swarm_agent_complete"
                    if result.status == "completed"
                    else "swarm_agent_error",
                    "agent_id": result.agent_id,
                    "role_id": result.role_id,
                    "status": result.status,
                    "error": result.error,
                }
            yield {"type": "swarm_stage_end", "stage_id": stage.stage_id}

        yield {"type": "swarm_synthesize", "agent_count": len(all_results)}
        final = self.synthesizer.synthesize(all_results, record.plan)
        record.results = all_results
        record.final_synthesis = final
        record.status = "completed"
        record.completed_at = datetime.now()
        record.total_tokens = self._aggregate_tokens(all_results)
        self.hooks.emit(
            HookEvent.SWARM_SYNTHESIZED,
            {
                "execution_id": record.execution_id,
                "agent_count": len(all_results),
                "final_length": len(final),
            },
        )
        yield {"type": "token", "content": final}
        yield {"type": "swarm_done", "execution_id": record.execution_id, "final_length": len(final)}

    def _prepare_execution(
        self,
        decision: SwarmDecision,
        user_message: str,
        session_id: str,
    ) -> tuple[SwarmExecutionRecord, SwarmBus]:
        plan = self.decomposer.decompose(user_message, decision)
        record = SwarmExecutionRecord(
            execution_id=uuid.uuid4().hex[:8],
            session_id=session_id,
            decision=decision,
            plan=plan,
        )
        with self._lock:
            self._executions[record.execution_id] = record
        self.hooks.emit(
            HookEvent.SWARM_PLANNED,
            {"execution_id": record.execution_id, "plan": plan.to_dict()},
        )
        bus = SwarmBus()
        for stage in plan.stages:
            for agent in stage.agents:
                bus.register_agent(agent.agent_id)
        return record, bus

    def _execute_stage(
        self,
        stage: Stage,
        bus: SwarmBus,
        history: Optional[List[Dict[str, Any]]],
        system_prompt: Optional[str],
    ) -> List[AgentResult]:
        if not stage.parallel or len(stage.agents) <= 1:
            return [self._run_agent(agent, bus, history, system_prompt) for agent in stage.agents]

        results: List[AgentResult] = []
        with ThreadPoolExecutor(max_workers=min(len(stage.agents), self.max_workers)) as executor:
            future_map = {
                executor.submit(self._run_agent, agent, bus, history, system_prompt): agent
                for agent in stage.agents
            }
            for future in as_completed(future_map):
                agent = future_map[future]
                try:
                    results.append(future.result(timeout=agent.timeout))
                except TimeoutError:
                    results.append(
                        AgentResult(
                            agent_id=agent.agent_id,
                            role_id=agent.role_id,
                            status="timeout",
                            error=f"Timed out after {agent.timeout}s",
                        )
                    )
                except Exception as exc:
                    results.append(
                        AgentResult(
                            agent_id=agent.agent_id,
                            role_id=agent.role_id,
                            status="failed",
                            error=str(exc),
                        )
                    )
        return results

    def _run_agent(
        self,
        spec: AgentSpec,
        bus: SwarmBus,
        history: Optional[List[Dict[str, Any]]],
        system_prompt: Optional[str],
    ) -> AgentResult:
        result = AgentResult(
            agent_id=spec.agent_id,
            role_id=spec.role_id,
            status="running",
            start_time=datetime.now(),
        )
        self.hooks.emit(
            HookEvent.SWARM_SPAWNED,
            {
                "agent_id": spec.agent_id,
                "role_id": spec.role_id,
                "tools": spec.tools,
            },
        )
        try:
            bus_context = bus.to_context_string(max_length=5000)
            prompt = spec.task_prompt
            if bus_context:
                prompt += f"\n\n以下是蜂群通信总线中的已有专家输出：\n{bus_context}\n"
            content = self._call_run_fn(
                prompt,
                history=history,
                system_prompt=spec.system_prompt or system_prompt,
                allowed_tools=spec.tools,
            )
            result.status = "completed"
            result.content = content
            result.end_time = datetime.now()
            bus.write(spec.agent_id, content, msg_type="result", metadata={"role_id": spec.role_id})
            self.hooks.emit(
                HookEvent.SWARM_AGENT_COMPLETE,
                {
                    "agent_id": spec.agent_id,
                    "role_id": spec.role_id,
                    "content_length": len(content),
                },
            )
        except Exception as exc:
            logger.exception("Swarm agent failed: %s", spec.agent_id)
            result.status = "failed"
            result.error = str(exc)
            result.end_time = datetime.now()
            self.hooks.emit(
                HookEvent.SWARM_AGENT_ERROR,
                {"agent_id": spec.agent_id, "role_id": spec.role_id, "error": str(exc)},
            )
        return result

    def delegate(
        self,
        session_id: str,
        prompt: str,
        tools: Optional[List[str]] = None,
        goal_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        role_id = "synthesizer" if not tools else "writer"
        spec = AgentSpec(
            agent_id=f"manual_{uuid.uuid4().hex[:4]}",
            role_id=role_id,
            task_prompt=prompt,
            tools=tools or [],
            metadata={"goal_id": goal_id},
        )
        bus = SwarmBus()
        bus.register_agent(spec.agent_id)
        result = self._run_agent(spec, bus, history=None, system_prompt=None)
        return {
            "id": result.agent_id,
            "status": result.status,
            "result": result.content[:500],
            "error": result.error,
        }

    def delegate_batch(self, session_id: str, tasks: List[Dict]) -> List[Dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=min(len(tasks) or 1, self.max_workers)) as executor:
            futures = [
                executor.submit(
                    self.delegate,
                    session_id,
                    task["prompt"],
                    task.get("tools"),
                    task.get("goal_id"),
                )
                for task in tasks
            ]
            return [future.result() for future in futures]

    def list_executions(self, session_id: Optional[str] = None) -> List[SwarmExecutionRecord]:
        with self._lock:
            records = list(self._executions.values())
        if session_id:
            records = [record for record in records if record.session_id == session_id]
        return sorted(records, key=lambda record: record.created_at, reverse=True)

    def get_execution(self, execution_id: str) -> Optional[SwarmExecutionRecord]:
        with self._lock:
            return self._executions.get(execution_id)

    @staticmethod
    def _aggregate_tokens(results: List[AgentResult]) -> Dict[str, int]:
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for result in results:
            for key in total:
                total[key] += int(result.tokens_used.get(key, 0))
        return total

    @staticmethod
    def _detect_keyword_run_fn(run_fn: Callable[..., str]) -> bool:
        try:
            params = signature(run_fn).parameters
        except (TypeError, ValueError):
            return True
        if any(param.kind == param.VAR_KEYWORD for param in params.values()):
            return True
        return {"history", "system_prompt", "allowed_tools", "allow_swarm"}.issubset(params)

    def _call_run_fn(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]],
        system_prompt: Optional[str],
        allowed_tools: Optional[List[str]],
    ) -> str:
        if self._run_fn_accepts_keywords:
            return self._run_fn(
                prompt,
                history=history,
                system_prompt=system_prompt,
                allowed_tools=allowed_tools,
                allow_swarm=False,
            )
        return self._run_fn(prompt, allowed_tools)
