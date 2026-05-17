"""Data models for SophiaAgent's automatic swarm orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SwarmDecision:
    """Decision produced by the task analyzer."""

    need_swarm: bool
    reason: str = ""
    estimated_roles: int = 0
    confidence: float = 0.0
    recommended_roles: List[str] = field(default_factory=list)
    workflow: str = "parallel"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentSpec:
    """Execution spec for one specialized sub-agent."""

    agent_id: str
    role_id: str
    task_prompt: str
    tools: List[str] = field(default_factory=list)
    timeout: int = 300
    depends_on: List[str] = field(default_factory=list)
    system_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Stage:
    """A stage in a swarm workflow."""

    stage_id: str
    parallel: bool = True
    agents: List[AgentSpec] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "parallel": self.parallel,
            "agents": [agent.to_dict() for agent in self.agents],
            "depends_on": list(self.depends_on),
        }


@dataclass
class SwarmPlan:
    """Full execution plan for a complex user request."""

    workflow: str = "parallel"
    stages: List[Stage] = field(default_factory=list)
    coordinator_prompt: str = (
        "整合所有专家输出，去除重复，标注冲突，形成一份面向用户的最终答复。"
    )
    original_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow": self.workflow,
            "stages": [stage.to_dict() for stage in self.stages],
            "coordinator_prompt": self.coordinator_prompt,
            "original_query": self.original_query,
        }

    @property
    def agent_count(self) -> int:
        return sum(len(stage.agents) for stage in self.stages)


@dataclass
class AgentResult:
    """Result from one sub-agent."""

    agent_id: str
    role_id: str
    status: str = "pending"
    content: str = ""
    error: Optional[str] = None
    tool_calls_made: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tokens_used: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[int]:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat() if self.start_time else None
        data["end_time"] = self.end_time.isoformat() if self.end_time else None
        data["duration_ms"] = self.duration_ms
        return data


@dataclass
class SwarmExecutionRecord:
    """In-memory record for one swarm execution."""

    execution_id: str
    session_id: str
    decision: SwarmDecision
    plan: SwarmPlan
    results: List[AgentResult] = field(default_factory=list)
    final_synthesis: str = ""
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_tokens: Dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "status": self.status,
            "need_swarm": self.decision.need_swarm,
            "workflow": self.plan.workflow,
            "stage_count": len(self.plan.stages),
            "agent_count": self.plan.agent_count,
            "completed_agents": sum(1 for r in self.results if r.status == "completed"),
            "failed_agents": sum(1 for r in self.results if r.status in {"failed", "timeout"}),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
