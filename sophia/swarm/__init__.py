"""Automatic swarm orchestration package for SophiaAgent."""

from sophia.swarm.analyzer import TaskAnalyzer
from sophia.swarm.bus import BusMessage, SwarmBus
from sophia.swarm.decomposer import TaskDecomposer
from sophia.swarm.models import AgentResult, AgentSpec, Stage, SwarmDecision, SwarmPlan
from sophia.swarm.orchestrator import FilteredToolRegistry, SwarmOrchestrator
from sophia.swarm.roles import RoleTemplate, RoleTemplateBank
from sophia.swarm.synthesizer import ResultSynthesizer

__all__ = [
    "AgentResult",
    "AgentSpec",
    "BusMessage",
    "FilteredToolRegistry",
    "ResultSynthesizer",
    "RoleTemplate",
    "RoleTemplateBank",
    "Stage",
    "SwarmBus",
    "SwarmDecision",
    "SwarmOrchestrator",
    "SwarmPlan",
    "TaskAnalyzer",
    "TaskDecomposer",
]
