from sophia.hooks import HookEvent, HookManager
from sophia.swarm.models import SwarmDecision
from sophia.swarm.orchestrator import SwarmOrchestrator


def test_hook_event_constants_exist():
    assert HookEvent.SWARM_ANALYZED == "swarm.analyzed"
    assert HookEvent.SWARM_SYNTHESIZED == "swarm.synthesized"


def test_orchestrator_emits_lifecycle_hooks():
    hooks = HookManager()
    events = []
    for event in [HookEvent.SWARM_PLANNED, HookEvent.SWARM_STAGE_START, HookEvent.SWARM_AGENT_COMPLETE, HookEvent.SWARM_SYNTHESIZED]:
        hooks.register(event, lambda ctx, event=event: (events.append(event), ctx)[1])
    orch = SwarmOrchestrator(lambda prompt, tools=None: "ok", hooks=hooks)
    orch.execute(SwarmDecision(True, recommended_roles=["writer"]), "复杂写作任务")
    assert HookEvent.SWARM_PLANNED in events
    assert HookEvent.SWARM_SYNTHESIZED in events
