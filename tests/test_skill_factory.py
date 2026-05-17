"""Tests for SkillFactory, ExecutionPatternMiner, and skill execution."""
import json
import time
import pytest

from sophia.learning import LearningManager
from sophia.skills import SkillManager
from sophia.skills.factory import SkillFactory
from sophia.skills.pattern_miner import ExecutionPatternMiner, ToolInvocation
from sophia.tools.registry import ToolRegistry


@pytest.fixture
def skill_mgr(tmp_path):
    return SkillManager(str(tmp_path / "skills.db"))


@pytest.fixture
def learning_mgr():
    return LearningManager()


@pytest.fixture
def factory(skill_mgr, learning_mgr):
    return SkillFactory(skill_manager=skill_mgr, learning_manager=learning_mgr)


class TestExecutionPatternMiner:
    def test_mine_empty_log(self):
        miner = ExecutionPatternMiner()
        patterns = miner.mine([])
        assert patterns == []

    def test_mine_single_sequence(self):
        miner = ExecutionPatternMiner(min_frequency=1, min_success_rate=0.5)
        log = [
            {"tool_name": "research_load_data", "phase": "end", "success": True, "score": 1.0, "args": {}, "timestamp": 1.0},
            {"tool_name": "research_validate_data", "phase": "end", "success": True, "score": 1.0, "args": {}, "timestamp": 2.0},
            {"tool_name": "research_did", "phase": "end", "success": True, "score": 1.0, "args": {}, "timestamp": 3.0},
        ]
        patterns = miner.mine(log)
        assert len(patterns) > 0
        top = patterns[0]
        assert top.sequence == ["research_load_data", "research_validate_data", "research_did"]
        assert top.success_rate == 1.0

    def test_mine_filters_low_success(self):
        miner = ExecutionPatternMiner(min_frequency=1, min_success_rate=0.9)
        log = [
            {"tool_name": "a", "phase": "end", "success": True, "score": 1.0, "timestamp": 1.0},
            {"tool_name": "b", "phase": "error", "success": False, "score": 0.0, "timestamp": 2.0},
        ]
        patterns = miner.mine(log)
        # Sequence a->b has 0% success rate, should be filtered
        assert all(p.success_rate >= 0.9 for p in patterns)

    def test_mine_multiple_sessions(self):
        miner = ExecutionPatternMiner(min_frequency=2, min_success_rate=0.5, max_gap_seconds=10)
        log = []
        base = 0
        for _ in range(3):
            log.append({"tool_name": "load", "phase": "end", "success": True, "score": 1.0, "timestamp": base + 1})
            log.append({"tool_name": "analyze", "phase": "end", "success": True, "score": 1.0, "timestamp": base + 2})
            base += 20  # gap > 10s creates new session
        patterns = miner.mine(log)
        assert len(patterns) > 0
        assert patterns[0].frequency >= 2

    def test_split_into_sessions_respects_gap(self):
        miner = ExecutionPatternMiner(max_gap_seconds=5)
        log = [
            {"tool_name": "a", "phase": "end", "timestamp": 1.0},
            {"tool_name": "b", "phase": "end", "timestamp": 2.0},
            {"tool_name": "c", "phase": "end", "timestamp": 20.0},
            {"tool_name": "d", "phase": "end", "timestamp": 21.0},
        ]
        sessions = miner._split_into_sessions(log)
        assert len(sessions) == 2
        assert [i.tool_name for i in sessions[0]] == ["a", "b"]
        assert [i.tool_name for i in sessions[1]] == ["c", "d"]


class TestSkillFactoryCreate:
    def test_create_skill_explicit(self, factory, skill_mgr):
        sid = factory.create_skill(
            name="Test Skill",
            workflow=[
                {"tool": "echo_tool", "params": {"text": "hello"}},
            ],
            trigger={"keywords": ["test"]},
            description="A test skill",
        )
        assert sid is not None
        skill = skill_mgr.get_skill(sid)
        assert skill is not None
        assert skill["name"] == "Test Skill"
        assert skill["workflow"][0]["tool"] == "echo_tool"
        assert skill["trigger"]["keywords"] == ["test"]

    def test_create_skill_generates_unique_id(self, factory):
        sid1 = factory.create_skill(name="Same Name", workflow=[{"tool": "t1", "params": {}}])
        sid2 = factory.create_skill(name="Same Name", workflow=[{"tool": "t1", "params": {}}])
        assert sid1 != sid2

    def test_skill_def_has_handler_code(self, factory, skill_mgr):
        sid = factory.create_skill(name="HC Test", workflow=[{"tool": "t1", "params": {}}])
        skill = skill_mgr.get_skill(sid)
        assert "handler_code" in skill
        assert "def handle" in skill["handler_code"]


class TestSkillFactoryExecute:
    def test_execute_skill_success(self, factory, skill_mgr):
        # Register a dummy tool
        reg = ToolRegistry()
        reg.register("double_tool", "Double", {"type": "object", "properties": {"x": {"type": "integer"}}},
                     lambda args: json.dumps({"result": args.get("x", 0) * 2}))

        sid = factory.create_skill(
            name="Double Workflow",
            workflow=[{"tool": "double_tool", "params": {"x": 5}}],
        )
        result = factory.execute_skill(sid, registry=reg)
        assert result["success"] is True
        assert result["completed_steps"] == 1

    def test_execute_skill_missing(self, factory):
        result = factory.execute_skill("nonexistent", registry=ToolRegistry())
        assert result["success"] is False
        assert "not found" in result["error"]

    def test_execute_skill_step_failure(self, factory, skill_mgr):
        reg = ToolRegistry()
        reg.register("fail_tool", "Fail", {"type": "object", "properties": {}},
                     lambda args: (_ for _ in ()).throw(RuntimeError("boom")))

        sid = factory.create_skill(
            name="Fail Workflow",
            workflow=[{"tool": "fail_tool", "params": {}}],
        )
        result = factory.execute_skill(sid, registry=reg)
        assert result["success"] is False
        assert result["completed_steps"] == 0

    def test_execute_skill_context_merge(self, factory, skill_mgr):
        reg = ToolRegistry()
        reg.register("step1", "S1", {"type": "object", "properties": {}},
                     lambda args: json.dumps({"result_id": "res_abc", "status": "success"}))
        reg.register("step2", "S2", {"type": "object", "properties": {"input_id": {"type": "string"}}},
                     lambda args: json.dumps({"received": args.get("input_id")}))

        sid = factory.create_skill(
            name="Context Merge",
            workflow=[
                {"tool": "step1", "params": {}, "result_slot": "step1_out"},
                {"tool": "step2", "params": {"input_id": "${step1_out.result_id}"}},
            ],
        )
        result = factory.execute_skill(sid, registry=reg)
        assert result["success"] is True
        # step2 should have received the result_id from step1
        step_results = result["step_results"]
        assert step_results[1]["status"] == "success"


class TestSkillFactoryAutoGenerate:
    def test_auto_generate_from_logs_empty(self, factory):
        installed = factory.auto_generate_from_logs()
        assert installed == []

    def test_auto_generate_from_logs(self, factory, learning_mgr):
        # Seed learning log with a repeated successful sequence
        for i in range(3):
            t = time.time() + i * 2
            learning_mgr.record_execution("tool.pre_dispatch", {"tool": "load", "args": {}})
            learning_mgr.record_execution("tool.post_dispatch", {"tool": "load", "args": {}, "result": json.dumps({"status": "success"}), "timestamp": t})
            learning_mgr.record_execution("tool.pre_dispatch", {"tool": "analyze", "args": {}})
            learning_mgr.record_execution("tool.post_dispatch", {"tool": "analyze", "args": {}, "result": json.dumps({"status": "success"}), "timestamp": t + 1})

        factory.learning_manager = learning_mgr
        installed = factory.auto_generate_from_logs(top_n=3)
        assert len(installed) > 0
        assert installed[0]["auto"] is True


class TestLearningManagerStructuredLog:
    def test_record_execution_structured(self):
        lm = LearningManager()
        lm.record_execution("tool.post_dispatch", {
            "tool": "research_ttest",
            "args": {"group1": [1, 2, 3]},
            "result": json.dumps({"status": "success", "p": 0.02}),
        })
        log = lm.get_structured_log()
        assert len(log) == 1
        assert log[0]["tool_name"] == "research_ttest"
        assert log[0]["score"] > 0.5

    def test_record_execution_error_zero_score(self):
        lm = LearningManager()
        lm.record_execution("tool.error", {
            "tool": "research_did",
            "args": {},
            "error": "Missing column",
        })
        log = lm.get_structured_log()
        assert log[0]["score"] == 0.0
        assert log[0]["success"] is False

    def test_extract_sequences(self):
        lm = LearningManager()
        for i in range(3):
            base = i * 400
            lm.record_execution("tool.post_dispatch", {"tool": "a", "args": {}, "result": "ok", "timestamp": base + 1})
            lm.record_execution("tool.post_dispatch", {"tool": "b", "args": {}, "result": "ok", "timestamp": base + 2})
        seqs = lm._extract_sequences()
        assert len(seqs) > 0
        assert seqs[0]["tools"] == ["a", "b"]
        assert seqs[0]["count"] == 3

    def test_analyze_detects_workflow(self):
        lm = LearningManager()
        for i in range(4):
            base = i * 400
            lm.record_execution("tool.post_dispatch", {"tool": "load", "args": {}, "result": json.dumps({"status": "success"}), "timestamp": base + 1})
            lm.record_execution("tool.post_dispatch", {"tool": "did", "args": {}, "result": json.dumps({"status": "success", "apa": "..."}), "timestamp": base + 2})
        analysis = lm.analyze_execution()
        pattern_types = [p["type"] for p in analysis["patterns"]]
        assert "workflow_sequence" in pattern_types


class TestSkillExecutionHistory:
    def test_execute_records_history(self, factory, skill_mgr):
        reg = ToolRegistry()
        reg.register("ok_tool", "OK", {"type": "object", "properties": {}},
                     lambda args: json.dumps({"status": "success"}))

        sid = factory.create_skill(name="History Test", workflow=[{"tool": "ok_tool", "params": {}}])
        result = factory.execute_skill(sid, registry=reg)
        assert result["success"] is True

        history = skill_mgr.get_execution_history(sid)
        assert len(history) == 1
        assert history[0]["success"] is True
        assert history[0]["completed_steps"] == 1

    def test_execute_failure_recorded(self, factory, skill_mgr):
        reg = ToolRegistry()
        reg.register("bad_tool", "Bad", {"type": "object", "properties": {}},
                     lambda args: (_ for _ in ()).throw(RuntimeError("boom")))

        sid = factory.create_skill(name="Fail History", workflow=[{"tool": "bad_tool", "params": {}}])
        result = factory.execute_skill(sid, registry=reg)
        assert result["success"] is False

        history = skill_mgr.get_execution_history(sid)
        assert len(history) == 1
        assert history[0]["success"] is False
        assert "boom" in history[0]["error"]

    def test_skill_stats_updated(self, factory, skill_mgr):
        reg = ToolRegistry()
        reg.register("flip_tool", "Flip", {"type": "object", "properties": {}},
                     lambda args: json.dumps({"status": "success"}))

        sid = factory.create_skill(name="Stats Test", workflow=[{"tool": "flip_tool", "params": {}}])
        for _ in range(3):
            factory.execute_skill(sid, registry=reg)

        stats = skill_mgr.get_skill_stats(sid)
        assert stats["total_executions"] == 3
        assert stats["successful_executions"] == 3
        assert stats["failed_executions"] == 0
        assert stats["success_rate"] == 1.0


class TestSkillEvolution:
    def test_evolve_bumps_version(self, skill_mgr):
        sid = skill_mgr.install({
            "id": "evolve_test",
            "name": "Evolve Test",
            "version": "1.0",
            "tool_schemas": [],
            "handler_code": "def handle(args): return '{}'",
            "workflow": [{"tool": "t1", "params": {}}],
            "trigger": {"keywords": ["test"]},
        })
        ok = skill_mgr.evolve_skill(sid, new_workflow=[{"tool": "t2", "params": {}}])
        assert ok is True
        skill = skill_mgr.get_skill(sid)
        assert skill["version"] == "1.1"
        assert skill["workflow"][0]["tool"] == "t2"

    def test_auto_evolve_noop_insufficient(self, factory, skill_mgr):
        reg = ToolRegistry()
        reg.register("ok_tool", "OK", {"type": "object", "properties": {}},
                     lambda args: json.dumps({"status": "success"}))

        sid = factory.create_skill(name="Evolve Noop", workflow=[{"tool": "ok_tool", "params": {}}])
        # Only 2 executions, need 5 by default
        for _ in range(2):
            factory.execute_skill(sid, registry=reg)

        result = factory.auto_evolve(sid)
        assert result["action"] == "noop"

    def test_auto_evolve_removes_failing_step(self, factory, skill_mgr):
        reg = ToolRegistry()
        reg.register("ok_tool", "OK", {"type": "object", "properties": {}},
                     lambda args: json.dumps({"status": "success"}))
        reg.register("bad_tool", "Bad", {"type": "object", "properties": {}},
                     lambda args: (_ for _ in ()).throw(RuntimeError("boom")))

        sid = factory.create_skill(
            name="Evolve Remove",
            workflow=[
                {"tool": "ok_tool", "params": {}},
                {"tool": "bad_tool", "params": {}},
            ],
        )
        # 5 executions, all fail at step 1 (bad_tool)
        for _ in range(5):
            factory.execute_skill(sid, registry=reg)

        result = factory.auto_evolve(sid, min_executions=5, success_rate_threshold=0.7)
        assert result["action"] == "evolved"
        assert any("Removed failing steps" in c for c in result["changes"])

        skill = skill_mgr.get_skill(sid)
        # bad_tool step should have been removed
        assert len(skill["workflow"]) == 1
        assert skill["workflow"][0]["tool"] == "ok_tool"


class TestSkillAgentIntegration:
    def test_skill_tools_registered(self, tmp_path):
        """Verify that agent._register_skill_tools actually registers the 5 tools."""
        from sophia.config import Config
        from sophia.agent import SophiaAgent

        config = Config()
        config.session.workspace = str(tmp_path)
        config.session.db_path = str(tmp_path / "agent.db")
        config.model.name = "test"
        config.model.max_turns = 5
        config.context.max_messages = 10
        config.context.compress_threshold = 100
        config.guardrail.max_consecutive_calls = 10
        config.guardrail.max_calls_per_minute = 100

        agent = SophiaAgent(config)
        tools = agent.tools.list_tools()
        assert "skill_create" in tools
        assert "skill_execute" in tools
        assert "skill_list" in tools
        assert "skill_auto_discover" in tools
        assert "skill_evolve" in tools

    def test_skill_create_via_registry(self, tmp_path):
        from sophia.config import Config
        from sophia.agent import SophiaAgent

        config = Config()
        config.session.workspace = str(tmp_path)
        config.session.db_path = str(tmp_path / "agent.db")
        config.model.name = "test"
        config.model.max_turns = 5
        config.context.max_messages = 10
        config.context.compress_threshold = 100
        config.guardrail.max_consecutive_calls = 10
        config.guardrail.max_calls_per_minute = 100

        agent = SophiaAgent(config)
        result = json.loads(agent.tools.dispatch("skill_create", {
            "name": "Integration Test",
            "workflow": [{"tool": "research_ttest", "params": {}}],
        }))
        assert result["success"] is True
        assert "skill_id" in result

        list_result = json.loads(agent.tools.dispatch("skill_list", {}))
        assert list_result["count"] >= 1

    def test_skill_evolve_via_registry(self, tmp_path):
        from sophia.config import Config
        from sophia.agent import SophiaAgent

        config = Config()
        config.session.workspace = str(tmp_path)
        config.session.db_path = str(tmp_path / "agent.db")
        config.model.name = "test"
        config.model.max_turns = 5
        config.context.max_messages = 10
        config.context.compress_threshold = 100
        config.guardrail.max_consecutive_calls = 10
        config.guardrail.max_calls_per_minute = 100

        agent = SophiaAgent(config)
        # Create a skill
        create_res = json.loads(agent.tools.dispatch("skill_create", {
            "name": "Evolve Registry Test",
            "workflow": [{"tool": "research_ttest", "params": {}}],
        }))
        sid = create_res["skill_id"]

        # Execute it a few times to build history
        for _ in range(3):
            agent.tools.dispatch("skill_execute", {"skill_id": sid})

        # Evolve it
        evolve_res = json.loads(agent.tools.dispatch("skill_evolve", {
            "skill_id": sid,
            "min_executions": 2,
            "success_rate_threshold": 0.5,
        }))
        assert evolve_res["success"] is True

    def test_skill_evolve_real_version_bump(self, tmp_path):
        """Verify auto_evolve actually bumps version when executions succeed."""
        from sophia.config import Config
        from sophia.agent import SophiaAgent

        config = Config()
        config.session.workspace = str(tmp_path)
        config.session.db_path = str(tmp_path / "agent.db")
        config.model.name = "test"
        config.model.max_turns = 5
        config.context.max_messages = 10
        config.context.compress_threshold = 100
        config.guardrail.max_consecutive_calls = 10
        config.guardrail.max_calls_per_minute = 100

        agent = SophiaAgent(config)
        # Register a dummy tool that always succeeds
        agent.tools.register(
            "dummy_evolve_ok",
            "Always succeeds",
            {"type": "object", "properties": {}},
            lambda args: json.dumps({"status": "success", "value": 42}),
        )

        create_res = json.loads(agent.tools.dispatch("skill_create", {
            "name": "Real Evolve Test",
            "workflow": [{"tool": "dummy_evolve_ok", "params": {}}],
            "trigger": {"keywords": ["test"]},
        }))
        sid = create_res["skill_id"]

        # Execute 5 times successfully
        for _ in range(5):
            exec_res = json.loads(agent.tools.dispatch("skill_execute", {"skill_id": sid}))
            assert exec_res["success"] is True

        # Verify pre-evolve version
        skill_pre = agent.skill_factory.skill_manager.get_skill(sid)
        assert skill_pre["version"] == "1.0"

        # Evolve with low threshold so success triggers version bump
        evolve_res = json.loads(agent.tools.dispatch("skill_evolve", {
            "skill_id": sid,
            "min_executions": 3,
            "success_rate_threshold": 0.5,
        }))
        assert evolve_res["success"] is True
        assert evolve_res["action"] == "evolved"
        assert any("Bumped version" in c for c in evolve_res["changes"])

        # Verify post-evolve version
        skill_post = agent.skill_factory.skill_manager.get_skill(sid)
        assert skill_post["version"] == "1.1"
