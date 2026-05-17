"""Tests for ExperimentManager: experiment lifecycle, runs, comparison."""
import json

import pytest

from sophia.experiment import (
    EXPERIMENTS_SCHEMA,
    Experiment,
    ExperimentManager,
    ExperimentRun,
    register_experiment_tools,
)
from sophia.hooks import HookManager
from sophia.tools.registry import ToolRegistry


@pytest.fixture
def exp_mgr(tmp_path):
    db_path = str(tmp_path / "test_exp.db")
    return ExperimentManager(db_path)


@pytest.fixture
def exp_mgr_with_hooks(tmp_path):
    db_path = str(tmp_path / "test_exp.db")
    hooks = HookManager()
    return ExperimentManager(db_path, hooks=hooks), hooks


class TestExperimentCreate:
    def test_create_basic(self, exp_mgr):
        exp = exp_mgr.create_experiment(
            session_id="s1", name="Test Experiment",
        )
        assert exp.id
        assert exp.name == "Test Experiment"
        assert exp.status == "designed"
        assert exp.session_id == "s1"

    def test_create_with_all_fields(self, exp_mgr):
        exp = exp_mgr.create_experiment(
            session_id="s1",
            name="Full Experiment",
            description="A detailed experiment",
            hypothesis="H1 predicts X",
            method="Randomized controlled trial",
            parameters={"alpha": 0.05, "samples": 1000},
            variables={
                "independent": ["treatment_dose"],
                "dependent": ["recovery_rate"],
                "control": ["age_group"],
            },
            tags=["clinical", "phase3"],
            goal_id="g123",
        )
        assert exp.hypothesis == "H1 predicts X"
        assert exp.method == "Randomized controlled trial"
        assert exp.parameters == {"alpha": 0.05, "samples": 1000}
        assert exp.variables["independent"] == ["treatment_dose"]
        assert exp.variables["dependent"] == ["recovery_rate"]
        assert exp.variables["control"] == ["age_group"]
        assert exp.tags == ["clinical", "phase3"]
        assert exp.goal_id == "g123"

    def test_create_default_parameters(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Defaults")
        assert exp.parameters == {}
        assert exp.variables == {}
        assert exp.tags == []
        assert exp.description == ""
        assert exp.hypothesis == ""


class TestExperimentGet:
    def test_get_existing(self, exp_mgr):
        created = exp_mgr.create_experiment(session_id="s1", name="Find Me")
        found = exp_mgr.get_experiment(created.id)
        assert found is not None
        assert found.name == "Find Me"

    def test_get_nonexistent(self, exp_mgr):
        assert exp_mgr.get_experiment("nope") is None


class TestExperimentUpdate:
    def test_update_name(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Old")
        updated = exp_mgr.update_experiment(exp.id, name="New")
        assert updated.name == "New"

    def test_update_status(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        updated = exp_mgr.update_experiment(exp.id, status="running")
        assert updated.status == "running"

    def test_update_parameters(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        updated = exp_mgr.update_experiment(
            exp.id, parameters={"lr": 0.001, "epochs": 50}
        )
        assert updated.parameters == {"lr": 0.001, "epochs": 50}

    def test_update_tags(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        updated = exp_mgr.update_experiment(exp.id, tags=["ml", "nlp"])
        assert updated.tags == ["ml", "nlp"]

    def test_update_no_allowed_fields(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        updated = exp_mgr.update_experiment(exp.id, fake_field="ignore")
        assert updated.name == "Exp"

    def test_update_nonexistent(self, exp_mgr):
        result = exp_mgr.update_experiment("nope", name="X")
        assert result is None


class TestExperimentList:
    def test_list_by_session(self, exp_mgr):
        exp_mgr.create_experiment(session_id="s1", name="A")
        exp_mgr.create_experiment(session_id="s1", name="B")
        exp_mgr.create_experiment(session_id="s2", name="C")
        exps = exp_mgr.list_experiments("s1")
        assert len(exps) == 2

    def test_list_with_status_filter(self, exp_mgr):
        exp_mgr.create_experiment(session_id="s1", name="A")
        exp = exp_mgr.create_experiment(session_id="s1", name="B")
        exp_mgr.update_experiment(exp.id, status="completed")
        designed = exp_mgr.list_experiments("s1", status="designed")
        assert len(designed) == 1
        assert designed[0].name == "A"

    def test_list_empty_session(self, exp_mgr):
        assert exp_mgr.list_experiments("empty") == []


class TestExperimentDelete:
    def test_delete_existing(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Bye")
        assert exp_mgr.delete_experiment(exp.id) is True
        assert exp_mgr.get_experiment(exp.id) is None

    def test_delete_nonexistent(self, exp_mgr):
        assert exp_mgr.delete_experiment("nope") is False

    def test_delete_cascades_runs(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Cascade")
        run = exp_mgr.create_run(exp.id, code="print(1)")
        assert exp_mgr.delete_experiment(exp.id) is True
        assert exp_mgr.get_run(run.id) is None


class TestRunCreate:
    def test_create_basic_run(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        run = exp_mgr.create_run(exp.id)
        assert run.id
        assert run.run_number == 1
        assert run.status == "pending"
        assert run.experiment_id == exp.id

    def test_create_run_with_code_and_params(self, exp_mgr):
        exp = exp_mgr.create_experiment(
            session_id="s1", name="Exp",
            parameters={"lr": 0.01},
        )
        run = exp_mgr.create_run(
            exp.id,
            code="model.train()",
            parameters_override={"lr": 0.001},
        )
        assert run.code == "model.train()"
        assert run.parameters_override == {"lr": 0.001}

    def test_create_run_auto_increments(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        r1 = exp_mgr.create_run(exp.id)
        r2 = exp_mgr.create_run(exp.id)
        r3 = exp_mgr.create_run(exp.id)
        assert r1.run_number == 1
        assert r2.run_number == 2
        assert r3.run_number == 3

    def test_create_run_updates_experiment_status(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        exp_mgr.create_run(exp.id)
        updated = exp_mgr.get_experiment(exp.id)
        assert updated.status == "running"

    def test_create_run_nonexistent_experiment(self, exp_mgr):
        with pytest.raises(ValueError, match="not found"):
            exp_mgr.create_run("nope")


class TestRunLifecycle:
    def test_start_run(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        run = exp_mgr.create_run(exp.id)
        started = exp_mgr.start_run(run.id)
        assert started.status == "running"
        assert started.started_at is not None

    def test_complete_run(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        run = exp_mgr.create_run(exp.id)
        exp_mgr.start_run(run.id)
        completed = exp_mgr.complete_run(
            run.id,
            metrics={"accuracy": 0.95, "f1": 0.92},
            artifacts=["/output/model.pkl"],
            logs="Training complete",
        )
        assert completed.status == "completed"
        assert completed.metrics == {"accuracy": 0.95, "f1": 0.92}
        assert completed.artifacts == ["/output/model.pkl"]
        assert completed.logs == "Training complete"
        assert completed.finished_at is not None
        assert completed.duration_seconds >= 0

    def test_complete_updates_experiment_status(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        run = exp_mgr.create_run(exp.id)
        exp_mgr.start_run(run.id)
        exp_mgr.complete_run(run.id, metrics={"acc": 0.9})
        updated = exp_mgr.get_experiment(exp.id)
        assert updated.status == "completed"

    def test_fail_run(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        run = exp_mgr.create_run(exp.id)
        exp_mgr.start_run(run.id)
        failed = exp_mgr.fail_run(run.id, error="OOM")
        assert failed.status == "failed"
        assert failed.error == "OOM"

    def test_fail_updates_experiment_status(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        run = exp_mgr.create_run(exp.id)
        exp_mgr.start_run(run.id)
        exp_mgr.fail_run(run.id, error="crash")
        updated = exp_mgr.get_experiment(exp.id)
        assert updated.status == "failed"

    def test_complete_nonexistent_run(self, exp_mgr):
        assert exp_mgr.complete_run("nope") is None

    def test_fail_nonexistent_run(self, exp_mgr):
        assert exp_mgr.fail_run("nope") is None


class TestRunList:
    def test_list_runs(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        exp_mgr.create_run(exp.id)
        exp_mgr.create_run(exp.id)
        runs = exp_mgr.list_runs(exp.id)
        assert len(runs) == 2
        assert runs[0].run_number == 1
        assert runs[1].run_number == 2

    def test_get_latest_run(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        exp_mgr.create_run(exp.id)
        exp_mgr.create_run(exp.id)
        latest = exp_mgr.get_latest_run(exp.id)
        assert latest.run_number == 2

    def test_get_latest_run_empty(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exp")
        assert exp_mgr.get_latest_run(exp.id) is None


class TestCompareRuns:
    def test_compare_two_runs(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Compare")
        r1 = exp_mgr.create_run(exp.id)
        r2 = exp_mgr.create_run(exp.id)

        exp_mgr.start_run(r1.id)
        exp_mgr.complete_run(r1.id, metrics={"accuracy": 0.85, "loss": 0.4})

        exp_mgr.start_run(r2.id)
        exp_mgr.complete_run(r2.id, metrics={"accuracy": 0.92, "loss": 0.25})

        result = exp_mgr.compare_runs(exp.id)
        assert result["runs_compared"] == 2
        assert "accuracy" in result["metrics"]
        assert result["metrics"]["accuracy"]["min"] == 0.85
        assert result["metrics"]["accuracy"]["max"] == 0.92
        assert abs(result["metrics"]["accuracy"]["mean"] - 0.885) < 0.001
        assert abs(result["metrics"]["accuracy"]["range"] - 0.07) < 0.001
        assert "loss" in result["metrics"]

    def test_compare_insufficient_runs(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Compare")
        r1 = exp_mgr.create_run(exp.id)
        exp_mgr.start_run(r1.id)
        exp_mgr.complete_run(r1.id, metrics={"acc": 0.9})

        result = exp_mgr.compare_runs(exp.id)
        assert "error" in result

    def test_compare_ignores_failed(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Compare")
        r1 = exp_mgr.create_run(exp.id)
        r2 = exp_mgr.create_run(exp.id)
        r3 = exp_mgr.create_run(exp.id)

        exp_mgr.start_run(r1.id)
        exp_mgr.complete_run(r1.id, metrics={"acc": 0.8})

        exp_mgr.start_run(r2.id)
        exp_mgr.fail_run(r2.id, error="crash")

        exp_mgr.start_run(r3.id)
        exp_mgr.complete_run(r3.id, metrics={"acc": 0.9})

        result = exp_mgr.compare_runs(exp.id)
        assert result["runs_compared"] == 2


class TestExperimentSummary:
    def test_summary(self, exp_mgr):
        exp = exp_mgr.create_experiment(
            session_id="s1",
            name="Summary Test",
            hypothesis="H1",
            method="cross-validation",
            parameters={"k": 5},
        )
        r1 = exp_mgr.create_run(exp.id)
        r2 = exp_mgr.create_run(exp.id)

        exp_mgr.start_run(r1.id)
        exp_mgr.complete_run(r1.id, metrics={"acc": 0.9})

        exp_mgr.start_run(r2.id)
        exp_mgr.fail_run(r2.id, error="timeout")

        summary = exp_mgr.get_experiment_summary(exp.id)
        assert summary["name"] == "Summary Test"
        assert summary["total_runs"] == 2
        assert summary["completed_runs"] == 1
        assert summary["failed_runs"] == 1
        assert summary["hypothesis"] == "H1"
        assert summary["parameters"] == {"k": 5}

    def test_summary_nonexistent(self, exp_mgr):
        result = exp_mgr.get_experiment_summary("nope")
        assert "error" in result


class TestExecuteRunCode:
    def test_execute_success(self, exp_mgr):
        exp = exp_mgr.create_experiment(
            session_id="s1", name="Exec",
            parameters={"x": 1},
        )
        run = exp_mgr.create_run(
            exp.id, code="result = x + 1",
            parameters_override={"x": 2},
        )

        def sandbox(code, params):
            return (True, "ok", {"result": params["x"] + 1})

        result = exp_mgr.execute_run_code(run.id, sandbox)
        assert result.status == "completed"
        assert result.metrics == {"result": 3}

    def test_execute_failure(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exec")
        run = exp_mgr.create_run(exp.id, code="bad code")

        def sandbox(code, params):
            return (False, "SyntaxError: invalid syntax", {})

        result = exp_mgr.execute_run_code(run.id, sandbox)
        assert result.status == "failed"
        assert "SyntaxError" in result.error

    def test_execute_exception(self, exp_mgr):
        exp = exp_mgr.create_experiment(session_id="s1", name="Exec")
        run = exp_mgr.create_run(exp.id, code="crash")

        def sandbox(code, params):
            raise RuntimeError("sandbox exploded")

        result = exp_mgr.execute_run_code(run.id, sandbox)
        assert result.status == "failed"
        assert "sandbox exploded" in result.error

    def test_execute_nonexistent_run(self, exp_mgr):
        with pytest.raises(ValueError, match="not found"):
            exp_mgr.execute_run_code("nope", lambda c, p: (True, "", {}))

    def test_execute_merges_parameters(self, exp_mgr):
        exp = exp_mgr.create_experiment(
            session_id="s1", name="Merge",
            parameters={"a": 1, "b": 2},
        )
        run = exp_mgr.create_run(
            exp.id, parameters_override={"b": 99, "c": 3},
        )

        received_params = {}

        def sandbox(code, params):
            received_params.update(params)
            return (True, "ok", {})

        exp_mgr.execute_run_code(run.id, sandbox)
        assert received_params == {"a": 1, "b": 99, "c": 3}


class TestExperimentTools:
    def _make_registry(self, exp_mgr):
        reg = ToolRegistry()
        register_experiment_tools(reg, exp_mgr)
        return reg

    def test_experiment_create_tool(self, exp_mgr):
        reg = self._make_registry(exp_mgr)
        result = json.loads(reg.dispatch("experiment_create", {
            "session_id": "s1",
            "name": "Tool Test",
            "hypothesis": "H1",
            "parameters": {"alpha": 0.05},
        }))
        assert result["name"] == "Tool Test"
        assert result["status"] == "designed"

    def test_experiment_get_tool(self, exp_mgr):
        reg = self._make_registry(exp_mgr)
        created = json.loads(reg.dispatch("experiment_create", {
            "session_id": "s1", "name": "Get Test",
        }))
        result = json.loads(reg.dispatch("experiment_get", {
            "experiment_id": created["id"],
        }))
        assert result["name"] == "Get Test"

    def test_experiment_get_not_found(self, exp_mgr):
        reg = self._make_registry(exp_mgr)
        result = json.loads(reg.dispatch("experiment_get", {
            "experiment_id": "nope",
        }))
        assert "error" in result

    def test_experiment_list_tool(self, exp_mgr):
        reg = self._make_registry(exp_mgr)
        reg.dispatch("experiment_create", {
            "session_id": "s1", "name": "A",
        })
        reg.dispatch("experiment_create", {
            "session_id": "s1", "name": "B",
        })
        result = json.loads(reg.dispatch("experiment_list", {
            "session_id": "s1",
        }))
        assert len(result) == 2

    def test_experiment_run_create_tool(self, exp_mgr):
        reg = self._make_registry(exp_mgr)
        created = json.loads(reg.dispatch("experiment_create", {
            "session_id": "s1", "name": "Run Test",
        }))
        result = json.loads(reg.dispatch("experiment_run_create", {
            "experiment_id": created["id"],
            "code": "print(1)",
            "parameters": {"override": True},
        }))
        assert result["run_number"] == 1
        assert result["status"] == "pending"

    def test_experiment_run_complete_tool(self, exp_mgr):
        reg = self._make_registry(exp_mgr)
        created = json.loads(reg.dispatch("experiment_create", {
            "session_id": "s1", "name": "Complete Test",
        }))
        run = json.loads(reg.dispatch("experiment_run_create", {
            "experiment_id": created["id"],
        }))
        exp_mgr.start_run(run["run_id"])
        result = json.loads(reg.dispatch("experiment_run_complete", {
            "run_id": run["run_id"],
            "metrics": {"accuracy": 0.93},
            "artifacts": ["/out/model.pkl"],
        }))
        assert result["status"] == "completed"
        assert result["metrics"]["accuracy"] == 0.93

    def test_experiment_compare_tool(self, exp_mgr):
        reg = self._make_registry(exp_mgr)
        created = json.loads(reg.dispatch("experiment_create", {
            "session_id": "s1", "name": "Compare Tool",
        }))
        eid = created["id"]
        r1 = exp_mgr.create_run(eid)
        r2 = exp_mgr.create_run(eid)
        exp_mgr.start_run(r1.id)
        exp_mgr.complete_run(r1.id, metrics={"acc": 0.8})
        exp_mgr.start_run(r2.id)
        exp_mgr.complete_run(r2.id, metrics={"acc": 0.9})

        result = json.loads(reg.dispatch("experiment_compare", {
            "experiment_id": eid,
        }))
        assert result["runs_compared"] == 2

    def test_experiment_summary_tool(self, exp_mgr):
        reg = self._make_registry(exp_mgr)
        created = json.loads(reg.dispatch("experiment_create", {
            "session_id": "s1", "name": "Summary Tool",
        }))
        result = json.loads(reg.dispatch("experiment_summary", {
            "experiment_id": created["id"],
        }))
        assert result["name"] == "Summary Tool"
        assert result["total_runs"] == 0

    def test_all_tools_registered(self, exp_mgr):
        reg = self._make_registry(exp_mgr)
        tools = reg.list_tools()
        expected = [
            "experiment_create", "experiment_get", "experiment_list",
            "experiment_run_create", "experiment_run_complete",
            "experiment_compare", "experiment_summary",
        ]
        for t in expected:
            assert t in tools, f"Missing tool: {t}"
