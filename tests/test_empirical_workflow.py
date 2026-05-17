"""Tests for the Sophia empirical workflow orchestrator."""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from sophia.research.advisor import MethodologyAdvisor
from sophia.research.empirical_workflow import EmpiricalWorkflowEngine
from sophia.research.pipeline import ExperimentPipeline
from sophia.research.result_store import ResultStore
from sophia.research.register import register_method_tools
from sophia.tools.registry import ToolRegistry


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def sample_csv(workspace):
    path = os.path.join(workspace, "panel.csv")
    df = pd.DataFrame({
        "unit": [1, 1, 2, 2, 3, 3],
        "year": [2020, 2021, 2020, 2021, 2020, 2021],
        "treat": [0, 1, 0, 1, 0, 0],
        "y": [10.0, 13.0, 11.0, 14.0, 9.0, 9.5],
        "age": [30, 31, 40, 41, 35, 36],
    })
    df.to_csv(path, index=False)
    return "panel.csv"


@pytest.fixture
def engine(workspace):
    store = ResultStore(workspace)
    pipeline = ExperimentPipeline(workspace, store=store)
    return EmpiricalWorkflowEngine(
        workspace,
        store=store,
        pipeline=pipeline,
        advisor=MethodologyAdvisor(),
    )


def test_plan_without_data_reports_missing_inputs(engine):
    out = json.loads(engine.plan({
        "research_question": "Does training increase wages?",
        "outcome": "y",
        "treatment": "treat",
    }))
    assert out["ready_to_run"] is False
    assert "data_path_or_result_id" in out["missing_inputs"]
    assert [s["stage_id"] for s in out["stages"]] == [
        "scope",
        "data_contract",
        "data_quality",
        "descriptives",
        "diagnostics",
        "estimation",
        "robustness",
        "further_analysis",
        "reporting",
    ]


def test_plan_infers_ml_causal_mode(engine, sample_csv):
    out = json.loads(engine.plan({
        "research_question": "Use DML and causal forest to estimate CATE",
        "data_path": sample_csv,
        "outcome": "y",
        "treatment": "treat",
    }))
    assert out["mode"] == "ml_causal"
    assert out["ready_to_run"] is True


def test_run_executes_real_data_and_stores_result(engine, sample_csv):
    out = json.loads(engine.run({
        "research_question": "Does training increase wages?",
        "data_path": sample_csv,
        "outcome": "y",
        "treatment": "treat",
        "unit": "unit",
        "time": "year",
        "covariates": ["age"],
        "design": "quasi-experimental",
    }))
    assert out["executed"] is True
    assert out["stage_outputs"]["data_contract"]["rows"] == 6
    assert out["stage_outputs"]["estimation"]["status"] == "completed"
    assert out["stage_outputs"]["estimation"]["models"][0]["focal_effect"]["variable"] == "treat"
    assert out["result_id"].startswith("res_")


def test_run_missing_columns_does_not_fake_estimation(engine, sample_csv):
    out = json.loads(engine.run({
        "research_question": "Does training increase wages?",
        "data_path": sample_csv,
        "outcome": "missing_y",
        "treatment": "treat",
    }))
    assert out["executed"] is True
    assert out["stage_outputs"]["estimation"]["status"] == "skipped"
    assert "missing_y" in out["stage_outputs"]["estimation"]["missing_columns"]


def test_capability_audit_lists_optional_packages(engine):
    out = json.loads(engine.capability_audit({}))
    capabilities = {c["capability"] for c in out["capabilities"]}
    assert "explicit_8_step_empirical_pipeline" in capabilities
    assert "pyfixest_hdfe_and_event_study" in capabilities
    assert isinstance(out["missing_optional_packages"], list)


def test_tools_are_registered(workspace):
    registry = ToolRegistry()
    store = ResultStore(workspace)
    pipeline = ExperimentPipeline(workspace, store=store)
    empirical = EmpiricalWorkflowEngine(
        workspace,
        store=store,
        pipeline=pipeline,
        advisor=MethodologyAdvisor(),
    )
    register_method_tools(registry, {
        "pipeline": pipeline,
        "advisor": MethodologyAdvisor(),
        "empirical_workflow": empirical,
    })
    tools = registry.list_tools()
    assert "empirical_workflow_plan" in tools
    assert "empirical_workflow_run" in tools
    assert "empirical_capability_audit" in tools
