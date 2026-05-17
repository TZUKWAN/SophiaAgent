"""Experiment system for SophiaAgent.

Full experiment lifecycle: design → run → track → compare → reproduce.
Supports hypothesis-driven research with parameter tracking, result versioning,
and statistical comparison.
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager
from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

EXPERIMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'designed',
    hypothesis TEXT DEFAULT '',
    method TEXT DEFAULT '',
    parameters TEXT DEFAULT '{}',
    variables TEXT DEFAULT '{}',
    tags TEXT DEFAULT '[]',
    goal_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    run_number INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending',
    code TEXT DEFAULT '',
    parameters_override TEXT DEFAULT '{}',
    metrics TEXT DEFAULT '{}',
    artifacts TEXT DEFAULT '[]',
    logs TEXT DEFAULT '',
    started_at TEXT,
    finished_at TEXT,
    duration_seconds REAL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_experiments_session ON experiments(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON experiment_runs(experiment_id);
"""


@dataclass
class Experiment:
    id: str
    session_id: str
    name: str
    description: str
    status: str              # designed | running | completed | failed | cancelled
    hypothesis: str
    method: str
    parameters: Dict
    variables: Dict          # {independent: [...], dependent: [...], control: [...]}
    tags: List[str]
    goal_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class ExperimentRun:
    id: str
    experiment_id: str
    run_number: int
    status: str              # pending | running | completed | failed
    code: str
    parameters_override: Dict
    metrics: Dict            # {metric_name: value}
    artifacts: List[str]     # file paths to outputs
    logs: str
    started_at: Optional[str]
    finished_at: Optional[str]
    duration_seconds: float
    error: Optional[str]
    created_at: str


VALID_EXPERIMENT_STATUSES = ["designed", "running", "completed", "failed", "cancelled"]
VALID_RUN_STATUSES = ["pending", "running", "completed", "failed"]


class ExperimentManager:
    def __init__(self, db_path: str, hooks: HookManager = None):
        self.db_path = db_path
        self.hooks = hooks
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(EXPERIMENTS_SCHEMA)

    # ── Experiment CRUD ───────────────────────────────────────

    def create_experiment(
        self,
        session_id: str,
        name: str,
        description: str = "",
        hypothesis: str = "",
        method: str = "",
        parameters: Optional[Dict] = None,
        variables: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        goal_id: Optional[str] = None,
    ) -> Experiment:
        exp_id = uuid.uuid4().hex[:8]
        params_json = json.dumps(parameters or {}, ensure_ascii=False)
        vars_json = json.dumps(variables or {}, ensure_ascii=False)
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO experiments "
                "(id, session_id, name, description, hypothesis, method, "
                "parameters, variables, tags, goal_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (exp_id, session_id, name, description, hypothesis, method,
                 params_json, vars_json, tags_json, goal_id),
            )
        return self.get_experiment(exp_id)

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
        if not row:
            return None
        return self._row_to_experiment(row)

    def update_experiment(self, exp_id: str, **kwargs) -> Optional[Experiment]:
        allowed = {"name", "description", "status", "hypothesis", "method",
                   "parameters", "variables", "tags"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return self.get_experiment(exp_id)
        if "parameters" in updates:
            updates["parameters"] = json.dumps(updates["parameters"], ensure_ascii=False)
        if "variables" in updates:
            updates["variables"] = json.dumps(updates["variables"], ensure_ascii=False)
        if "tags" in updates:
            updates["tags"] = json.dumps(updates["tags"], ensure_ascii=False)
        updates["updated_at"] = "datetime('now')"
        set_parts = [f"{k}=?" for k in updates]
        values = list(updates.values()) + [exp_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE experiments SET {', '.join(set_parts)} WHERE id=?", values
            )
        return self.get_experiment(exp_id)

    def list_experiments(self, session_id: str, status: Optional[str] = None) -> List[Experiment]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE session_id=? AND status=? ORDER BY created_at DESC",
                    (session_id, status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE session_id=? ORDER BY created_at DESC",
                    (session_id,),
                ).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def delete_experiment(self, exp_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
            return cursor.rowcount > 0

    # ── Experiment Runs ───────────────────────────────────────

    def create_run(
        self,
        experiment_id: str,
        code: str = "",
        parameters_override: Optional[Dict] = None,
    ) -> ExperimentRun:
        run_id = uuid.uuid4().hex[:8]
        exp = self.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Auto-increment run_number
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(run_number) as max_run FROM experiment_runs WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
            run_number = (row["max_run"] or 0) + 1

            params_json = json.dumps(parameters_override or {}, ensure_ascii=False)
            conn.execute(
                "INSERT INTO experiment_runs "
                "(id, experiment_id, run_number, code, parameters_override) "
                "VALUES (?,?,?,?,?)",
                (run_id, experiment_id, run_number, code, params_json),
            )

        # Update experiment status
        self.update_experiment(experiment_id, status="running")
        return self.get_run(run_id)

    def start_run(self, run_id: str) -> Optional[ExperimentRun]:
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiment_runs SET status='running', "
                "started_at=datetime('now') WHERE id=?",
                (run_id,),
            )
        return self.get_run(run_id)

    def complete_run(self, run_id: str, metrics: Optional[Dict] = None,
                     artifacts: Optional[List[str]] = None, logs: str = "") -> Optional[ExperimentRun]:
        run = self.get_run(run_id)
        if not run:
            return None

        metrics_json = json.dumps(metrics or {}, ensure_ascii=False)
        artifacts_json = json.dumps(artifacts or [], ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                "UPDATE experiment_runs SET status='completed', metrics=?, "
                "artifacts=?, logs=?, finished_at=datetime('now') WHERE id=?",
                (metrics_json, artifacts_json, logs, run_id),
            )
            # Calculate duration
            conn.execute(
                "UPDATE experiment_runs SET duration_seconds = "
                "(julianday(finished_at) - julianday(started_at)) * 86400 "
                "WHERE id=? AND started_at IS NOT NULL",
                (run_id,),
            )

        # Update experiment status
        self.update_experiment(run.experiment_id, status="completed")
        return self.get_run(run_id)

    def fail_run(self, run_id: str, error: str = "") -> Optional[ExperimentRun]:
        run = self.get_run(run_id)
        if not run:
            return None

        with self._connect() as conn:
            conn.execute(
                "UPDATE experiment_runs SET status='failed', error=?, "
                "finished_at=datetime('now') WHERE id=?",
                (error, run_id),
            )

        self.update_experiment(run.experiment_id, status="failed")
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM experiment_runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return None
        return self._row_to_run(row)

    def list_runs(self, experiment_id: str) -> List[ExperimentRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiment_runs WHERE experiment_id=? ORDER BY run_number",
                (experiment_id,),
            ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def get_latest_run(self, experiment_id: str) -> Optional[ExperimentRun]:
        runs = self.list_runs(experiment_id)
        return runs[-1] if runs else None

    # ── Comparison & Analysis ─────────────────────────────────

    def compare_runs(self, experiment_id: str) -> Dict:
        runs = self.list_runs(experiment_id)
        completed = [r for r in runs if r.status == "completed"]
        if len(completed) < 2:
            return {"error": "Need at least 2 completed runs to compare"}

        all_metrics = set()
        for r in completed:
            all_metrics.update(r.metrics.keys())

        comparison = {}
        for metric in sorted(all_metrics):
            values = []
            for r in completed:
                if metric in r.metrics:
                    values.append({
                        "run": r.run_number,
                        "value": r.metrics[metric],
                    })
            if len(values) >= 2:
                nums = [v["value"] for v in values if isinstance(v["value"], (int, float))]
                if nums:
                    comparison[metric] = {
                        "values": values,
                        "min": min(nums),
                        "max": max(nums),
                        "mean": sum(nums) / len(nums),
                        "range": max(nums) - min(nums),
                    }

        return {
            "experiment_id": experiment_id,
            "runs_compared": len(completed),
            "metrics": comparison,
        }

    def get_experiment_summary(self, experiment_id: str) -> Dict:
        exp = self.get_experiment(experiment_id)
        if not exp:
            return {"error": "Experiment not found"}
        runs = self.list_runs(experiment_id)
        completed = [r for r in runs if r.status == "completed"]
        failed = [r for r in runs if r.status == "failed"]
        return {
            "id": exp.id,
            "name": exp.name,
            "status": exp.status,
            "hypothesis": exp.hypothesis,
            "method": exp.method,
            "parameters": exp.parameters,
            "variables": exp.variables,
            "total_runs": len(runs),
            "completed_runs": len(completed),
            "failed_runs": len(failed),
            "runs": [{
                "run_number": r.run_number,
                "status": r.status,
                "metrics": r.metrics,
                "duration_seconds": round(r.duration_seconds, 2),
            } for r in runs],
        }

    def execute_run_code(self, run_id: str, sandbox_fn) -> ExperimentRun:
        """Execute a run's code using the provided sandbox function.

        Args:
            run_id: The run to execute.
            sandbox_fn: Callable(code: str, params: dict) -> (success: bool, result: str, metrics: dict)

        Returns:
            Updated ExperimentRun.
        """
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")

        self.start_run(run_id)
        exp = self.get_experiment(run.experiment_id)

        # Merge parameters: experiment defaults + run overrides
        merged_params = {**exp.parameters, **run.parameters_override}

        try:
            success, logs, metrics = sandbox_fn(run.code, merged_params)
            if success:
                return self.complete_run(run_id, metrics=metrics, logs=logs)
            else:
                return self.fail_run(run_id, error=logs)
        except Exception as e:
            return self.fail_run(run_id, error=str(e))

    # ── Row Converters ────────────────────────────────────────

    def _row_to_experiment(self, row) -> Experiment:
        return Experiment(
            id=row["id"],
            session_id=row["session_id"],
            name=row["name"],
            description=row["description"],
            status=row["status"],
            hypothesis=row["hypothesis"],
            method=row["method"],
            parameters=json.loads(row["parameters"]),
            variables=json.loads(row["variables"]),
            tags=json.loads(row["tags"]),
            goal_id=row["goal_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_run(self, row) -> ExperimentRun:
        return ExperimentRun(
            id=row["id"],
            experiment_id=row["experiment_id"],
            run_number=row["run_number"],
            status=row["status"],
            code=row["code"],
            parameters_override=json.loads(row["parameters_override"]),
            metrics=json.loads(row["metrics"]),
            artifacts=json.loads(row["artifacts"]),
            logs=row["logs"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            duration_seconds=row["duration_seconds"],
            error=row["error"],
            created_at=row["created_at"],
        )


# ── Tool Registration ─────────────────────────────────────────

def register_experiment_tools(registry: ToolRegistry, exp_mgr: ExperimentManager):
    """Register experiment tools into the tool registry."""

    def _exp_create(args):
        exp = exp_mgr.create_experiment(
            session_id=args.get("session_id", "default"),
            name=args["name"],
            description=args.get("description", ""),
            hypothesis=args.get("hypothesis", ""),
            method=args.get("method", ""),
            parameters=args.get("parameters"),
            variables=args.get("variables"),
            tags=args.get("tags"),
            goal_id=args.get("goal_id"),
        )
        return json.dumps({
            "id": exp.id, "name": exp.name, "status": exp.status,
            "hypothesis": exp.hypothesis,
        }, ensure_ascii=False)

    registry.register(
        "experiment_create",
        "Create a new experiment with hypothesis, method, parameters, and variables",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "name": {"type": "string", "description": "Experiment name"},
                "description": {"type": "string"},
                "hypothesis": {"type": "string", "description": "Research hypothesis to test"},
                "method": {"type": "string", "description": "Experimental method description"},
                "parameters": {
                    "type": "object",
                    "description": "Default parameters for all runs, e.g. {\"alpha\": 0.05, \"samples\": 1000}",
                },
                "variables": {
                    "type": "object",
                    "description": "Variable definitions: {\"independent\": [...], \"dependent\": [...], \"control\": [...]}",
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "goal_id": {"type": "string"},
            },
            "required": ["name"],
        },
        _exp_create,
    )

    def _exp_get(args):
        exp = exp_mgr.get_experiment(args["experiment_id"])
        if not exp:
            return json.dumps({"error": "Experiment not found"})
        return json.dumps({
            "id": exp.id, "name": exp.name, "status": exp.status,
            "hypothesis": exp.hypothesis, "method": exp.method,
            "parameters": exp.parameters, "variables": exp.variables,
        }, ensure_ascii=False)

    registry.register(
        "experiment_get",
        "Get experiment details",
        {"type": "object", "properties": {
            "experiment_id": {"type": "string"},
        }, "required": ["experiment_id"]},
        _exp_get,
    )

    def _exp_list(args):
        exps = exp_mgr.list_experiments(
            args.get("session_id", "default"),
            status=args.get("status"),
        )
        return json.dumps([{
            "id": e.id, "name": e.name, "status": e.status,
            "hypothesis": e.hypothesis[:100],
        } for e in exps], ensure_ascii=False)

    registry.register(
        "experiment_list",
        "List experiments for a session",
        {"type": "object", "properties": {
            "session_id": {"type": "string"},
            "status": {"type": "string"},
        }},
        _exp_list,
    )

    def _exp_run_create(args):
        run = exp_mgr.create_run(
            experiment_id=args["experiment_id"],
            code=args.get("code", ""),
            parameters_override=args.get("parameters"),
        )
        return json.dumps({
            "run_id": run.id, "run_number": run.run_number,
            "status": run.status,
        }, ensure_ascii=False)

    registry.register(
        "experiment_run_create",
        "Create a new run for an experiment with optional code and parameter overrides",
        {
            "type": "object",
            "properties": {
                "experiment_id": {"type": "string"},
                "code": {"type": "string", "description": "Python code to execute for this run"},
                "parameters": {
                    "type": "object",
                    "description": "Override experiment default parameters for this run",
                },
            },
            "required": ["experiment_id"],
        },
        _exp_run_create,
    )

    def _exp_run_complete(args):
        run = exp_mgr.complete_run(
            run_id=args["run_id"],
            metrics=args.get("metrics"),
            artifacts=args.get("artifacts"),
            logs=args.get("logs", ""),
        )
        if not run:
            return json.dumps({"error": "Run not found"})
        return json.dumps({
            "run_id": run.id, "status": run.status,
            "metrics": run.metrics, "duration": run.duration_seconds,
        }, ensure_ascii=False)

    registry.register(
        "experiment_run_complete",
        "Mark a run as completed with metrics and artifacts",
        {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "metrics": {
                    "type": "object",
                    "description": "Measured results, e.g. {\"accuracy\": 0.95, \"f1\": 0.92}",
                },
                "artifacts": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Paths to output files",
                },
                "logs": {"type": "string"},
            },
            "required": ["run_id"],
        },
        _exp_run_complete,
    )

    def _exp_compare(args):
        result = exp_mgr.compare_runs(args["experiment_id"])
        return json.dumps(result, ensure_ascii=False)

    registry.register(
        "experiment_compare",
        "Compare metrics across multiple runs of an experiment",
        {"type": "object", "properties": {
            "experiment_id": {"type": "string"},
        }, "required": ["experiment_id"]},
        _exp_compare,
    )

    def _exp_summary(args):
        result = exp_mgr.get_experiment_summary(args["experiment_id"])
        return json.dumps(result, ensure_ascii=False, default=str)

    registry.register(
        "experiment_summary",
        "Get full summary of an experiment including all runs",
        {"type": "object", "properties": {
            "experiment_id": {"type": "string"},
        }, "required": ["experiment_id"]},
        _exp_summary,
    )
