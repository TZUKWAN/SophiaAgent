"""Skill management for SophiaAgent."""
import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from sophia.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SKILLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0',
    description TEXT DEFAULT '',
    author TEXT DEFAULT '',
    category TEXT DEFAULT 'general',
    tool_schemas TEXT NOT NULL,
    handler_code TEXT NOT NULL,
    workflow TEXT DEFAULT '[]',
    trigger TEXT DEFAULT '{}',
    success_rate REAL DEFAULT 0.0,
    execution_count INTEGER DEFAULT 0,
    avg_score REAL DEFAULT 0.0,
    auto_generated INTEGER DEFAULT 0,
    installed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

EXECUTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 0,
    error TEXT DEFAULT '',
    completed_steps INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    elapsed_seconds REAL DEFAULT 0.0,
    context TEXT DEFAULT '{}',
    step_results TEXT DEFAULT '[]',
    executed_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_exec_skill ON skill_executions(skill_id);
CREATE INDEX IF NOT EXISTS idx_exec_time ON skill_executions(executed_at);
"""


class SkillManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(SKILLS_SCHEMA)
            conn.executescript(EXECUTIONS_SCHEMA)

    def install(self, skill_def: Dict) -> str:
        skill_id = skill_def.get("id", skill_def["name"].lower().replace(" ", "_"))
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO skills (id, name, version, description, author, "
                "category, tool_schemas, handler_code, workflow, trigger, success_rate, "
                "execution_count, avg_score, auto_generated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (skill_id, skill_def["name"], skill_def.get("version", "1.0"),
                 skill_def.get("description", ""), skill_def.get("author", ""),
                 skill_def.get("category", "general"),
                 json.dumps(skill_def.get("tool_schemas", []), ensure_ascii=False),
                 skill_def.get("handler_code", ""),
                 json.dumps(skill_def.get("workflow", []), ensure_ascii=False),
                 json.dumps(skill_def.get("trigger", {}), ensure_ascii=False),
                 skill_def.get("success_rate", 0.0),
                 skill_def.get("execution_count", 0),
                 skill_def.get("avg_score", 0.0),
                 1 if skill_def.get("auto_generated") else 0),
            )
        return skill_id

    def uninstall(self, skill_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM skills WHERE id=?", (skill_id,))
            return cursor.rowcount > 0

    def list_skills(self, category: Optional[str] = None) -> List[Dict]:
        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    "SELECT id, name, version, description, author, category, installed_at, "
                    "success_rate, execution_count, avg_score "
                    "FROM skills WHERE category=? ORDER BY name", (category,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, name, version, description, author, category, installed_at, "
                    "success_rate, execution_count, avg_score "
                    "FROM skills ORDER BY name",
                ).fetchall()
        return [dict(r) for r in rows]

    def get_skill(self, skill_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
        if not row:
            return None
        skill = dict(row)
        # Deserialize JSON fields
        for field in ("tool_schemas", "workflow", "trigger"):
            val = skill.get(field)
            if isinstance(val, str):
                try:
                    skill[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    skill[field] = [] if field != "trigger" else {}
        skill["auto_generated"] = bool(skill.get("auto_generated", 0))
        return skill

    def register_skill_tools(self, skill_id: str, registry: ToolRegistry) -> int:
        skill = self.get_skill(skill_id)
        if not skill:
            return 0
        tool_schemas = skill.get("tool_schemas", [])
        if isinstance(tool_schemas, str):
            tool_schemas = json.loads(tool_schemas)
        handler_code = skill["handler_code"]
        count = 0
        for tool_def in tool_schemas:
            name = tool_def["name"]
            description = tool_def.get("description", "")
            parameters = tool_def.get("parameters", {"type": "object", "properties": {}})
            try:
                ns = {}
                exec(handler_code, ns)
                handler = ns.get("handle")
                if handler:
                    registry.register(name, description, parameters, handler)
                    count += 1
            except Exception as e:
                logger.warning("Failed to register skill tool %s: %s", name, e)
        return count

    # ------------------------------------------------------------------
    # Execution history & evolution
    # ------------------------------------------------------------------

    def record_execution(self, skill_id: str, result: Dict[str, Any]) -> int:
        """Record a skill execution result and update aggregated stats.

        Returns:
            The execution record id.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO skill_executions "
                "(skill_id, success, error, completed_steps, total_steps, elapsed_seconds, context, step_results) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    skill_id,
                    1 if result.get("success") else 0,
                    result.get("error", ""),
                    result.get("completed_steps", 0),
                    result.get("total_steps", result.get("completed_steps", 0)),
                    result.get("elapsed_seconds", 0.0),
                    json.dumps(result.get("context", {}), ensure_ascii=False, default=str),
                    json.dumps(result.get("step_results", []), ensure_ascii=False, default=str),
                ),
            )
            exec_id = cursor.lastrowid

            # Update aggregated stats
            self._update_stats(conn, skill_id)
            return exec_id

    def _update_stats(self, conn: sqlite3.Connection, skill_id: str) -> None:
        """Recompute success_rate, execution_count, avg_score from history."""
        row = conn.execute(
            "SELECT COUNT(*) as cnt, AVG(success) as sr, AVG(elapsed_seconds) as avg_elapsed "
            "FROM skill_executions WHERE skill_id=?",
            (skill_id,),
        ).fetchone()
        if row and row["cnt"]:
            conn.execute(
                "UPDATE skills SET execution_count=?, success_rate=?, avg_score=? WHERE id=?",
                (row["cnt"], round(row["sr"], 4), round(row["avg_elapsed"], 4), skill_id),
            )

    def get_execution_history(self, skill_id: str, limit: int = 50) -> List[Dict]:
        """Return recent execution records for a skill."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM skill_executions WHERE skill_id=? ORDER BY executed_at DESC LIMIT ?",
                (skill_id, limit),
            ).fetchall()
        history = []
        for r in rows:
            d = dict(r)
            for field, default in (("context", {}), ("step_results", [])):
                try:
                    d[field] = json.loads(d.get(field, json.dumps(default)))
                except (json.JSONDecodeError, TypeError):
                    d[field] = default
            d["success"] = bool(d["success"])
            history.append(d)
        return history

    def get_skill_stats(self, skill_id: str) -> Optional[Dict]:
        """Return aggregated stats plus recent history summary."""
        skill = self.get_skill(skill_id)
        if not skill:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total, SUM(success) as successes FROM skill_executions WHERE skill_id=?",
                (skill_id,),
            ).fetchone()
        total = row["total"] if row else 0
        successes = row["successes"] if row else 0
        return {
            "skill_id": skill_id,
            "name": skill["name"],
            "version": skill["version"],
            "success_rate": skill.get("success_rate", 0.0),
            "execution_count": skill.get("execution_count", 0),
            "avg_score": skill.get("avg_score", 0.0),
            "total_executions": total,
            "successful_executions": successes,
            "failed_executions": total - successes,
        }

    def evolve_skill(self, skill_id: str, new_workflow: Optional[List[Dict]] = None,
                     new_trigger: Optional[Dict] = None) -> bool:
        """Update a skill's workflow or trigger and bump its version.

        This is the core mutation operation for skill evolution.
        """
        skill = self.get_skill(skill_id)
        if not skill:
            return False

        # Bump version (simple semver: 1.0 -> 1.1)
        version = skill.get("version", "1.0")
        try:
            major, minor = version.split(".")
            new_version = f"{major}.{int(minor) + 1}"
        except (ValueError, TypeError):
            new_version = "1.1"

        workflow = new_workflow if new_workflow is not None else skill.get("workflow", [])
        trigger = new_trigger if new_trigger is not None else skill.get("trigger", {})

        with self._connect() as conn:
            conn.execute(
                "UPDATE skills SET version=?, workflow=?, trigger=? WHERE id=?",
                (new_version,
                 json.dumps(workflow, ensure_ascii=False),
                 json.dumps(trigger, ensure_ascii=False),
                 skill_id),
            )
        logger.info("Evolved skill '%s' to version %s", skill_id, new_version)
        return True
