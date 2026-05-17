"""SQLite-backed persistent store for research tool results.

Allows tool-to-tool data flow via lightweight result_id references instead of
forcing the LLM to manually pass DataFrames between tools. Small results are
stored inline as JSON; large payloads (DataFrames, fitted models, ndarrays)
are pickled to disk under the workspace's .research/cache/results/ directory.

Schema
------
results(
    id TEXT PRIMARY KEY,           -- "res_<8-hex>"
    kind TEXT NOT NULL,            -- dataframe|result|model|figure|array|text
    tool TEXT NOT NULL,            -- producing tool name
    params TEXT,                   -- JSON-encoded args passed to the tool
    parents TEXT DEFAULT '[]',     -- JSON list of upstream result_ids
    payload_path TEXT,             -- absolute path to pickle file, or NULL
    payload_inline TEXT,           -- inline JSON for small payloads, or NULL
    summary TEXT,                  -- short JSON preview (shape/columns/keys)
    created_at TEXT NOT NULL,      -- ISO8601 timestamp
    workspace TEXT NOT NULL        -- absolute workspace path
)
"""
from __future__ import annotations

import json
import os
import pickle
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

INLINE_LIMIT_BYTES = 8 * 1024  # 8KB
ALLOWED_KINDS = {
    "dataframe",
    "result",
    "model",
    "figure",
    "array",
    "text",
    "dict",
}


class ResultStore:
    """SQLite-backed persistent store for tool results, indexed by result_id."""

    def __init__(self, workspace: str):
        self.workspace = os.path.realpath(workspace)
        research_dir = os.path.join(self.workspace, ".research")
        os.makedirs(research_dir, exist_ok=True)
        self.payload_dir = os.path.join(research_dir, "cache", "results")
        os.makedirs(self.payload_dir, exist_ok=True)
        self.db_path = os.path.join(research_dir, "results.db")
        self._conn = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------
    def _init_schema(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                tool TEXT NOT NULL,
                params TEXT,
                parents TEXT DEFAULT '[]',
                payload_path TEXT,
                payload_inline TEXT,
                summary TEXT,
                created_at TEXT NOT NULL,
                workspace TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kind ON results(kind)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tool ON results(tool)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_created ON results(created_at)")

    # ------------------------------------------------------------------
    # id generation
    # ------------------------------------------------------------------
    @staticmethod
    def _new_id() -> str:
        return "res_" + secrets.token_hex(4)

    # ------------------------------------------------------------------
    # serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_complex_types(data: Any) -> bool:
        """Return True if data contains numpy / pandas / non-jsonable objects."""
        if isinstance(data, (np.ndarray, np.generic, pd.DataFrame, pd.Series)):
            return True
        if isinstance(data, dict):
            return any(
                ResultStore._has_complex_types(k) or ResultStore._has_complex_types(v)
                for k, v in data.items()
            )
        if isinstance(data, (list, tuple, set)):
            return any(ResultStore._has_complex_types(x) for x in data)
        return False

    @classmethod
    def _is_small_inlinable(cls, data: Any) -> bool:
        if data is None:
            return True
        if isinstance(data, (int, float, bool, str)):
            return True
        if isinstance(data, (list, tuple, dict)):
            # Refuse inline if any element is numpy/pandas — JSON round-trip
            # would silently coerce arrays to strings via default=str.
            if cls._has_complex_types(data):
                return False
            try:
                blob = json.dumps(data)
                return len(blob.encode("utf-8")) <= INLINE_LIMIT_BYTES
            except (TypeError, ValueError):
                return False
        return False

    @staticmethod
    def _summarize(data: Any, kind: str) -> str:
        try:
            if isinstance(data, pd.DataFrame):
                return json.dumps(
                    {
                        "shape": list(data.shape),
                        "columns": list(data.columns)[:30],
                        "dtypes": {c: str(data[c].dtype) for c in list(data.columns)[:30]},
                    },
                    default=str,
                )
            if isinstance(data, np.ndarray):
                return json.dumps(
                    {"shape": list(data.shape), "dtype": str(data.dtype)},
                    default=str,
                )
            if isinstance(data, dict):
                return json.dumps(
                    {"keys": list(data.keys())[:30], "size": len(data)},
                    default=str,
                )
            if isinstance(data, (list, tuple)):
                return json.dumps({"length": len(data), "type": type(data).__name__})
            if isinstance(data, (int, float, bool, str)):
                preview = str(data)[:200]
                return json.dumps({"type": type(data).__name__, "preview": preview})
            return json.dumps({"type": type(data).__name__})
        except Exception as exc:  # pragma: no cover - defensive
            return json.dumps({"error": f"summary_failed: {exc}"})

    def _write_pickle(self, result_id: str, data: Any) -> str:
        path = os.path.join(self.payload_dir, f"{result_id}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def _read_pickle(self, path: str) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # store
    # ------------------------------------------------------------------
    def store(
        self,
        data: Any,
        kind: str,
        tool: str,
        params: Optional[Dict[str, Any]] = None,
        parents: Optional[List[str]] = None,
    ) -> str:
        """Persist a tool result and return its result_id."""
        if kind not in ALLOWED_KINDS:
            raise ValueError(
                f"Unknown kind={kind!r}; allowed: {sorted(ALLOWED_KINDS)}"
            )
        if not tool or not isinstance(tool, str):
            raise ValueError("tool must be a non-empty string")

        parents = list(parents) if parents else []
        # Validate parent ids exist
        for pid in parents:
            if not self.exists(pid):
                raise ValueError(f"Parent result_id not found: {pid}")

        result_id = self._new_id()
        # Ensure uniqueness (cosmically rare collision but cheap to guard)
        while self.exists(result_id):
            result_id = self._new_id()

        # Always pickle DataFrames / ndarrays / models for fidelity
        payload_path = None
        payload_inline = None
        if isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
            payload_path = self._write_pickle(result_id, data)
        elif kind == "model":
            payload_path = self._write_pickle(result_id, data)
        elif self._is_small_inlinable(data):
            try:
                payload_inline = json.dumps(data, default=str)
            except (TypeError, ValueError):
                payload_path = self._write_pickle(result_id, data)
        else:
            payload_path = self._write_pickle(result_id, data)

        summary = self._summarize(data, kind)
        params_json = json.dumps(params or {}, default=str)
        parents_json = json.dumps(parents)
        created_at = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO results
            (id, kind, tool, params, parents, payload_path, payload_inline,
             summary, created_at, workspace)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                kind,
                tool,
                params_json,
                parents_json,
                payload_path,
                payload_inline,
                summary,
                created_at,
                self.workspace,
            ),
        )
        return result_id

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------
    def exists(self, result_id: str) -> bool:
        if not result_id:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM results WHERE id = ?", (result_id,)
        ).fetchone()
        return row is not None

    def get(self, result_id: str) -> Any:
        """Materialize the payload for a result_id, or raise KeyError."""
        row = self._conn.execute(
            "SELECT payload_path, payload_inline FROM results WHERE id = ?",
            (result_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"result_id not found: {result_id}")
        payload_path, payload_inline = row
        if payload_path:
            return self._read_pickle(payload_path)
        if payload_inline is not None:
            return json.loads(payload_inline)
        return None

    def get_dataframe(self, result_id: str) -> pd.DataFrame:
        """Materialize as a DataFrame; raises TypeError if conversion impossible."""
        data = self.get(result_id)
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, pd.Series):
            return data.to_frame()
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        if isinstance(data, list):
            if all(isinstance(x, dict) for x in data):
                return pd.DataFrame(data)
            return pd.DataFrame({"value": data})
        if isinstance(data, dict):
            try:
                return pd.DataFrame(data)
            except Exception as exc:
                raise TypeError(
                    f"Cannot coerce dict result {result_id} to DataFrame: {exc}"
                ) from exc
        raise TypeError(
            f"Result {result_id} of type {type(data).__name__} is not a DataFrame"
        )

    def get_metadata(self, result_id: str) -> Dict[str, Any]:
        """Return metadata (kind/tool/params/parents/summary) without loading payload."""
        row = self._conn.execute(
            """
            SELECT id, kind, tool, params, parents, summary, created_at,
                   payload_path, payload_inline
            FROM results WHERE id = ?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"result_id not found: {result_id}")
        return {
            "id": row[0],
            "kind": row[1],
            "tool": row[2],
            "params": json.loads(row[3]) if row[3] else {},
            "parents": json.loads(row[4]) if row[4] else [],
            "summary": json.loads(row[5]) if row[5] else {},
            "created_at": row[6],
            "payload_path": row[7],
            "inline": row[8] is not None,
        }

    # ------------------------------------------------------------------
    # list / search
    # ------------------------------------------------------------------
    def list_by_kind(
        self,
        kind: Optional[str] = None,
        tool: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if tool:
            clauses.append("tool = ?")
            params.append(tool)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        rows = self._conn.execute(
            f"""
            SELECT id, kind, tool, summary, created_at, parents
            FROM results{where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            {
                "id": r[0],
                "kind": r[1],
                "tool": r[2],
                "summary": json.loads(r[3]) if r[3] else {},
                "created_at": r[4],
                "parents": json.loads(r[5]) if r[5] else [],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # lineage
    # ------------------------------------------------------------------
    def lineage(self, result_id: str, max_depth: int = 32) -> List[Dict[str, Any]]:
        """Return upstream lineage (BFS) starting from result_id.

        Returns a flat list, root first (the queried result_id), then its
        immediate parents, then grandparents, etc. Cycles are guarded.
        """
        if not self.exists(result_id):
            raise KeyError(f"result_id not found: {result_id}")
        seen = set()
        queue = [(result_id, 0)]
        out: List[Dict[str, Any]] = []
        while queue:
            cur, depth = queue.pop(0)
            if cur in seen or depth > max_depth:
                continue
            seen.add(cur)
            try:
                meta = self.get_metadata(cur)
            except KeyError:
                continue
            meta["depth"] = depth
            out.append(meta)
            for p in meta.get("parents", []):
                if p not in seen:
                    queue.append((p, depth + 1))
        return out

    # ------------------------------------------------------------------
    # delete / cleanup
    # ------------------------------------------------------------------
    def delete(self, result_id: str) -> bool:
        meta_row = self._conn.execute(
            "SELECT payload_path FROM results WHERE id = ?", (result_id,)
        ).fetchone()
        if meta_row is None:
            return False
        payload_path = meta_row[0]
        cur = self._conn.execute("DELETE FROM results WHERE id = ?", (result_id,))
        if payload_path and os.path.exists(payload_path):
            try:
                os.remove(payload_path)
            except OSError:
                pass
        return cur.rowcount > 0

    def clear(self) -> int:
        """Remove all entries (and pickled payloads). Returns count removed."""
        rows = self._conn.execute(
            "SELECT payload_path FROM results WHERE payload_path IS NOT NULL"
        ).fetchall()
        for (path,) in rows:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        cur = self._conn.execute("DELETE FROM results")
        return cur.rowcount

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        by_kind = dict(
            self._conn.execute(
                "SELECT kind, COUNT(*) FROM results GROUP BY kind"
            ).fetchall()
        )
        by_tool = dict(
            self._conn.execute(
                "SELECT tool, COUNT(*) FROM results GROUP BY tool"
            ).fetchall()
        )
        inline_count = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE payload_inline IS NOT NULL"
        ).fetchone()[0]
        disk_count = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE payload_path IS NOT NULL"
        ).fetchone()[0]
        return {
            "total": total,
            "inline": inline_count,
            "on_disk": disk_count,
            "by_kind": by_kind,
            "by_tool": by_tool,
        }

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------
    def close(self):
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
