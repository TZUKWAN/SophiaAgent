"""Experiment pipeline: data loading, validation, transformation, reporting.

P1.3 update — wires in ResultStore and GlobalSeed:

- ``load_data`` stores the loaded DataFrame in :class:`ResultStore` and returns
  a ``result_id`` alongside the existing preview JSON. Downstream tools can pass
  that id directly instead of re-reading the file or shuttling rows through the
  LLM by hand.
- ``validate_data`` / ``transform`` accept a ``result_id`` (via the unified
  ``_input.resolve_dataframe`` helper) and, in the case of ``transform``, store
  the transformed DataFrame as a new result with the source ``result_id`` as
  its parent — lineage is preserved automatically.
- ``save_results`` accepts a ``result_id`` to flush stored content to disk.
- ``snapshot`` records ``result_ids`` (and their full lineage) alongside the
  file hashes so an experiment can be replayed end to end.
- ``seed_manager`` delegates to :class:`GlobalSeed`, which propagates to
  numpy, ``random``, ``PYTHONHASHSEED`` and (optionally) torch / tensorflow.

All changes are additive — every existing call signature still works.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sophia.research.workspace_guard import WorkspaceGuard
from sophia.research._input import resolve_dataframe, resolve_parent_ids
from sophia.research.seed import GlobalSeed


class ExperimentPipeline:
    """Manages experiment data loading, validation, transformation, and reporting."""

    def __init__(self, workspace: str, store: Optional[Any] = None):
        """Initialize the pipeline.

        Args:
            workspace: absolute workspace path (used by WorkspaceGuard).
            store: optional :class:`sophia.research.result_store.ResultStore`.
                When provided, ``load_data`` / ``transform`` persist their
                DataFrames and return a ``result_id``. Other tools can then
                accept that id via the standard ``_input.resolve_dataframe``
                helper, avoiding the LLM-as-data-pipe anti-pattern.
        """
        self.guard = WorkspaceGuard(workspace)
        self.store = store

    # ------------------------------------------------------------------
    # load_data
    # ------------------------------------------------------------------
    def load_data(self, args: dict) -> str:
        """Load data from file.

        Args:
            args: {
                path: str,
                format: str ('csv'|'excel'|'json'|'auto'),
                sheet: str (for excel),
                columns: list (select columns)
            }
        All paths relative to workspace.

        Returns:
            JSON string with shape, columns with dtypes, the first 5 rows, and
            (when a ResultStore is configured) the ``result_id`` of the stored
            DataFrame.
        """
        path = args.get("path", "")
        fmt = args.get("format", "auto")
        sheet = args.get("sheet", 0)
        columns = args.get("columns", None)

        try:
            resolved = self.guard.resolve_read(path)
        except (FileNotFoundError, PermissionError) as exc:
            return json.dumps({"error": str(exc)})

        if fmt == "auto":
            ext = os.path.splitext(resolved)[1].lower()
            fmt_map = {".csv": "csv", ".xlsx": "excel", ".xls": "excel", ".json": "json"}
            fmt = fmt_map.get(ext, "csv")

        try:
            if fmt == "csv":
                df = pd.read_csv(resolved)
            elif fmt == "excel":
                df = pd.read_excel(resolved, sheet_name=sheet)
            elif fmt == "json":
                df = pd.read_json(resolved)
            else:
                return json.dumps({"error": f"Unsupported format: {fmt}"})
        except Exception as exc:
            return json.dumps({"error": f"Failed to load data: {exc}"})

        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return json.dumps({"error": f"Columns not found: {missing}"})
            df = df[columns]

        dtypes = {col: str(df[col].dtype) for col in df.columns}
        head = df.head(5).to_dict(orient="records")
        # Convert numpy types so json.dumps doesn't choke
        head = json.loads(json.dumps(head, default=str))

        out: Dict[str, Any] = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": dtypes,
            "head": head,
        }

        if self.store is not None:
            rid = self.store.store(
                df,
                kind="dataframe",
                tool="research_load_data",
                params={
                    "path": path,
                    "format": fmt,
                    "sheet": sheet,
                    "columns": columns,
                },
            )
            out["result_id"] = rid

        return json.dumps(out)

    # ------------------------------------------------------------------
    # validate_data
    # ------------------------------------------------------------------
    def validate_data(self, args: dict) -> str:
        """Validate a DataFrame.

        Args:
            args: {
                result_id: str (preferred — look up in ResultStore), or
                data: list of dicts, or
                path: str,
                checks: list (e.g. ['missing', 'outliers', 'types', 'distribution'])
            }

        Returns:
            JSON string with validation report. When a ResultStore is
            configured the report is also persisted (kind='result') and its
            ``result_id`` is included in the response with the source frame as
            its parent.
        """
        checks = args.get("checks", ["missing", "outliers", "types", "distribution"])
        df = self._resolve_dataframe(args)
        if df is None:
            return json.dumps({
                "error": "No data provided. Use 'result_id', 'data', or 'path'."
            })

        report: Dict[str, Any] = {"rows": len(df), "columns": len(df.columns)}

        if "missing" in checks:
            missing_counts = df.isnull().sum().to_dict()
            missing_pct = (df.isnull().mean() * 100).to_dict()
            # Convert to plain Python types
            report["missing"] = {
                col: {
                    "count": int(missing_counts[col]),
                    "percent": round(float(missing_pct[col]), 2),
                }
                for col in df.columns
            }

        if "outliers" in checks:
            outlier_info: Dict[str, Any] = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = series[(series < lower) | (series > upper)]
                outlier_info[col] = {
                    "count": int(len(outliers)),
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                    "outlier_values": [round(float(v), 4) for v in outliers.head(10).tolist()],
                }
            report["outliers"] = outlier_info

        if "types" in checks:
            type_info: Dict[str, Any] = {}
            for col in df.columns:
                series = df[col].dropna()
                if len(series) == 0:
                    type_info[col] = {"dtype": str(df[col].dtype), "inferred": "empty", "consistent": True}
                    continue
                inferred = str(pd.api.types.infer_dtype(series, skipna=True))
                type_info[col] = {
                    "dtype": str(df[col].dtype),
                    "inferred": inferred,
                    "non_null": int(len(series)),
                    "null": int(df[col].isnull().sum()),
                }
            report["types"] = type_info

        if "distribution" in checks:
            dist_info: Dict[str, Any] = {}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                dist_info[col] = {
                    "mean": round(float(series.mean()), 4),
                    "std": round(float(series.std()), 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                    "median": round(float(series.median()), 4),
                    "skew": round(float(series.skew()), 4) if len(series) > 2 else None,
                    "kurtosis": round(float(series.kurtosis()), 4) if len(series) > 3 else None,
                }
            report["distribution"] = dist_info

        report["issues_found"] = self._count_issues(report)

        if self.store is not None:
            parents = resolve_parent_ids(args)
            rid = self.store.store(
                report,
                kind="result",
                tool="research_validate_data",
                params={k: v for k, v in args.items() if k != "data"},
                parents=parents,
            )
            report["result_id"] = rid

        return json.dumps(report, default=str)

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, args: dict) -> str:
        """Transform data.

        Args:
            args: {
                result_id: str (preferred), or data: list of dicts, or path: str,
                operations: list of {type: str, params: dict}
            }
        Operations: 'standardize', 'normalize', 'encode', 'impute', 'filter',
                    'select', 'rename'

        Returns:
            JSON string with transformed data summary. When a ResultStore is
            configured the transformed DataFrame is persisted as a new
            ``dataframe`` result whose parent is the source frame.
        """
        df = self._resolve_dataframe(args)
        if df is None:
            return json.dumps({
                "error": "No data provided. Use 'result_id', 'data', or 'path'."
            })

        operations = args.get("operations", [])
        applied: List[Dict[str, Any]] = []

        for op in operations:
            op_type = op.get("type", "")
            params = op.get("params", {})

            if op_type == "standardize":
                cols = params.get("columns", list(df.select_dtypes(include=[np.number]).columns))
                for col in cols:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        mean = df[col].mean()
                        std = df[col].std()
                        if std != 0:
                            df[col] = (df[col] - mean) / std
                        applied.append({"operation": "standardize", "column": col,
                                        "mean": round(float(mean), 4),
                                        "std": round(float(std), 4)})

            elif op_type == "normalize":
                cols = params.get("columns", list(df.select_dtypes(include=[np.number]).columns))
                method = params.get("method", "minmax")  # minmax or maxabs
                for col in cols:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        if method == "minmax":
                            mn = df[col].min()
                            mx = df[col].max()
                            rng = mx - mn
                            if rng != 0:
                                df[col] = (df[col] - mn) / rng
                            applied.append({"operation": "normalize", "column": col, "method": "minmax"})
                        elif method == "maxabs":
                            max_abs = df[col].abs().max()
                            if max_abs != 0:
                                df[col] = df[col] / max_abs
                            applied.append({"operation": "normalize", "column": col, "method": "maxabs"})

            elif op_type == "encode":
                cols = params.get("columns", [])
                method = params.get("method", "onehot")  # onehot or label
                for col in cols:
                    if col not in df.columns:
                        continue
                    if method == "onehot":
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                        applied.append({"operation": "encode", "column": col, "method": "onehot",
                                        "new_columns": list(dummies.columns)})
                    elif method == "label":
                        codes, uniques = pd.factorize(df[col])
                        df[col] = codes
                        applied.append({"operation": "encode", "column": col, "method": "label",
                                        "unique_values": list(uniques)})

            elif op_type == "impute":
                cols = params.get("columns", list(df.columns))
                strategy = params.get("strategy", "mean")  # mean, median, mode, constant
                fill_value = params.get("value", None)
                for col in cols:
                    if col not in df.columns:
                        continue
                    missing_before = int(df[col].isnull().sum())
                    if missing_before == 0:
                        continue
                    if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == "mode":
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].fillna(mode_val.iloc[0])
                    elif strategy == "constant":
                        df[col] = df[col].fillna(fill_value)
                    applied.append({"operation": "impute", "column": col,
                                    "strategy": strategy, "filled": missing_before})

            elif op_type == "filter":
                column = params.get("column", "")
                condition = params.get("condition", "")
                value = params.get("value", None)
                rows_before = len(df)
                if column and column in df.columns:
                    if condition == "==" and value is not None:
                        df = df[df[column] == value]
                    elif condition == "!=" and value is not None:
                        df = df[df[column] != value]
                    elif condition == ">" and value is not None:
                        df = df[df[column] > value]
                    elif condition == "<" and value is not None:
                        df = df[df[column] < value]
                    elif condition == ">=" and value is not None:
                        df = df[df[column] >= value]
                    elif condition == "<=" and value is not None:
                        df = df[df[column] <= value]
                    elif condition == "in" and isinstance(value, list):
                        df = df[df[column].isin(value)]
                    elif condition == "notnull":
                        df = df[df[column].notnull()]
                applied.append({"operation": "filter", "column": column,
                                "condition": condition, "rows_before": rows_before,
                                "rows_after": len(df)})

            elif op_type == "select":
                cols = params.get("columns", [])
                existing = [c for c in cols if c in df.columns]
                df = df[existing]
                applied.append({"operation": "select", "columns": existing})

            elif op_type == "rename":
                mapping = params.get("mapping", {})
                df = df.rename(columns=mapping)
                applied.append({"operation": "rename", "mapping": mapping})

        result = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "operations_applied": len(applied),
            "details": applied,
            "head": json.loads(df.head(5).to_json(orient="records", default_handler=str)),
        }

        if self.store is not None:
            parents = resolve_parent_ids(args)
            rid = self.store.store(
                df,
                kind="dataframe",
                tool="research_transform",
                params={"operations": operations},
                parents=parents,
            )
            result["result_id"] = rid

        return json.dumps(result, default=str)

    # ------------------------------------------------------------------
    # save_results
    # ------------------------------------------------------------------
    def save_results(self, args: dict) -> str:
        """Save results to file.

        Args:
            args: {
                result_id: str (preferred — load from ResultStore), or
                data: any,
                path: str,
                format: str ('json'|'csv')
            }
        Path relative to workspace/.research/cache/.

        Returns:
            JSON string with saved path and file size.
        """
        path = args.get("path", "results.json")
        fmt = args.get("format", "json")

        # Source of truth: explicit data > result_id > error
        data = args.get("data", None)
        if data is None and self.store is not None and args.get("result_id"):
            rid = args["result_id"]
            if not self.store.exists(rid):
                return json.dumps({"error": f"result_id not found: {rid}"})
            data = self.store.get(rid)

        if data is None:
            return json.dumps({"error": "No data provided. Use 'data' or 'result_id'."})

        resolved = self.guard.resolve_write(path, subdir="cache")

        try:
            if fmt == "json":
                # If we pulled a DataFrame from the store, convert to records
                if isinstance(data, pd.DataFrame):
                    out_data = json.loads(data.to_json(orient="records", default_handler=str))
                elif isinstance(data, np.ndarray):
                    out_data = data.tolist()
                else:
                    out_data = data
                with open(resolved, "w", encoding="utf-8") as f:
                    json.dump(out_data, f, indent=2, default=str)
            elif fmt == "csv":
                if isinstance(data, pd.DataFrame):
                    data.to_csv(resolved, index=False)
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                    df.to_csv(resolved, index=False)
                else:
                    return json.dumps({"error": "CSV format requires a DataFrame or list of dicts"})
            else:
                return json.dumps({"error": f"Unsupported format: {fmt}"})
        except Exception as exc:
            return json.dumps({"error": f"Failed to save: {exc}"})

        size = os.path.getsize(resolved)
        return json.dumps({
            "path": resolved,
            "relative_path": path,
            "format": fmt,
            "size_bytes": size,
            "size_human": f"{size} bytes",
        })

    # ------------------------------------------------------------------
    # export_report
    # ------------------------------------------------------------------
    def export_report(self, args: dict) -> str:
        """Export a research report.

        Args:
            args: {
                title: str,
                sections: list of {heading: str, content: str},
                path: str,
                format: str ('markdown'|'html')
            }
        Saved to workspace/.research/reports/.

        Returns:
            JSON string with path.
        """
        title = args.get("title", "Research Report")
        sections = args.get("sections", [])
        path = args.get("path", "report.md")
        fmt = args.get("format", "markdown")

        resolved = self.guard.resolve_write(path, subdir="reports")

        if fmt == "markdown":
            content = f"# {title}\n\n"
            content += f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
            for sec in sections:
                heading = sec.get("heading", "Section")
                body = sec.get("content", "")
                content += f"## {heading}\n\n{body}\n\n"
        elif fmt == "html":
            content = "<!DOCTYPE html>\n<html>\n<head>\n"
            content += f"<title>{title}</title>\n"
            content += "<style>body{font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px;}"
            content += "h1{color:#333}h2{color:#555;border-bottom:1px solid #eee;padding-bottom:5px}</style>\n"
            content += "</head>\n<body>\n"
            content += f"<h1>{title}</h1>\n"
            content += f"<p><em>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</em></p>\n"
            for sec in sections:
                heading = sec.get("heading", "Section")
                body = sec.get("content", "")
                content += f"<h2>{heading}</h2>\n<p>{body}</p>\n"
            content += "</body>\n</html>"
        else:
            return json.dumps({"error": f"Unsupported format: {fmt}"})

        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)

        size = os.path.getsize(resolved)
        return json.dumps({
            "path": resolved,
            "relative_path": path,
            "format": fmt,
            "size_bytes": size,
            "sections": len(sections),
        })

    # ------------------------------------------------------------------
    # snapshot
    # ------------------------------------------------------------------
    def snapshot(self, args: dict) -> str:
        """Create experiment snapshot.

        Args:
            args: {
                label: str,
                data_paths: list,        # files to hash for reproducibility
                result_ids: list,        # stored results to capture (with lineage)
                code: str
            }

        Returns:
            JSON string with snapshot_id, timestamp, data_hashes, and (when
            applicable) the full lineage of each captured result_id.
        """
        label = args.get("label", "unnamed")
        data_paths = args.get("data_paths", [])
        result_ids = args.get("result_ids", [])
        code = args.get("code", "")

        hashes: Dict[str, str] = {}
        for dp in data_paths:
            try:
                resolved = self.guard.resolve_read(dp)
                md5 = hashlib.md5()
                with open(resolved, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        md5.update(chunk)
                hashes[dp] = md5.hexdigest()
            except (FileNotFoundError, PermissionError):
                hashes[dp] = "ERROR: file not found or not accessible"

        # Capture result_id lineage if a store is configured
        result_lineage: Dict[str, Any] = {}
        if self.store is not None and result_ids:
            for rid in result_ids:
                try:
                    lineage = self.store.lineage(rid)
                    result_lineage[rid] = [
                        {
                            "id": item["id"],
                            "kind": item["kind"],
                            "tool": item["tool"],
                            "depth": item["depth"],
                            "created_at": item["created_at"],
                        }
                        for item in lineage
                    ]
                except KeyError:
                    result_lineage[rid] = {"error": "result_id not found"}

        ts = time.strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"snap_{ts}_{hashlib.md5(label.encode()).hexdigest()[:8]}"

        # Save snapshot metadata
        meta: Dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "label": label,
            "timestamp": ts,
            "data_hashes": hashes,
            "code_hash": hashlib.md5(code.encode()).hexdigest() if code else None,
            "seed": GlobalSeed.get(),
        }
        if result_lineage:
            meta["result_lineage"] = result_lineage
        meta_path = self.guard.resolve_write(f"{snapshot_id}.json", subdir="cache")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return json.dumps(meta, default=str)

    # ------------------------------------------------------------------
    # seed_manager
    # ------------------------------------------------------------------
    def seed_manager(self, args: dict) -> str:
        """Manage the global random seed.

        Delegates to :class:`GlobalSeed`, which propagates to numpy,
        ``random``, ``PYTHONHASHSEED`` and (optionally) torch / tensorflow.

        Args:
            args: {action: str ('set'|'get'|'reset'), seed: int}

        Returns:
            JSON string with current seed.
        """
        action = args.get("action", "get")

        if action == "set":
            seed = args.get("seed")
            if seed is None:
                return json.dumps({"error": "Seed value required for 'set' action"})
            try:
                applied = GlobalSeed.set(seed)
            except (TypeError, ValueError) as exc:
                return json.dumps({"error": f"Invalid seed value: {exc}"})
            return json.dumps({"action": "set", "seed": applied})

        if action == "reset":
            GlobalSeed.reset()
            return json.dumps({"action": "reset", "seed": None})

        # get
        return json.dumps({"action": "get", "seed": GlobalSeed.get()})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_dataframe(self, args: dict) -> Optional[pd.DataFrame]:
        """Resolve a DataFrame from args via the unified resolver.

        Precedence (highest first): ``result_id`` > ``data`` > ``path``. Falls
        back to ``None`` so callers can emit the legacy ``error`` payload.
        """
        try:
            df = resolve_dataframe(args, store=self.store, guard=self.guard)
        except Exception:  # noqa: BLE001 — match prior best-effort semantics
            return None
        return df

    @staticmethod
    def _count_issues(report: dict) -> int:
        """Count total issues in a validation report."""
        count = 0
        if "missing" in report:
            for col_info in report["missing"].values():
                if col_info["count"] > 0:
                    count += 1
        if "outliers" in report:
            for col_info in report["outliers"].values():
                if col_info["count"] > 0:
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Backwards-compat shim for legacy attribute
    # ------------------------------------------------------------------
    @property
    def _current_seed(self) -> Optional[int]:  # pragma: no cover - shim
        return GlobalSeed.get()

    @_current_seed.setter
    def _current_seed(self, value: Optional[int]):  # pragma: no cover - shim
        if value is None:
            GlobalSeed.reset()
        else:
            GlobalSeed.set(value)
