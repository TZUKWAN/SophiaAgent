"""Unified input resolution for research engines.

Engines previously accepted only `data` (inline JSON) or `path` (file).
With ResultStore, callers can also pass `result_id` to reference a stored
DataFrame produced by a prior tool call. This module centralizes that
resolution so every engine gets the same precedence and the same error
messages.

Precedence (highest first):
    1. `result_id`  — look up via ResultStore, coerce to DataFrame
    2. `data`       — inline JSON (list-of-dicts, dict-of-cols, DataFrame)
    3. `path`       — read CSV / Excel / JSON via WorkspaceGuard
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd


class InputResolutionError(ValueError):
    """Raised when args cannot be resolved to a DataFrame and resolution is required."""


def _read_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            # try line-delimited
            return pd.read_json(path, lines=True)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    raise InputResolutionError(f"Unsupported file extension: {ext}")


def resolve_dataframe(
    args: dict,
    store=None,
    guard=None,
    *,
    require: bool = False,
    data_key: str = "data",
    path_key: str = "path",
    result_key: str = "result_id",
) -> Optional[pd.DataFrame]:
    """Resolve args to a DataFrame.

    Args:
        args: tool arguments dict.
        store: ResultStore (required if args contain result_id).
        guard: WorkspaceGuard (required if args contain path).
        require: if True, raise InputResolutionError when nothing resolves.
        data_key/path_key/result_key: argument key names to inspect.

    Returns:
        DataFrame, or None if no recognizable input and require=False.
    """
    # 1. result_id
    rid = args.get(result_key)
    if rid:
        if store is None:
            raise InputResolutionError(
                f"{result_key}={rid} given but no ResultStore configured"
            )
        if not store.exists(rid):
            raise InputResolutionError(f"{result_key} not found: {rid}")
        try:
            return store.get_dataframe(rid)
        except TypeError as exc:
            raise InputResolutionError(
                f"result_id {rid} could not be coerced to DataFrame: {exc}"
            ) from exc

    # 2. inline data
    data = args.get(data_key)
    if data is not None:
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, pd.Series):
            return data.to_frame()
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        if isinstance(data, list):
            if len(data) == 0:
                return pd.DataFrame()
            if all(isinstance(x, dict) for x in data):
                return pd.DataFrame(data)
            return pd.DataFrame({"value": data})
        if isinstance(data, dict):
            # dict-of-columns
            try:
                return pd.DataFrame(data)
            except Exception as exc:
                raise InputResolutionError(
                    f"Cannot construct DataFrame from dict: {exc}"
                ) from exc

    # 3. path
    path = args.get(path_key)
    if path:
        if guard is None:
            # Permit absolute-resolvable read without guard (tests / unit calls)
            if not os.path.isfile(path):
                raise InputResolutionError(f"File not found: {path}")
            return _read_file(path)
        resolved = guard.resolve_read(path)
        return _read_file(resolved)

    if require:
        raise InputResolutionError(
            f"No data provided: expected one of {result_key!r} / {data_key!r} / {path_key!r}"
        )
    return None


def resolve_parent_ids(args: dict, *keys: str) -> list:
    """Collect any `result_id`-shaped values from args for lineage tracking."""
    out = []
    candidate_keys = list(keys) if keys else [
        "result_id", "result_ids", "parent_id", "parents",
        "x_result_id", "y_result_id", "group1_result_id", "group2_result_id",
    ]
    for k in candidate_keys:
        v = args.get(k)
        if isinstance(v, str) and v.startswith("res_"):
            out.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.startswith("res_"):
                    out.append(item)
    # Dedupe preserving order
    seen = set()
    uniq = []
    for rid in out:
        if rid not in seen:
            seen.add(rid)
            uniq.append(rid)
    return uniq
