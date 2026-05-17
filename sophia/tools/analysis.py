"""Data analysis tool for SophiaAgent.

Provides descriptive statistics, statistical tests, visualization,
and code execution for social science data analysis.
"""

import builtins
import io
import json
import logging
import os
import traceback
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _data_dir(workspace: str) -> str:
    return os.path.join(workspace, ".sophia", "data")


def _ensure_data_dir(workspace: str) -> None:
    os.makedirs(_data_dir(workspace), exist_ok=True)


def _load_dataframe(file_path: str) -> Any:
    """Load a CSV or Excel file into a DataFrame."""
    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for data analysis. Install with: pip install pandas")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    elif ext == ".json":
        return pd.read_json(file_path)
    elif ext in (".sav", ".zsav"):
        try:
            import pyreadstat
            df, _ = pyreadstat.read_sav(file_path)
            return df
        except ImportError:
            raise RuntimeError(
                "pyreadstat is required for SPSS files. "
                "Install with: pip install pyreadstat"
            )
    elif ext == ".dta":
        return pd.read_stata(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def data_load(args: Dict[str, Any], workspace: str) -> str:
    """Load a dataset for analysis.

    Args: {path: str}
    """
    path = args.get("path", "")
    if not path:
        return json.dumps({"error": "path is required"}, ensure_ascii=False)

    if not os.path.isabs(path):
        path = os.path.join(workspace, path)

    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)

    try:
        df = _load_dataframe(path)
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            nulls = int(df[col].isnull().sum())
            sample = df[col].dropna().head(3).tolist()
            columns.append({
                "name": col,
                "dtype": dtype,
                "nulls": nulls,
                "sample": sample,
            })

        return json.dumps({
            "file": path,
            "rows": len(df),
            "columns_count": len(df.columns),
            "columns": columns,
        }, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def data_describe(args: Dict[str, Any], workspace: str) -> str:
    """Generate descriptive statistics for a dataset.

    Args: {path: str, columns: list[str]}
    """
    path = args.get("path", "")
    if not path:
        return json.dumps({"error": "path is required"}, ensure_ascii=False)

    if not os.path.isabs(path):
        path = os.path.join(workspace, path)

    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)

    try:
        df = _load_dataframe(path)
        selected_cols = args.get("columns", [])
        if selected_cols:
            missing = [c for c in selected_cols if c not in df.columns]
            if missing:
                return json.dumps({"error": f"Columns not found: {missing}"}, ensure_ascii=False)
            df = df[selected_cols]

        desc = df.describe(include="all").to_dict()

        result = {}
        for col, stats in desc.items():
            col_stats = {}
            for stat_name, value in stats.items():
                if pd.isna(value):
                    continue
                if isinstance(value, float):
                    col_stats[stat_name] = round(value, 4)
                else:
                    col_stats[stat_name] = str(value)
            result[col] = col_stats

        return json.dumps({
            "file": path,
            "rows": len(df),
            "statistics": result,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def data_visualize(args: Dict[str, Any], workspace: str) -> str:
    """Generate a chart from data.

    Args: {path: str, chart_type: str, x: str, y: str, title: str, output: str}
    """
    if not HAS_MATPLOTLIB:
        return json.dumps({"error": "matplotlib is required for visualization"}, ensure_ascii=False)
    if not HAS_PANDAS:
        return json.dumps({"error": "pandas is required"}, ensure_ascii=False)

    path = args.get("path", "")
    if not path:
        return json.dumps({"error": "path is required"}, ensure_ascii=False)

    if not os.path.isabs(path):
        path = os.path.join(workspace, path)

    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)

    try:
        df = _load_dataframe(path)
        chart_type = args.get("chart_type", "bar")
        x_col = args.get("x", "")
        y_col = args.get("y", "")
        title = args.get("title", "")

        _ensure_data_dir(workspace)
        output = args.get("output", "chart.png")
        if not os.path.isabs(output):
            output = os.path.join(_data_dir(workspace), output)

        fig, ax = plt.subplots(figsize=(10, 6))
        # Configure CJK font for Chinese labels
        cjk_fonts = [
            'SimHei', 'Microsoft YaHei',
            'STSong', 'WenQuanYi Micro Hei', 'Arial Unicode MS',
        ]
        for font_name in cjk_fonts:
            try:
                import matplotlib.font_manager as fm
                if any(font_name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    break
            except Exception:
                continue

        if chart_type == "bar":
            if x_col and y_col:
                df.plot.bar(x=x_col, y=y_col, ax=ax)
            else:
                df.head(20).plot.bar(ax=ax)
        elif chart_type == "line":
            if x_col and y_col:
                df.plot.line(x=x_col, y=y_col, ax=ax)
            else:
                df.head(50).plot.line(ax=ax)
        elif chart_type == "scatter":
            if x_col and y_col:
                df.plot.scatter(x=x_col, y=y_col, ax=ax)
            else:
                return json.dumps({"error": "scatter requires x and y columns"}, ensure_ascii=False)
        elif chart_type == "hist":
            col = y_col or x_col
            if col:
                df[col].plot.hist(ax=ax, bins=30)
            else:
                df.select_dtypes(include="number").iloc[:, 0].plot.hist(ax=ax, bins=30)
        elif chart_type == "box":
            cols = [y_col] if y_col else None
            if cols:
                df[cols].plot.box(ax=ax)
            else:
                df.select_dtypes(include="number").iloc[:, :5].plot.box(ax=ax)
        elif chart_type == "pie":
            col = y_col or x_col
            if col:
                df[col].value_counts().head(10).plot.pie(ax=ax)
            else:
                return json.dumps({"error": "pie requires a column name"}, ensure_ascii=False)
        else:
            return json.dumps({"error": f"Unknown chart_type: {chart_type}"}, ensure_ascii=False)

        if title:
            ax.set_title(title)
        plt.tight_layout()
        fig.savefig(output, dpi=150)
        plt.close(fig)

        return json.dumps({
            "action": "chart_created",
            "chart_type": chart_type,
            "output": output,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def code_execute(args: Dict[str, Any], workspace: str) -> str:
    """Execute Python code for data analysis.

    Args: {code: str}
    """
    code = args.get("code", "")
    if not code:
        return json.dumps({"error": "code is required"}, ensure_ascii=False)

    # Run code in a separate thread with a timeout for cross-platform support
    import threading

    result_box: Dict[str, Any] = {}
    stdout_capture = io.StringIO()

    def _run():
        local_vars: Dict[str, Any] = {}

        if HAS_PANDAS:
            local_vars["pd"] = pd
        if HAS_MATPLOTLIB:
            local_vars["plt"] = plt

        try:
            import numpy as np
            local_vars["np"] = np
        except ImportError:
            pass

        # Restricted builtins
        _original_import = builtins.__import__
        BLOCKED_MODULES = {
            "os", "subprocess", "shutil", "ctypes", "socket", "sys",
            "importlib", "builtins", "signal", "multiprocessing",
            "threading", "pathlib",
        }

        def _restricted_import(name, *a, **kw):
            top_level = name.split(".")[0]
            if top_level in BLOCKED_MODULES:
                raise ImportError(f"Module '{name}' is not available in sandbox mode")
            return _original_import(name, *a, **kw)

        safe_builtins = {
            k: v for k, v in builtins.__dict__.items() if k not in (
                "exec", "eval", "compile", "__import__",
                "breakpoint", "exit", "quit", "open", "globals", "locals",
            )
        }
        safe_builtins["__import__"] = _restricted_import
        safe_builtins["print"] = print

        import sys
        old_stdout = sys.stdout
        sys.stdout = stdout_capture
        try:
            exec(code, {"__builtins__": safe_builtins}, local_vars)
        except Exception as e:
            result_box["error"] = f"{type(e).__name__}: {e}"
            return
        finally:
            sys.stdout = old_stdout

        _ensure_data_dir(workspace)

        result_vars = {}
        for name, value in local_vars.items():
            if name.startswith("_") or name in ("pd", "plt", "np"):
                continue
            try:
                json.dumps(value, default=str)
                result_vars[name] = value
            except (TypeError, ValueError):
                result_vars[name] = str(type(value))

        result_box["ok"] = {
            "stdout": stdout_capture.getvalue()[:5000],
            "variables": result_vars,
        }

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=120)

    if worker.is_alive():
        return json.dumps({
            "error": "Code execution timed out (120s limit)",
        }, ensure_ascii=False)

    if "ok" in result_box:
        return json.dumps(result_box["ok"], ensure_ascii=False, default=str)

    if "error" in result_box:
        return json.dumps({"error": result_box["error"]}, ensure_ascii=False)

    return json.dumps({"error": "Code execution produced no output"}, ensure_ascii=False)


def register_analysis_tools(registry, workspace: str):
    """Register data analysis tools."""
    from functools import partial

    registry.register(
        name="data_load",
        description=(
            "Load a dataset for analysis. "
            "Supports CSV, Excel (.xlsx/.xls), JSON, SPSS (.sav), Stata (.dta). "
            "Returns column names, types, null counts, and sample values."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to data file "
                        "(relative to workspace or absolute)"
                    ),
                },
            },
            "required": ["path"],
        },
        handler=partial(data_load, workspace=workspace),
    )

    registry.register(
        name="data_describe",
        description=(
            "Generate descriptive statistics for a dataset. "
            "Returns count, mean, std, min, max, quartiles for numeric columns."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to data file"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to describe (omit for all)",
                },
            },
            "required": ["path"],
        },
        handler=partial(data_describe, workspace=workspace),
    )

    registry.register(
        name="data_visualize",
        description=(
            "Generate charts from data. "
            "Chart types: bar, line, scatter, hist, box, pie."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to data file"},
                "chart_type": {
                    "type": "string",
                    "default": "bar",
                    "enum": ["bar", "line", "scatter", "hist", "box", "pie"],
                },
                "x": {"type": "string", "description": "Column for X axis"},
                "y": {"type": "string", "description": "Column for Y axis"},
                "title": {"type": "string", "description": "Chart title"},
                "output": {"type": "string", "description": "Output filename (default: chart.png)"},
            },
            "required": ["path"],
        },
        handler=partial(data_visualize, workspace=workspace),
    )

    registry.register(
        name="code_execute",
        description=(
            "Execute Python code for custom data analysis. "
            "pandas (pd), numpy (np), matplotlib (plt) are pre-imported. "
            "workspace variable contains the current workspace path."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
        handler=partial(code_execute, workspace=workspace),
    )
