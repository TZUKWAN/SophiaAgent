"""File operations tool for SophiaAgent."""

import json
import os
from pathlib import Path
from typing import Any, Dict


def _resolve_path(path: str, workspace: str) -> str:
    """Resolve a path relative to workspace, preventing traversal."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(workspace) / p
    resolved = p.resolve()
    ws = Path(workspace).resolve()
    if not str(resolved).startswith(str(ws)):
        return ""
    return str(resolved)


def file_read(args: Dict[str, Any], workspace: str) -> str:
    """Read file content.

    Args schema: {path: str, encoding: str, max_lines: int}
    """
    path = _resolve_path(args.get("path", ""), workspace)
    if not path:
        return json.dumps({"error": "Invalid or unsafe path"}, ensure_ascii=False)

    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"}, ensure_ascii=False)

    encoding = args.get("encoding", "utf-8")
    max_lines = args.get("max_lines", 2000)

    try:
        with open(path, "r", encoding=encoding) as f:
            lines = f.readlines()
        content = "".join(lines[:max_lines])
        if len(lines) > max_lines:
            content += f"\n... (truncated, {len(lines) - max_lines} more lines)"
        return json.dumps(
            {"path": path, "content": content, "lines": len(lines)},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def file_write(args: Dict[str, Any], workspace: str) -> str:
    """Write content to a file.

    Args schema: {path: str, content: str, encoding: str, append: bool}
    """
    path = _resolve_path(args.get("path", ""), workspace)
    if not path:
        return json.dumps({"error": "Invalid or unsafe path"}, ensure_ascii=False)

    content = args.get("content", "")
    encoding = args.get("encoding", "utf-8")
    append = args.get("append", False)

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding=encoding) as f:
            f.write(content)
        return json.dumps(
            {"path": path, "bytes_written": len(content.encode(encoding))},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def file_list(args: Dict[str, Any], workspace: str) -> str:
    """List files in a directory.

    Args schema: {path: str, pattern: str}
    """
    path = _resolve_path(args.get("path", "."), workspace)
    if not path:
        return json.dumps({"error": "Invalid or unsafe path"}, ensure_ascii=False)

    pattern = args.get("pattern", "*")

    try:
        p = Path(path)
        if not p.is_dir():
            return json.dumps({"error": f"Not a directory: {path}"}, ensure_ascii=False)

        entries = []
        for entry in sorted(p.glob(pattern)):
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else 0,
            })
        return json.dumps({"path": path, "entries": entries}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def register_file_tools(registry, workspace: str):
    """Register all file tools with the given registry."""
    from functools import partial

    registry.register(
        name="file_read",
        description="Read the content of a file. Returns file content with line count.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to read",
                    "default": 2000,
                },
            },
            "required": ["path"],
        },
        handler=partial(file_read, workspace=workspace),
    )

    registry.register(
        name="file_write",
        description="Write content to a file. Creates parent directories if needed.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                "append": {
                    "type": "boolean",
                    "description": "Append to file instead of overwrite",
                    "default": False,
                },
            },
            "required": ["path", "content"],
        },
        handler=partial(file_write, workspace=workspace),
    )

    registry.register(
        name="file_list",
        description="List files and directories at the given path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path", "default": "."},
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter",
                    "default": "*",
                },
            },
            "required": [],
        },
        handler=partial(file_list, workspace=workspace),
    )
