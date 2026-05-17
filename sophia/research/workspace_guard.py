"""Workspace isolation guard for all file operations."""
import os


class WorkspaceGuard:
    def __init__(self, workspace: str):
        self.workspace = os.path.realpath(workspace)
        os.makedirs(os.path.join(self.workspace, ".research", "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace, ".research", "cache"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace, ".research", "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace, ".research", "models"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace, ".research", "discovery"), exist_ok=True)

    def resolve_read(self, path: str) -> str:
        real = os.path.realpath(os.path.join(self.workspace, path))
        if not real.startswith(self.workspace + os.sep) and real != self.workspace:
            raise PermissionError(f"Path escapes workspace: {path}")
        if not os.path.exists(real):
            raise FileNotFoundError(f"File not found: {path}")
        return real

    def resolve_write(self, path: str, subdir: str = "") -> str:
        base = os.path.join(self.workspace, ".research", subdir) if subdir else self.workspace
        real = os.path.realpath(os.path.join(base, path))
        if not real.startswith(self.workspace + os.sep):
            raise PermissionError(f"Path escapes workspace: {path}")
        os.makedirs(os.path.dirname(real), exist_ok=True)
        return real

    def is_within(self, path: str) -> bool:
        real = os.path.realpath(path)
        return real.startswith(self.workspace + os.sep) or real == self.workspace

    def figures_dir(self) -> str:
        return os.path.join(self.workspace, ".research", "figures")

    def cache_dir(self) -> str:
        return os.path.join(self.workspace, ".research", "cache")

    def reports_dir(self) -> str:
        return os.path.join(self.workspace, ".research", "reports")

    def models_dir(self) -> str:
        return os.path.join(self.workspace, ".research", "models")

    def discovery_dir(self) -> str:
        return os.path.join(self.workspace, ".research", "discovery")
