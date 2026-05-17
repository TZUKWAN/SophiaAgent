"""Dependency manager: safe pip install with logging, conflict check, and lock."""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import importlib.util
from typing import List, Optional


# Mapping from PyPI package name → Python import name.
_IMPORT_NAME_MAP = {
    "scikit-learn": "sklearn",
    "scikit_learn": "sklearn",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "opencv-python": "cv2",
    "opencv_python": "cv2",
    "sentence-transformers": "sentence_transformers",
    "imbalanced-learn": "imblearn",
    "imbalanced_learn": "imblearn",
    "scikit-image": "skimage",
    "scikit_image": "skimage",
    "pytorch": "torch",
    "beautifulsoup4": "bs4",
    "python-dateutil": "dateutil",
    "python_dateutil": "dateutil",
}


class DependencyManager:
    SAFE_PACKAGES = {
        "scipy", "numpy", "pandas", "statsmodels", "pingouin", "researchpy",
        "scikit-learn", "matplotlib", "seaborn", "linearmodels", "pydoe2",
        "networkx", "nltk", "factor-analyzer", "dowhy", "causalimpact",
        "xgboost", "lightgbm", "optuna", "shap", "flaml",
        "gensim", "bertopic", "vadersentiment", "torch", "transformers",
        "sentence-transformers", "lifelines", "pymc", "girth", "semopy",
        "textblob", "spacy", "cmdstanpy", "prophet", "geopandas", "pillow",
        "simpy", "scikit-image", "sympy", "plotly", "bokeh", "altair",
        "arch", "pmdarima", "tslearn", "imbalanced-learn",
        "category-encoders", "feature-engine", "patsy", "formulaic",
        "pyyaml", "opencv-python", "beautifulsoup4", "python-dateutil",
        "jupyter", "ipykernel", "nbformat", "jsonschema", "requests",
        "httpx", "tqdm", "joblib", "cloudpickle", "dill", "lz4", "zstandard",
        "fastparquet", "pyarrow", "openpyxl", "xlrd", "xlsxwriter",
        "tabulate", "markdown", "jinja2", "mako", "weasyprint", "reportlab",
    }

    # Known aliases: pip name → canonical pip name for whitelist lookup.
    _PACKAGE_ALIASES = {
        "sklearn": "scikit-learn",
        "pil": "pillow",
        "cv2": "opencv-python",
        "pytorch": "torch",
    }

    _install_lock = threading.Lock()

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def install(self, args: dict) -> str:
        """Install a package.

        Args:
            args: dict with 'package' (str) and optional 'version' (str)

        Returns:
            JSON string with success, package, version
        """
        package = args.get("package", "").strip()
        version = args.get("version")

        if not package:
            return json.dumps({
                "success": False,
                "error": "No package name provided",
            }, ensure_ascii=False)

        canonical_pkg = self._canonical_name(package)
        in_whitelist = self._is_safe(canonical_pkg)

        result = self._do_install(package, version)
        result["whitelisted"] = in_whitelist

        if not in_whitelist and result.get("success"):
            result["warning"] = (
                f"Package '{package}' is not in the pre-approved whitelist. "
                "Review before using in production."
            )

        return json.dumps(result, ensure_ascii=False)

    def check_installed(self, args: dict) -> str:
        """Check if package is installed using importlib.metadata (no import side-effects).

        Args:
            args: dict with 'package' (str)

        Returns:
            JSON string with installed (bool), version
        """
        package = args.get("package", "").strip()
        if not package:
            return json.dumps({
                "package": "",
                "installed": False,
                "error": "No package name provided",
            }, ensure_ascii=False)

        canonical_pkg = self._canonical_name(package)
        version = self._get_version_by_metadata(canonical_pkg)
        if version is not None:
            return json.dumps({
                "package": package,
                "installed": True,
                "version": version,
            }, ensure_ascii=False)

        # Fallback to find_spec for packages where metadata may be missing.
        import_name = self._to_import_name(canonical_pkg)
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            return json.dumps({
                "package": package,
                "installed": True,
                "version": "unknown",
            }, ensure_ascii=False)

        return json.dumps({
            "package": package,
            "installed": False,
            "version": None,
        }, ensure_ascii=False)

    def safe_install(self, requirements: List[str]) -> str:
        """Batch install multiple packages with individual error isolation.

        Args:
            requirements: list of package spec strings (e.g. "numpy>=1.20", "pandas")

        Returns:
            JSON string with results per package
        """
        results = []
        for req in requirements:
            pkg_name, version, spec = self._parse_requirement(req)
            install_result = self._do_install(pkg_name, version, spec=spec)
            install_result["requirement"] = req
            results.append(install_result)

        successful = sum(1 for r in results if r.get("success"))
        return json.dumps({
            "total": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "results": results,
        }, ensure_ascii=False)

    def _do_install(self, package: str, version: str = None, spec: str = None) -> dict:
        """Internal: run pip install with lock, logging, and conflict check."""
        if not package:
            return {"success": False, "package": "", "error": "Empty package name"}

        canonical_pkg = self._canonical_name(package)
        install_spec = spec or (f"{package}=={version}" if version else package)

        # Acquire thread lock to prevent concurrent pip invocations.
        with self._install_lock:
            try:
                cmd = [sys.executable, "-m", "pip", "install", install_spec, "--quiet"]
                # Run in clean env to avoid PYTHONPATH pollution.
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    env=env,
                )

                if proc.returncode == 0:
                    installed_version = self._get_version_by_metadata(canonical_pkg) or version or "unknown"
                    result = {
                        "success": True,
                        "package": package,
                        "version": version or installed_version,
                        "installed_version": installed_version,
                    }
                    # Post-install dependency conflict check.
                    conflict = self._pip_check()
                    if conflict:
                        result["dependency_conflict"] = conflict
                    self._log_install(package, version, True, None)
                    return result
                else:
                    error_msg = proc.stderr.strip() if proc.stderr else "Unknown pip error"
                    self._log_install(package, version, False, error_msg)
                    return {
                        "success": False,
                        "package": package,
                        "error": error_msg,
                        "returncode": proc.returncode,
                    }
            except subprocess.TimeoutExpired:
                self._log_install(package, version, False, "Installation timed out after 180 seconds")
                return {
                    "success": False,
                    "package": package,
                    "error": "Installation timed out after 180 seconds",
                }
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)}"
                self._log_install(package, version, False, err)
                return {
                    "success": False,
                    "package": package,
                    "error": err,
                }

    def _is_safe(self, package: str) -> bool:
        """Check if package is in the safe whitelist."""
        canonical = self._canonical_name(package)
        # Also check alias mapping.
        aliased = self._PACKAGE_ALIASES.get(canonical, canonical)
        return aliased in self.SAFE_PACKAGES

    @staticmethod
    def _to_import_name(package: str) -> str:
        """Convert a PyPI package name to its Python import name."""
        normalized = package.lower().strip()
        if normalized in _IMPORT_NAME_MAP:
            return _IMPORT_NAME_MAP[normalized]
        # Strip extras, e.g. package[extra] → package
        base = re.sub(r"\[.*\]", "", normalized)
        return base.replace("-", "_")

    @staticmethod
    def _canonical_name(package: str) -> str:
        """Normalize package name: lower-case, strip extras, replace underscore with hyphen."""
        p = package.lower().strip()
        p = re.sub(r"\[.*\]", "", p)
        p = p.replace("_", "-")
        return p

    @staticmethod
    def _parse_requirement(req: str):
        """Parse a requirement string into (package_name, version, spec_for_pip).

        For == constraints version is extracted and spec is reconstructed.
        For >= / <= / > / < the original requirement (minus extras) is kept
        as spec so pip can resolve it correctly.
        """
        req = req.strip()
        # Strip extras first.
        base = re.sub(r"\[.*?\]", "", req).strip()
        for sep in ("==", ">=", "<=", ">", "<"):
            if sep in base:
                parts = base.split(sep, 1)
                pkg = parts[0].strip()
                ver = parts[1].strip()
                if sep == "==":
                    return pkg, ver, f"{pkg}=={ver}"
                # For inequalities keep the base constraint as spec.
                return pkg, ver, base
        return base, None, base

    @staticmethod
    def _get_version_by_metadata(package: str) -> Optional[str]:
        """Get installed version via importlib.metadata (no import side-effects)."""
        try:
            from importlib.metadata import version as md_version
            return md_version(package)
        except Exception:
            return None

    def _pip_check(self) -> Optional[str]:
        """Run 'pip check' and return any conflict message, or None."""
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "check", "--quiet"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode != 0 and proc.stdout:
                return proc.stdout.strip()
            return None
        except Exception:
            return None

    def _log_install(self, package: str, version: Optional[str], success: bool, error: Optional[str]):
        """Append install operation to log file."""
        if not self.log_dir:
            return
        import datetime
        log_path = os.path.join(self.log_dir, "install.log")
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        status = "OK" if success else "FAIL"
        ver = version or "latest"
        err = f" | error={error}" if error else ""
        line = f"{ts} {status} {package}=={ver}{err}\n"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
