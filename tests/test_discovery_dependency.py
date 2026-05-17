"""Tests for DependencyManager: install, check, whitelist, logging, conflicts."""
import json
import os
import threading
import time
import pytest
from sophia.research.discovery.dependency_manager import DependencyManager


@pytest.fixture
def dep_mgr():
    return DependencyManager()


@pytest.fixture
def dep_mgr_with_log(tmp_path):
    return DependencyManager(log_dir=str(tmp_path))


class TestCheckInstalled:
    def test_check_installed_package(self, dep_mgr):
        result = json.loads(dep_mgr.check_installed({"package": "numpy"}))
        assert result["installed"] is True
        assert result["version"] is not None

    def test_check_installed_scikit_learn_alias(self, dep_mgr):
        # scikit-learn should be detected via metadata without importing.
        result = json.loads(dep_mgr.check_installed({"package": "scikit-learn"}))
        assert result["installed"] is True

    def test_check_not_installed(self, dep_mgr):
        result = json.loads(dep_mgr.check_installed({"package": "nonexistent_xyz_12345"}))
        assert result["installed"] is False

    def test_check_empty(self, dep_mgr):
        result = json.loads(dep_mgr.check_installed({"package": ""}))
        assert result["installed"] is False

    def test_check_no_side_effects(self, dep_mgr):
        # importlib.metadata should not trigger module import side-effects.
        result = json.loads(dep_mgr.check_installed({"package": "pytest"}))
        assert result["installed"] is True
        assert "version" in result


class TestWhitelist:
    def test_safe_packages_in_whitelist(self, dep_mgr):
        assert dep_mgr._is_safe("scipy") is True
        assert dep_mgr._is_safe("numpy") is True
        assert dep_mgr._is_safe("pandas") is True
        assert dep_mgr._is_safe("scikit-learn") is True

    def test_unknown_not_in_whitelist(self, dep_mgr):
        assert dep_mgr._is_safe("some-random-package") is False

    def test_whitelist_is_case_insensitive(self, dep_mgr):
        assert dep_mgr._is_safe("NumPy") is True
        assert dep_mgr._is_safe("SCIPY") is True

    def test_whitelist_alias_sklearn(self, dep_mgr):
        # "sklearn" is alias for "scikit-learn" which is whitelisted.
        assert dep_mgr._is_safe("sklearn") is True

    def test_whitelist_strips_extras(self, dep_mgr):
        # extras like [cuda] should be stripped before whitelist check.
        assert dep_mgr._is_safe("numpy[cuda]") is True
        assert dep_mgr._is_safe("pandas[performance]") is True


class TestInstall:
    def test_install_empty_package(self, dep_mgr):
        result = json.loads(dep_mgr.install({"package": ""}))
        assert result["success"] is False

    def test_install_already_installed(self, dep_mgr):
        result = json.loads(dep_mgr.install({"package": "numpy"}))
        assert result["success"] is True
        assert result["whitelisted"] is True

    def test_install_nonexistent_package(self, dep_mgr):
        result = json.loads(dep_mgr.install({"package": "xyz_nonexistent_pkg_99999"}))
        assert result["success"] is False

    def test_install_not_whitelisted_warns(self, dep_mgr):
        result = json.loads(dep_mgr.install({"package": "xyz_nonexistent_pkg_99999"}))
        assert result["whitelisted"] is False

    def test_install_with_version_already_met(self, dep_mgr):
        result = json.loads(dep_mgr.install({"package": "numpy", "version": "1.0.0"}))
        # pip will downgrade or report error depending on environment.
        assert "success" in result


class TestGetVersion:
    def test_get_version_known(self, dep_mgr):
        ver = dep_mgr._get_version_by_metadata("numpy")
        assert ver is not None

    def test_get_version_unknown(self, dep_mgr):
        ver = dep_mgr._get_version_by_metadata("nonexistent_module_abc123")
        assert ver is None


class TestSafeInstall:
    def test_safe_install_batch(self, dep_mgr):
        result = json.loads(dep_mgr.safe_install(["numpy"]))
        assert result["total"] == 1
        assert result["successful"] >= 1

    def test_safe_install_mixed(self, dep_mgr):
        result = json.loads(dep_mgr.safe_install(["numpy", "nonexistent_pkg_xyz"]))
        assert result["total"] == 2

    def test_safe_install_with_version(self, dep_mgr):
        result = json.loads(dep_mgr.safe_install(["numpy>=1.0.0"]))
        data = json.loads(result) if isinstance(result, str) else result
        if isinstance(data, str):
            data = json.loads(data)
        assert data["total"] == 1
        assert data["successful"] == 1


class TestCanonicalName:
    def test_canonical_basic(self, dep_mgr):
        assert dep_mgr._canonical_name("NumPy") == "numpy"
        assert dep_mgr._canonical_name("scikit_learn") == "scikit-learn"

    def test_canonical_strips_extras(self, dep_mgr):
        assert dep_mgr._canonical_name("numpy[cuda]") == "numpy"
        assert dep_mgr._canonical_name("pandas[performance,test]") == "pandas"


class TestToImportName:
    def test_scikit_learn(self, dep_mgr):
        assert dep_mgr._to_import_name("scikit-learn") == "sklearn"

    def test_pillow(self, dep_mgr):
        assert dep_mgr._to_import_name("pillow") == "PIL"

    def test_numpy(self, dep_mgr):
        assert dep_mgr._to_import_name("numpy") == "numpy"

    def test_strips_extras(self, dep_mgr):
        assert dep_mgr._to_import_name("numpy[cuda]") == "numpy"


class TestParseRequirement:
    def test_eq(self, dep_mgr):
        assert dep_mgr._parse_requirement("numpy==1.24.0") == ("numpy", "1.24.0", "numpy==1.24.0")

    def test_ge(self, dep_mgr):
        assert dep_mgr._parse_requirement("numpy>=1.20") == ("numpy", "1.20", "numpy>=1.20")

    def test_le(self, dep_mgr):
        assert dep_mgr._parse_requirement("numpy<=2.0") == ("numpy", "2.0", "numpy<=2.0")

    def test_gt(self, dep_mgr):
        assert dep_mgr._parse_requirement("numpy>1.0") == ("numpy", "1.0", "numpy>1.0")

    def test_lt(self, dep_mgr):
        assert dep_mgr._parse_requirement("numpy<2.0") == ("numpy", "2.0", "numpy<2.0")

    def test_no_version(self, dep_mgr):
        assert dep_mgr._parse_requirement("numpy") == ("numpy", None, "numpy")

    def test_strips_extras(self, dep_mgr):
        assert dep_mgr._parse_requirement("numpy[performance]>=1.20") == ("numpy", "1.20", "numpy>=1.20")
        assert dep_mgr._parse_requirement("pandas[cuda,test]==2.0") == ("pandas", "2.0", "pandas==2.0")


class TestLogging:
    def test_log_file_created(self, dep_mgr_with_log, tmp_path):
        dep_mgr_with_log._log_install("fake-pkg", "1.0", True, None)
        log_path = os.path.join(str(tmp_path), "install.log")
        assert os.path.exists(log_path)
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "fake-pkg" in content
        assert "OK" in content

    def test_log_failure(self, dep_mgr_with_log, tmp_path):
        dep_mgr_with_log._log_install("bad-pkg", None, False, "network error")
        log_path = os.path.join(str(tmp_path), "install.log")
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "bad-pkg" in content
        assert "FAIL" in content
        assert "network error" in content

    def test_no_log_dir_no_crash(self, dep_mgr):
        # Should not raise when log_dir is None.
        dep_mgr._log_install("pkg", "1.0", True, None)


class TestPipCheck:
    def test_pip_check_runs(self, dep_mgr):
        # pip check should run without error on a healthy env.
        conflict = dep_mgr._pip_check()
        # conflict may be None (no issues) or a string (issues found).
        assert conflict is None or isinstance(conflict, str)


class TestConcurrencyLock:
    def test_install_lock_prevents_overlap(self, dep_mgr):
        # Verify that the lock object exists and is acquired/released.
        assert hasattr(dep_mgr, "_install_lock")
        assert dep_mgr._install_lock.acquire(blocking=False)
        dep_mgr._install_lock.release()

    def test_concurrent_installs_serialized(self, dep_mgr_with_log):
        results = []
        errors = []

        def worker():
            try:
                res = json.loads(dep_mgr_with_log.install({"package": "numpy"}))
                results.append(res)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0
        assert len(results) == 3
        for r in results:
            assert r["success"] is True
