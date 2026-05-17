"""Tests for HandlerSandbox: AST-based code safety validation."""

import pytest

from sophia.research.discovery.sandbox import HandlerSandbox, SandboxViolation


class TestSandboxValidation:
    def test_valid_handler_passes(self):
        code = (
            "import json\n"
            "import traceback\n"
            "def handle(args):\n"
            "    return json.dumps({'ok': True})\n"
        )
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is True
        assert error is None

    def test_eval_blocked(self):
        code = "def handle(args):\n    return eval(args['x'])"
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is False
        assert "eval" in error.lower()

    def test_exec_blocked(self):
        code = "def handle(args):\n    exec('print(1)')"
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is False
        assert "exec" in error.lower()

    def test_import_os_blocked(self):
        code = "import os\ndef handle(args):\n    return os.getcwd()"
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is False
        assert "os" in error.lower()

    def test_import_subprocess_blocked(self):
        code = "import subprocess\ndef handle(args):\n    return 'ok'"
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is False
        assert "subprocess" in error.lower()

    def test_dangerous_attribute_blocked(self):
        code = (
            "def handle(args):\n"
            "    import os\n"
            "    return os.system('ls')\n"
        )
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is False

    def test_private_attribute_blocked(self):
        code = "def handle(args):\n    return args.__class__"
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is False
        assert "__class__" in error.lower()

    def test_class_def_blocked(self):
        code = "class Bad:\n    pass\ndef handle(args):\n    return 'ok'"
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is False
        assert "ClassDef" in error

    def test_safe_import_allowed(self):
        code = (
            "import json\n"
            "import numpy\n"
            "def handle(args):\n"
            "    return json.dumps({'mean': float(numpy.mean([1,2,3]))})\n"
        )
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is True

    def test_pandas_import_allowed(self):
        code = (
            "import pandas\n"
            "def handle(args):\n"
            "    df = pandas.DataFrame({'a': [1, 2]})\n"
            "    return df.to_json()\n"
        )
        is_safe, error = HandlerSandbox.validate(code)
        assert is_safe is True

    def test_empty_code_fails(self):
        is_safe, error = HandlerSandbox.validate("")
        assert is_safe is False

    def test_none_code_fails(self):
        is_safe, error = HandlerSandbox.validate(None)
        assert is_safe is False


class TestSandboxExecSafe:
    def test_exec_safe_runs_code(self):
        code = (
            "import json\n"
            "def handle(args):\n"
            "    return json.dumps({'result': args['x'] * 2})\n"
        )
        ns = HandlerSandbox.exec_safe(code)
        assert "handle" in ns
        result = ns["handle"]({"x": 5})
        assert "10" in result

    def test_exec_safe_raises_on_violation(self):
        code = "def handle(args):\n    return eval('1+1')"
        with pytest.raises(SandboxViolation):
            HandlerSandbox.exec_safe(code)

    def test_exec_safe_with_custom_globals(self):
        code = (
            "def handle(args):\n"
            "    return json.dumps({'ok': True})\n"
        )
        ns = HandlerSandbox.exec_safe(code, {"json": __import__("json")})
        assert "handle" in ns

    def test_exec_safe_returns_empty_locals_for_no_bindings(self):
        code = "x = 1"
        ns = HandlerSandbox.exec_safe(code)
        assert ns == {"x": 1}
