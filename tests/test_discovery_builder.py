"""Tests for MethodBuilder: handler generation, schema, validation."""
import json
import pytest
from sophia.research.discovery.method_catalog import MethodCatalog
from sophia.research.discovery.method_builder import MethodBuilder


@pytest.fixture
def builder(tmp_path):
    catalog = MethodCatalog(str(tmp_path / "test.db"))
    return MethodBuilder(catalog)


class TestBuildMethod:
    def test_build_with_importable_library(self, builder):
        candidate = {
            "name": "Numpy Test",
            "library": "numpy",
            "pip_name": "numpy",
            "description": "Test numpy wrapper",
            "category": "statistics",
        }
        result = builder.build(candidate, "test context")
        data = json.loads(result)
        assert data["success"] is True
        assert data["method_id"] is not None
        assert data["tool_name"] is not None

    def test_build_with_missing_library(self, builder):
        candidate = {
            "name": "Missing",
            "library": "nonexistent_lib_xyz_12345",
            "description": "Test missing",
            "category": "test",
        }
        result = builder.build(candidate, "test")
        data = json.loads(result)
        # Builder generates template handler with try/except ImportError,
        # so build succeeds even for missing libs. The handler will report
        # the missing dependency at runtime.
        assert data["success"] is True

    def test_build_empty_library(self, builder):
        candidate = {"name": "Empty", "library": "", "description": "Empty"}
        result = builder.build(candidate, "test")
        data = json.loads(result)
        assert data["success"] is False


class TestHandlerGeneration:
    def test_generates_valid_python(self, builder):
        code = builder._generate_handler("scipy", "scipy stats", "do a t-test")
        assert code is not None
        assert "def handle" in code
        assert "import scipy" in code

    def test_handler_is_validatable(self, builder):
        code = builder._generate_handler("numpy", "numpy", "array ops")
        assert builder._validate(code, {})


class TestSchemaGeneration:
    def test_generates_schema(self, builder):
        schema = builder._generate_schema("test_method", "A test method")
        assert schema is not None
        assert "description" in schema
        assert "parameters" in schema


class TestValidation:
    def test_validate_good_code(self, builder):
        code = 'import json\ndef handle(args): return json.dumps({"ok": True})'
        assert builder._validate(code, {}) is True

    def test_validate_bad_syntax(self, builder):
        code = "def handle(args: return "  # syntax error
        assert builder._validate(code, {}) is False

    def test_validate_no_handle(self, builder):
        code = "import json\nx = 1"
        assert builder._validate(code, {}) is False

    def test_validate_empty(self, builder):
        assert builder._validate("", {}) is False
        assert builder._validate(None, {}) is False


class MockProvider:
    """Mock LLM provider that returns predefined responses."""
    def __init__(self, response_text: str):
        self.response_text = response_text

    def chat(self, messages, tools=None):
        from sophia.providers.base import ProviderResponse
        return ProviderResponse(content=self.response_text)


class TestFunctionalHandlers:
    """Verify that category-aware templates produce handlers with real functionality."""

    def test_numpy_handler_computes_stats(self, builder):
        code = builder._generate_handler("numpy", "Descriptive statistics", "compute mean")
        assert code is not None
        # Execute via sandbox and verify real computation
        from sophia.research.discovery.sandbox import HandlerSandbox
        ns = HandlerSandbox.exec_safe(code)
        handle = ns["handle"]
        result = json.loads(handle({"data": [1, 2, 3, 4, 5]}))
        assert result["status"] == "success"
        assert abs(result["mean"] - 3.0) < 0.01
        assert abs(result["std"] - 1.58) < 0.1
        assert result["median"] == 3.0

    def test_scipy_handler_ttest(self, builder):
        code = builder._generate_handler("scipy", "T-test", "compare two groups")
        assert code is not None
        from sophia.research.discovery.sandbox import HandlerSandbox
        ns = HandlerSandbox.exec_safe(code)
        handle = ns["handle"]
        result = json.loads(handle({
            "data": {"group_a": [1, 2, 3, 4, 5], "group_b": [6, 7, 8, 9, 10]},
            "method": "ttest_ind",
        }))
        assert result["status"] == "success"
        assert "t_statistic" in result
        assert result["significant"] is True

    def test_pandas_handler_describe(self, builder):
        code = builder._generate_handler("pandas", "Data description", "describe a dataset")
        assert code is not None
        from sophia.research.discovery.sandbox import HandlerSandbox
        ns = HandlerSandbox.exec_safe(code)
        handle = ns["handle"]
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
        result = json.loads(handle({"data": records, "method": "describe"}))
        assert result["status"] == "success"
        assert result["shape"] == [3, 2]
        assert "describe" in result

    def test_sklearn_handler_train(self, builder):
        code = builder._generate_handler("sklearn", "ML training", "train a classifier")
        assert code is not None
        from sophia.research.discovery.sandbox import HandlerSandbox
        ns = HandlerSandbox.exec_safe(code)
        handle = ns["handle"]
        result = json.loads(handle({
            "data": {
                "X": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
                "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            },
            "parameters": {"model": "LogisticRegression", "random_state": 42},
        }))
        assert result["status"] == "success"
        assert "train_score" in result
        assert "test_score" in result

    def test_generic_handler_returns_info(self, builder):
        # Use 'json' (builtin, not in template map) so import succeeds
        code = builder._generate_handler("json", "Unknown", "do something")
        assert code is not None
        from sophia.research.discovery.sandbox import HandlerSandbox
        ns = HandlerSandbox.exec_safe(code)
        handle = ns["handle"]
        result = json.loads(handle({"data": []}))
        assert result["status"] == "info"
        assert result["library"] == "json"
        assert "available_functions" in result


class TestLLMHandlerGeneration:
    def test_llm_generates_handler(self, tmp_path):
        catalog = MethodCatalog(str(tmp_path / "test_llm.db"))
        handler_code = (
            "import json\n"
            "import traceback\n"
            "def handle(args):\n"
            "    try:\n"
            "        import numpy\n"
            "        data = args.get('data', [])\n"
            "        result = {'mean': float(numpy.mean(data))}\n"
            "        return json.dumps(result, ensure_ascii=False, default=str)\n"
            "    except Exception as e:\n"
            "        return json.dumps({'error': str(e)}, ensure_ascii=False)\n"
        )
        provider = MockProvider(handler_code)
        builder = MethodBuilder(catalog, provider=provider)

        candidate = {
            "name": "Numpy Mean",
            "library": "numpy",
            "pip_name": "numpy",
            "description": "Compute mean with numpy",
            "category": "statistics",
        }
        result = builder.build(candidate, "test context")
        data = json.loads(result)
        assert data["success"] is True
        assert data["tool_name"] is not None

        # Verify the built method can be loaded and executed
        method_id = data["method_id"]
        method = catalog.get(method_id)
        assert method is not None
        assert method["handler_code"] is not None

        # Execute handler and verify it returns valid JSON
        local_ns = {}
        exec(method["handler_code"], {"json": json, "traceback": __import__("traceback"), "__builtins__": __builtins__}, local_ns)
        handle = local_ns["handle"]
        output = handle({"data": [1, 2, 3, 4, 5]})
        parsed = json.loads(output)
        assert "mean" in parsed
        assert abs(parsed["mean"] - 3.0) < 0.01

    def test_llm_bad_code_falls_back_to_template(self, tmp_path):
        catalog = MethodCatalog(str(tmp_path / "test_llm2.db"))
        provider = MockProvider("def handle(args\n  return broken")  # syntax error
        builder = MethodBuilder(catalog, provider=provider)

        candidate = {
            "name": "Bad LLM",
            "library": "json",
            "description": "Should fallback to template",
            "category": "test",
        }
        result = builder.build(candidate, "test")
        data = json.loads(result)
        # Falls back to template, which is syntactically valid
        assert data["success"] is True
