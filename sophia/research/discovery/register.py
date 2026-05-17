"""Register self-evolving discovery tools into the ToolRegistry."""
import json
import traceback
from typing import Any, Callable, Dict

from sophia.research.discovery.sandbox import HandlerSandbox, SandboxViolation
from sophia.tools.registry import ToolRegistry


def _activate_method(registry, method):
    """Activate a single method into the registry by safely exec-ing its handler_code."""
    hc = method.get("handler_code")
    ts = method.get("tool_schema")
    tn = method.get("tool_name")
    if not all([hc, ts, tn]):
        return False
    try:
        ns = HandlerSandbox.exec_safe(
            hc,
            {"json": json, "traceback": traceback, "__builtins__": __builtins__},
        )
        fn = ns.get("handle")
        if fn is None or not callable(fn):
            return False
        registry.register(
            tn,
            ts.get("description", method.get("description", "")),
            ts.get("parameters", {}),
            fn,
        )
        return True
    except SandboxViolation:
        return False
    except Exception:
        return False


def register_discovery_tools(registry: ToolRegistry, components: dict):
    """Register method discovery tools.

    components = {
        "catalog": MethodCatalog,
        "searcher": MethodSearcher,
        "builder": MethodBuilder,
        "dep_manager": DependencyManager,
    }
    """
    catalog = components["catalog"]
    searcher = components["searcher"]
    builder = components["builder"]
    dep_mgr = components["dep_manager"]

    # =====================================================================
    # method_search
    # =====================================================================
    def _method_search(args):
        return searcher.search(
            args.get("description", ""),
            category=args.get("category"),
        )

    registry.register(
        "method_search",
        "Search for research methods in the catalog and external sources. "
        "Returns installed methods, known methods, or external candidates.",
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description of the research method to find",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter (statistics, causal, ml, etc.)",
                },
            },
            "required": ["description"],
        },
        _method_search,
    )

    # =====================================================================
    # method_install
    # =====================================================================
    def _method_install(args):
        package = args.get("package", "").strip()
        version = args.get("version")
        if not package:
            return json.dumps({
                "success": False,
                "error": "No package name provided",
            }, ensure_ascii=False)
        return dep_mgr.install({"package": package, "version": version})

    registry.register(
        "method_install",
        "Install a Python package for a research method. Checks whitelist and verifies installation.",
        {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package name to install (e.g. 'girth', 'semopy')",
                },
                "version": {
                    "type": "string",
                    "description": "Optional version constraint (e.g. '1.0.0')",
                },
            },
            "required": ["package"],
        },
        _method_install,
    )

    # =====================================================================
    # method_auto_discover
    # =====================================================================
    def _method_auto_discover(args):
        """Full pipeline: search -> install deps -> build -> register -> activate."""
        description = args.get("description", "").strip()
        category = args.get("category")
        user_context = args.get("context", description)

        if not description:
            return json.dumps({
                "success": False,
                "error": "No method description provided",
            }, ensure_ascii=False)

        steps = []

        # Step 1: Search for candidates
        search_result_str = searcher.search(description, category=category)
        try:
            search_result = json.loads(search_result_str)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({
                "success": False,
                "error": "Search phase failed: invalid JSON returned",
            }, ensure_ascii=False)

        steps.append({"step": "search", "result": search_result.get("message", "done")})

        # If already installed, return directly
        if search_result.get("source") == "catalog" and search_result.get("status") == "installed":
            search_result["steps"] = steps
            return json.dumps(search_result, ensure_ascii=False)

        candidates = search_result.get("candidates", [])
        if not candidates:
            # Also check methods list from catalog search
            methods = search_result.get("methods", [])
            if methods:
                search_result["steps"] = steps
                return json.dumps(search_result, ensure_ascii=False)
            return json.dumps({
                "success": False,
                "error": f"No candidates found for '{description}'",
                "steps": steps,
            }, ensure_ascii=False)

        # Pick the best candidate:
        # 1. Prefer candidate whose library/pip_name matches words in user description.
        # 2. Fallback to first importable candidate.
        # 3. Fallback to first candidate overall.
        desc_lower = description.lower()
        best = None
        for c in candidates:
            lib = (c.get("library") or "").lower()
            pip = (c.get("pip_name") or "").lower()
            if lib in desc_lower or pip in desc_lower:
                best = c
                break
        if best is None:
            for c in candidates:
                if c.get("importable"):
                    best = c
                    break
        if best is None:
            best = candidates[0]

        steps.append({"step": "select_candidate", "library": best.get("library")})

        # Step 2: Install dependencies if needed
        if not best.get("importable"):
            pip_name = best.get("pip_name", best.get("library", ""))
            install_result_str = dep_mgr.install({"package": pip_name})
            try:
                install_result = json.loads(install_result_str)
            except (json.JSONDecodeError, TypeError):
                install_result = {"success": False, "error": "Install returned invalid JSON"}
            steps.append({"step": "install", "result": install_result})

            if not install_result.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to install dependency '{pip_name}': {install_result.get('error')}",
                    "steps": steps,
                }, ensure_ascii=False)
        else:
            steps.append({"step": "install", "result": "already installed, skipped"})

        # Step 3: Build the method
        build_result_str = builder.build(best, user_context)
        try:
            build_result = json.loads(build_result_str)
        except (json.JSONDecodeError, TypeError):
            build_result = {"success": False, "error": "Build returned invalid JSON"}

        steps.append({"step": "build", "result": build_result.get("message", build_result.get("error"))})

        if not build_result.get("success"):
            return json.dumps({
                "success": False,
                "error": f"Build failed: {build_result.get('error')}",
                "steps": steps,
            }, ensure_ascii=False)

        # Step 4: Activate (register into tool registry)
        tool_name = build_result.get("tool_name")
        method_id = build_result.get("method_id")
        activated = 0

        if tool_name and method_id:
            method = catalog.get(method_id)
            if method and method.get("handler_code") and method.get("tool_schema"):
                if _activate_method(registry, method):
                    activated = 1

        steps.append({"step": "activate", "activated": activated})

        return json.dumps({
            "success": True,
            "method_id": method_id,
            "tool_name": tool_name,
            "library": best.get("library"),
            "steps": steps,
            "message": f"Method '{description}' auto-discovered and {'activated' if activated else 'built but not activated'}",
        }, ensure_ascii=False)

    registry.register(
        "method_auto_discover",
        "Full auto-discovery pipeline: search for a method, install dependencies, "
        "build handler, and register as an active tool. One-call method expansion.",
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description of the research method to discover and install",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter",
                },
                "context": {
                    "type": "string",
                    "description": "User context or requirement for the method",
                },
            },
            "required": ["description"],
        },
        _method_auto_discover,
    )

    # =====================================================================
    # method_list_available
    # =====================================================================
    def _method_list_available(args):
        category = args.get("category")
        status = args.get("status")
        source = args.get("source")
        methods = catalog.list_methods(category=category, status=status, source=source)
        stats = catalog.get_stats()
        return json.dumps({
            "total": len(methods),
            "stats": stats,
            "methods": methods,
        }, ensure_ascii=False)

    registry.register(
        "method_list_available",
        "List all available methods with optional category, status, or source filters.",
        {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (statistics, causal, ml, etc.)",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status (installed, known, candidate)",
                },
                "source": {
                    "type": "string",
                    "description": "Filter by source (builtin, auto_discovered, user)",
                },
            },
            "required": [],
        },
        _method_list_available,
    )

    # =====================================================================
    # method_verify
    # =====================================================================
    def _method_verify(args):
        method_id = args.get("method_id", "").strip()
        tool_name = args.get("tool_name", "").strip()

        if not method_id and not tool_name:
            return json.dumps({
                "success": False,
                "error": "Provide method_id or tool_name",
            }, ensure_ascii=False)

        # Find method by ID or tool name
        method = None
        if method_id:
            method = catalog.get(method_id)
        if method is None and tool_name:
            method = catalog.get_by_tool(tool_name)

        if not method:
            return json.dumps({
                "success": False,
                "valid": False,
                "error": f"Method not found",
            }, ensure_ascii=False)

        mid = method.get("id", method_id)

        handler_code = method.get("handler_code")
        tool_schema = method.get("tool_schema")

        # Built-in methods without handler code are considered valid
        if not handler_code:
            return json.dumps({
                "success": True,
                "valid": True,
                "method_id": mid,
                "note": "Built-in method, no handler code to verify",
            }, ensure_ascii=False)

        # Validate syntax and sandbox policy
        from sophia.research.discovery.sandbox import HandlerSandbox, SandboxViolation
        sandbox_ok = False
        sandbox_error = None
        try:
            sandbox_ok, sandbox_error = HandlerSandbox.validate(handler_code)
        except Exception as e:
            sandbox_error = str(e)

        syntax_ok = False
        syntax_error = None
        if sandbox_ok:
            try:
                compile(handler_code, "<handler>", "exec")
                syntax_ok = True
            except SyntaxError as e:
                syntax_error = str(e)

        # Validate exec and handle function
        exec_ok = False
        exec_error = None
        handle_callable = False

        if syntax_ok:
            try:
                local_ns = HandlerSandbox.exec_safe(
                    handler_code,
                    {"json": json, "traceback": traceback, "__builtins__": __builtins__},
                )
                handle_fn = local_ns.get("handle")
                handle_callable = callable(handle_fn) if handle_fn else False
                exec_ok = True
            except SandboxViolation as e:
                exec_error = f"Sandbox violation: {e}"
            except Exception as e:
                exec_error = str(e)

        # Validate schema
        schema_valid = isinstance(tool_schema, dict) and "parameters" in tool_schema

        # Overall verdict
        valid = sandbox_ok and syntax_ok and exec_ok and handle_callable
        issues = []
        if not sandbox_ok:
            issues.append(f"Sandbox violation: {sandbox_error}")
        if not syntax_ok:
            issues.append(f"Syntax error: {syntax_error}")
        if not exec_ok:
            issues.append(f"Exec error: {exec_error}")
        if not handle_callable:
            issues.append("No callable 'handle' function found")
        if not schema_valid:
            issues.append("Invalid or missing tool schema")

        # Update catalog verification status
        catalog.update(mid, verified=int(valid), last_error="; ".join(issues) if issues else None)

        return json.dumps({
            "success": True,
            "method_id": mid,
            "valid": valid,
            "syntax_ok": syntax_ok,
            "exec_ok": exec_ok,
            "handle_callable": handle_callable,
            "schema_valid": schema_valid,
            "issues": issues if issues else None,
        }, ensure_ascii=False)

    registry.register(
        "method_verify",
        "Verify a method's handler code and tool schema. Checks syntax, exec, and schema validity.",
        {
            "type": "object",
            "properties": {
                "method_id": {
                    "type": "string",
                    "description": "ID of the method to verify",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Tool name (alternative to method_id)",
                },
            },
            "required": [],
        },
        _method_verify,
    )
