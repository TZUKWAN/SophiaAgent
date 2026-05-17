"""AST-based sandbox for safely executing dynamically generated handler code.

Validates that code only uses safe AST nodes and does not call dangerous
builtins or access private attributes.  Designed for the MethodBuilder / Discovery
pipeline where LLM-generated code may be exec'd at runtime.
"""

import ast
import json
from typing import List, Optional, Set, Tuple


# AST node types that are considered safe for research handler code.
_ALLOWED_NODES: Set[str] = {
    # Module structure
    "Module", "Interactive", "Expression",
    # Functions
    "FunctionDef",
    # Arguments
    "arguments", "arg",
    # Statements
    "Return", "Expr", "Assign", "AugAssign", "AnnAssign", "Pass",
    "If", "For", "While", "Break", "Continue",
    "Try", "ExceptHandler", "Raise", "With", "withitem",
    "Assert", "Delete", "Global", "Nonlocal",
    # Expressions
    "Call", "Name", "Constant", "Attribute",
    "Subscript", "Slice", "ExtSlice", "Index",
    "Tuple", "List", "Dict", "Set", "Starred",
    "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp",
    "Lambda", "Yield", "YieldFrom", "Await",
    "FormattedValue", "JoinedStr",
    # Operators
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "MatMult",
    "UAdd", "USub", "Invert", "Not",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
    "And", "Or",
    # Contexts
    "Load", "Store", "Del", "Param",
    # Keywords (named arguments in function calls)
    "keyword",
    # Comprehensions
    "ListComp", "DictComp", "SetComp", "GeneratorExp", "comprehension",
    # Imports
    "Import", "ImportFrom", "alias",
}

# Node types that are NEVER allowed.
_DISALLOWED_NODES: Set[str] = {
    "ClassDef",
    "AsyncFunctionDef",
    "AsyncFor",
    "AsyncWith",
    "Match",
    "MatchValue", "MatchSingleton", "MatchSequence", "MatchMapping",
    "MatchClass", "MatchStar", "MatchAs", "MatchOr",
}

# Built-in functions / names that are dangerous and must not be called.
_DANGEROUS_NAMES: Set[str] = {
    "eval", "exec", "compile", "__import__",
    "open", "input", "breakpoint",
    "exit", "quit",
    "help", "copyright", "credits", "license",
}

# Module-level imports that are dangerous.
_DANGEROUS_MODULES: Set[str] = {
    "os", "subprocess", "sys", "socket", "urllib", "http", "ftplib",
    "pickle", "marshal", "shelve", "dbm",
}

# Attribute access patterns that are dangerous (e.g. os.system).
_DANGEROUS_ATTRIBUTES: Set[str] = {
    "system", "popen", "spawn", "fork", "kill", "remove", "unlink",
    "rmdir", "removedirs", "rename", "replace",
    "execv", "execve", "execl", "execle", "execlp", "execvp", "execvpe",
}


class SandboxViolation(Exception):
    """Raised when handler code violates the sandbox policy."""
    pass


class HandlerSandbox:
    """AST-based validator for dynamically generated handler code."""

    @staticmethod
    def validate(source: str) -> Tuple[bool, Optional[str]]:
        """Validate source code for safety.

        Returns:
            (is_safe, error_message)
            is_safe=True means the code passed all checks.
            is_safe=False means it should NOT be exec'd; error_message explains why.
        """
        if not source or not isinstance(source, str):
            return False, "Source code must be a non-empty string"

        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Walk AST and check each node
        for node in ast.walk(tree):
            node_type = type(node).__name__

            # Check disallowed node types
            if node_type in _DISALLOWED_NODES:
                return False, f"Disallowed AST node: {node_type}"

            # Check unknown node types (defense in depth)
            if node_type not in _ALLOWED_NODES:
                return False, f"Unknown/untrusted AST node: {node_type}"

            # Check dangerous names in Call / Name nodes
            if isinstance(node, ast.Name):
                if node.id in _DANGEROUS_NAMES:
                    return False, f"Dangerous name usage: {node.id}"

            # Check dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _DANGEROUS_MODULES:
                        return False, f"Dangerous import: {alias.name}"

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in _DANGEROUS_MODULES:
                        return False, f"Dangerous import from: {node.module}"

            # Check dangerous attribute access (e.g. os.system)
            if isinstance(node, ast.Attribute):
                if node.attr in _DANGEROUS_ATTRIBUTES:
                    return False, f"Dangerous attribute access: {node.attr}"
                # Block dunder attribute access (e.g. obj.__class__)
                if node.attr.startswith("__") and not node.attr.endswith("__"):
                    return False, f"Private attribute access blocked: {node.attr}"
                # Block __class__, __bases__, etc.
                if node.attr in ("__class__", "__bases__", "__mro__", "__subclasses__",
                                  "__globals__", "__code__", "__func__", "__self__"):
                    return False, f"Metaprogramming attribute blocked: {node.attr}"

        return True, None

    @staticmethod
    def exec_safe(source: str, globals_dict: Optional[dict] = None) -> dict:
        """Validate then exec source code in a restricted namespace.

        Args:
            source: Python source code string.
            globals_dict: Optional globals dict.  If None, a minimal safe dict is used.

        Returns:
            The local namespace after exec.

        Raises:
            SandboxViolation: If code fails validation.
        """
        is_safe, error = HandlerSandbox.validate(source)
        if not is_safe:
            raise SandboxViolation(error)

        if globals_dict is None:
            globals_dict = {
                "__builtins__": __builtins__,
                "json": __import__("json"),
                "traceback": __import__("traceback"),
            }

        local_ns = {}
        exec(source, globals_dict, local_ns)
        return local_ns
