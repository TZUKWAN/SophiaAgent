"""Security utilities for SophiaAgent."""
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"forget\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(all\s+)?(your|the)\s+instructions?",
    r"you\s+are\s+now\s+a\s+(different|new)",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"###\s*instruction",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
]


class SecurityManager:
    def __init__(self, injection_threshold: float = 0.5):
        self.injection_threshold = injection_threshold
        self._compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

    def scan_prompt_injection(self, text: str) -> Tuple[bool, str]:
        matches = []
        for i, pattern in enumerate(self._compiled):
            if pattern.search(text):
                matches.append(INJECTION_PATTERNS[i])
        if matches:
            return True, f"Potential injection: matched {len(matches)} pattern(s)"
        return False, ""

    def redact_credentials(self, text: str) -> str:
        text = re.sub(r'sk-[a-zA-Z0-9]{10,}', 'sk-[REDACTED]', text)
        text = re.sub(r'(api[_-]?key\s*[=:]\s*)["\']?\S+["\']?',
                       r'\1[REDACTED]', text, flags=re.IGNORECASE)
        text = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', text)
        text = re.sub(r'(password\s*[=:]\s*)["\']?\S+["\']?',
                       r'\1[REDACTED]', text, flags=re.IGNORECASE)
        return text

    def validate_file_path(self, path: str, workspace: str) -> Tuple[bool, str]:
        try:
            resolved = Path(os.path.normpath(os.path.join(workspace, path))).resolve()
            ws_resolved = Path(workspace).resolve()
            if not str(resolved).startswith(str(ws_resolved)):
                return False, f"Path traversal: {path}"
            return True, ""
        except Exception as e:
            return False, f"Invalid path: {e}"

    def sanitize_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for key, value in args.items():
            if isinstance(value, str):
                value = self.redact_credentials(value)
            sanitized[key] = value
        return sanitized
