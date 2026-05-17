"""Error recovery manager with classification, retry, and backoff.

Classifies errors into categories and applies configurable retry
strategies with exponential backoff. Integrates with CredentialPool
for automatic failover on rate-limit errors.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)


class ErrorCategory:
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    TOOL = "tool"
    CONTEXT = "context"
    PROVIDER = "provider"
    UNKNOWN = "unknown"


class RecoveryManager:
    def __init__(self, hooks: HookManager = None, credential_pool=None):
        self.hooks = hooks
        self.credential_pool = credential_pool
        self._retry_config = {
            "network": {"max_retries": 3, "backoff": [1, 3, 5]},
            "rate_limit": {"max_retries": 2, "backoff": [10, 30]},
            "provider": {"max_retries": 2, "backoff": [2, 5]},
            "tool": {"max_retries": 1, "backoff": [1]},
        }

    def classify_error(self, error: Exception) -> str:
        """Classify an error into a category based on error type and message.

        Check for: ConnectionError/TimeoutError -> network
                   RateLimitError/429 -> rate_limit
                   AuthenticationError/401/403 -> auth
                   ToolExecutionError -> tool
                   ContextLengthError -> context
                   Other provider errors -> provider
                   Default -> unknown
        """
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()

        # Check by exception type first
        if isinstance(error, (ConnectionError, TimeoutError, OSError)):
            return ErrorCategory.NETWORK

        # Check type name for common patterns
        if "ratelimit" in error_type or "ratelimit" in error_msg:
            return ErrorCategory.RATE_LIMIT

        if "auth" in error_type:
            return ErrorCategory.AUTH

        if "tool" in error_type and "execution" in error_type:
            return ErrorCategory.TOOL

        if "context" in error_type and "length" in error_type:
            return ErrorCategory.CONTEXT

        # Check message content for HTTP status codes and keywords
        if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
            return ErrorCategory.RATE_LIMIT

        if ("401" in error_msg or "403" in error_msg or "unauthorized" in error_msg
                or "forbidden" in error_msg or "authentication" in error_msg):
            return ErrorCategory.AUTH

        if "tool execution" in error_msg or "tool execution" in error_type:
            return ErrorCategory.TOOL

        if "context length" in error_msg or "max tokens" in error_msg or "token limit" in error_msg:
            return ErrorCategory.CONTEXT

        # Provider-level errors (e.g. 500, 502, 503 from API)
        if "500" in error_msg or "502" in error_msg or "503" in error_msg:
            return ErrorCategory.PROVIDER

        if "provider" in error_type:
            return ErrorCategory.PROVIDER

        return ErrorCategory.UNKNOWN

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if we should retry based on error category and attempt count."""
        category = self.classify_error(error)

        # Never retry auth or unknown errors
        if category in (ErrorCategory.AUTH, ErrorCategory.UNKNOWN, ErrorCategory.CONTEXT):
            return False

        config = self._retry_config.get(category)
        if not config:
            return False

        return attempt < config["max_retries"]

    def get_backoff(self, category: str, attempt: int) -> float:
        """Get backoff time in seconds for the given category and attempt.

        attempt is 0-indexed. If attempt exceeds the backoff list length,
        uses the last backoff value.
        """
        config = self._retry_config.get(category)
        if not config:
            return 1.0

        backoff_list = config["backoff"]
        if not backoff_list:
            return 1.0

        idx = min(attempt, len(backoff_list) - 1)
        return backoff_list[idx]

    def execute_with_recovery(self, fn: Callable, *args, **kwargs) -> Any:
        """Execute fn with automatic retry on recoverable errors.

        For each attempt:
        1. Try fn(*args, **kwargs)
        2. On error, classify it
        3. If should_retry, wait backoff, then retry
        4. If rate_limit error and credential_pool available, try failover
        5. Emit RECOVERY_RETRY hook on each retry
        6. If max retries exceeded, raise the last error
        """
        last_error = None
        attempt = 0

        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                category = self.classify_error(error=e)
                logger.info(
                    "Error on attempt %d: category=%s, type=%s, msg=%s",
                    attempt, category, type(e).__name__, str(e),
                )

                if not self.should_retry(e, attempt):
                    logger.error(
                        "Non-retryable error (category=%s): %s", category, e
                    )
                    raise

                backoff = self.get_backoff(category, attempt)

                # Emit recovery retry hook
                if self.hooks:
                    self.hooks.emit(HookEvent.RECOVERY_RETRY, {
                        "attempt": attempt,
                        "category": category,
                        "backoff": backoff,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    })

                # If rate_limit and credential pool available, try failover
                if category == ErrorCategory.RATE_LIMIT and self.credential_pool:
                    new_cred = self.credential_pool.failover()
                    if new_cred:
                        logger.info(
                            "Failover to credential id=%d for rate limit recovery",
                            new_cred.id,
                        )

                logger.info(
                    "Retrying in %.1fs (attempt %d, category=%s)",
                    backoff, attempt + 1, category,
                )
                time.sleep(backoff)
                attempt += 1

    def on_tool_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook handler for tool.error - can suggest retry."""
        error = context.get("error")
        if error is None:
            return context

        if isinstance(error, str):
            # Wrap string errors for classification
            error = RuntimeError(error)

        category = self.classify_error(error)
        should_retry = self.should_retry(error, context.get("attempt", 0))

        context["error_category"] = category
        context["retry_suggested"] = should_retry

        if should_retry:
            backoff = self.get_backoff(category, context.get("attempt", 0))
            context["retry_backoff"] = backoff

        return context
