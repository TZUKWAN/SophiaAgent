"""Tests for RecoveryManager: error classification, retry, and backoff."""
import time
from unittest.mock import MagicMock, patch

import pytest

from sophia.credentials import CredentialPool
from sophia.hooks import HookEvent, HookManager
from sophia.recovery import ErrorCategory, RecoveryManager


@pytest.fixture
def recovery():
    """Create a RecoveryManager without credential pool."""
    return RecoveryManager()


@pytest.fixture
def recovery_with_hooks():
    """Create a RecoveryManager with hooks."""
    hooks = HookManager()
    mgr = RecoveryManager(hooks=hooks)
    return mgr, hooks


@pytest.fixture
def recovery_with_pool(tmp_path):
    """Create a RecoveryManager with a credential pool and hooks."""
    hooks = HookManager()
    db_path = str(tmp_path / "test_creds.db")
    pool = CredentialPool(db_path, hooks=hooks)
    pool.add("openai", "key1", "https://api.openai.com/v1")
    pool.add("openai", "key2", "https://api.openai.com/v1")
    mgr = RecoveryManager(hooks=hooks, credential_pool=pool)
    return mgr, hooks, pool


class TestErrorClassification:
    def test_classify_connection_error(self, recovery):
        """ConnectionError -> network."""
        assert recovery.classify_error(ConnectionError("refused")) == ErrorCategory.NETWORK

    def test_classify_timeout_error(self, recovery):
        """TimeoutError -> network."""
        assert recovery.classify_error(TimeoutError("timed out")) == ErrorCategory.NETWORK

    def test_classify_os_error(self, recovery):
        """OSError (network-level) -> network."""
        assert recovery.classify_error(OSError("network unreachable")) == ErrorCategory.NETWORK

    def test_classify_rate_limit_429(self, recovery):
        """Error with '429' in message -> rate_limit."""
        assert recovery.classify_error(RuntimeError("HTTP 429 Too Many Requests")) == ErrorCategory.RATE_LIMIT

    def test_classify_rate_limit_keyword(self, recovery):
        """Error with 'rate limit' in message -> rate_limit."""
        assert recovery.classify_error(RuntimeError("rate limit exceeded")) == ErrorCategory.RATE_LIMIT

    def test_classify_rate_limit_too_many_requests(self, recovery):
        """Error with 'too many requests' in message -> rate_limit."""
        assert recovery.classify_error(RuntimeError("Too many requests")) == ErrorCategory.RATE_LIMIT

    def test_classify_auth_401(self, recovery):
        """Error with '401' in message -> auth."""
        assert recovery.classify_error(RuntimeError("HTTP 401 Unauthorized")) == ErrorCategory.AUTH

    def test_classify_auth_403(self, recovery):
        """Error with '403' in message -> auth."""
        assert recovery.classify_error(RuntimeError("HTTP 403 Forbidden")) == ErrorCategory.AUTH

    def test_classify_auth_keyword(self, recovery):
        """Error with 'authentication' in message -> auth."""
        assert recovery.classify_error(RuntimeError("Authentication failed")) == ErrorCategory.AUTH

    def test_classify_auth_unauthorized(self, recovery):
        """Error with 'unauthorized' in message -> auth."""
        assert recovery.classify_error(RuntimeError("Unauthorized access")) == ErrorCategory.AUTH

    def test_classify_auth_forbidden(self, recovery):
        """Error with 'forbidden' in message -> auth."""
        assert recovery.classify_error(RuntimeError("Access forbidden")) == ErrorCategory.AUTH

    def test_classify_context_length(self, recovery):
        """Error with 'context length' in message -> context."""
        assert recovery.classify_error(RuntimeError("context length exceeded")) == ErrorCategory.CONTEXT

    def test_classify_max_tokens(self, recovery):
        """Error with 'max tokens' in message -> context."""
        assert recovery.classify_error(RuntimeError("max tokens exceeded")) == ErrorCategory.CONTEXT

    def test_classify_provider_500(self, recovery):
        """Error with '500' in message -> provider."""
        assert recovery.classify_error(RuntimeError("HTTP 500 Internal Server Error")) == ErrorCategory.PROVIDER

    def test_classify_provider_502(self, recovery):
        """Error with '502' in message -> provider."""
        assert recovery.classify_error(RuntimeError("HTTP 502 Bad Gateway")) == ErrorCategory.PROVIDER

    def test_classify_provider_503(self, recovery):
        """Error with '503' in message -> provider."""
        assert recovery.classify_error(RuntimeError("HTTP 503 Service Unavailable")) == ErrorCategory.PROVIDER

    def test_classify_unknown(self, recovery):
        """Unclassified error -> unknown."""
        assert recovery.classify_error(RuntimeError("something weird")) == ErrorCategory.UNKNOWN

    def test_classify_value_error(self, recovery):
        """Plain ValueError -> unknown."""
        assert recovery.classify_error(ValueError("bad value")) == ErrorCategory.UNKNOWN


class TestShouldRetry:
    def test_retry_network_first_attempt(self, recovery):
        """Network errors should retry on first attempt."""
        assert recovery.should_retry(ConnectionError("refused"), 0) is True

    def test_retry_network_exceeds_max(self, recovery):
        """Network errors should not retry beyond max_retries=3."""
        assert recovery.should_retry(ConnectionError("refused"), 3) is False

    def test_retry_rate_limit(self, recovery):
        """Rate limit errors should retry up to max_retries=2."""
        assert recovery.should_retry(RuntimeError("429"), 0) is True
        assert recovery.should_retry(RuntimeError("429"), 1) is True
        assert recovery.should_retry(RuntimeError("429"), 2) is False

    def test_no_retry_auth(self, recovery):
        """Auth errors should never retry."""
        assert recovery.should_retry(RuntimeError("401"), 0) is False

    def test_no_retry_unknown(self, recovery):
        """Unknown errors should never retry."""
        assert recovery.should_retry(RuntimeError("weird"), 0) is False

    def test_no_retry_context(self, recovery):
        """Context errors should never retry."""
        assert recovery.should_retry(RuntimeError("context length"), 0) is False

    def test_retry_provider(self, recovery):
        """Provider errors should retry up to max_retries=2."""
        assert recovery.should_retry(RuntimeError("500"), 0) is True
        assert recovery.should_retry(RuntimeError("500"), 1) is True
        assert recovery.should_retry(RuntimeError("500"), 2) is False


class TestGetBackoff:
    def test_network_backoff(self, recovery):
        """Network backoff: [1, 3, 5]."""
        assert recovery.get_backoff(ErrorCategory.NETWORK, 0) == 1
        assert recovery.get_backoff(ErrorCategory.NETWORK, 1) == 3
        assert recovery.get_backoff(ErrorCategory.NETWORK, 2) == 5

    def test_rate_limit_backoff(self, recovery):
        """Rate limit backoff: [10, 30]."""
        assert recovery.get_backoff(ErrorCategory.RATE_LIMIT, 0) == 10
        assert recovery.get_backoff(ErrorCategory.RATE_LIMIT, 1) == 30

    def test_backoff_clamps_to_last(self, recovery):
        """If attempt exceeds backoff list, uses last value."""
        assert recovery.get_backoff(ErrorCategory.NETWORK, 10) == 5
        assert recovery.get_backoff(ErrorCategory.RATE_LIMIT, 10) == 30

    def test_provider_backoff(self, recovery):
        """Provider backoff: [2, 5]."""
        assert recovery.get_backoff(ErrorCategory.PROVIDER, 0) == 2
        assert recovery.get_backoff(ErrorCategory.PROVIDER, 1) == 5

    def test_unknown_category_default(self, recovery):
        """Unknown category returns default 1.0 backoff."""
        assert recovery.get_backoff(ErrorCategory.UNKNOWN, 0) == 1.0


class TestExecuteWithRecovery:
    def test_successful_function(self, recovery):
        """Successful function returns result without retry."""
        fn = MagicMock(return_value="ok")
        result = recovery.execute_with_recovery(fn, "arg1", key="val")
        assert result == "ok"
        assert fn.call_count == 1

    @patch("sophia.recovery.time.sleep")
    def test_retry_on_network_error(self, mock_sleep, recovery):
        """Retries on recoverable network error."""
        fn = MagicMock(side_effect=[
            ConnectionError("refused"),
            "success",
        ])
        result = recovery.execute_with_recovery(fn)
        assert result == "success"
        assert fn.call_count == 2
        assert mock_sleep.called  # backoff sleep was called

    @patch("sophia.recovery.time.sleep")
    def test_multiple_retries(self, mock_sleep, recovery):
        """Retries multiple times on persistent network error."""
        fn = MagicMock(side_effect=[
            ConnectionError("e1"),
            ConnectionError("e2"),
            ConnectionError("e3"),
            "finally",
        ])
        result = recovery.execute_with_recovery(fn)
        assert result == "finally"
        assert fn.call_count == 4
        assert mock_sleep.called  # sleep was called for backoff

    def test_raises_non_retryable_auth(self, recovery):
        """Raises auth error immediately without retry."""
        fn = MagicMock(side_effect=RuntimeError("401 Unauthorized"))
        with pytest.raises(RuntimeError, match="401"):
            recovery.execute_with_recovery(fn)
        assert fn.call_count == 1

    def test_raises_after_max_retries(self, recovery):
        """Raises the last error after max retries are exhausted."""
        fn = MagicMock(side_effect=ConnectionError("persistent"))
        with pytest.raises(ConnectionError, match="persistent"):
            recovery.execute_with_recovery(fn)
        # network max_retries=3, so 1 initial + 3 retries = 4 calls
        assert fn.call_count == 4

    @patch("sophia.recovery.time.sleep")
    def test_emits_recovery_retry_hook(self, mock_sleep, recovery_with_hooks):
        """Each retry emits RECOVERY_RETRY hook."""
        recovery, hooks = recovery_with_hooks
        events = []
        hooks.register(
            HookEvent.RECOVERY_RETRY,
            lambda ctx: (events.append(ctx), ctx)[1],
        )

        fn = MagicMock(side_effect=[
            ConnectionError("e1"),
            "ok",
        ])
        recovery.execute_with_recovery(fn)

        assert len(events) == 1
        assert events[0]["attempt"] == 0
        assert events[0]["category"] == ErrorCategory.NETWORK
        assert events[0]["backoff"] == 1

    @patch("sophia.recovery.time.sleep")
    def test_rate_limit_triggers_failover(self, mock_sleep, recovery_with_pool):
        """Rate limit error triggers credential failover."""
        recovery, hooks, pool = recovery_with_pool

        fn = MagicMock(side_effect=[
            RuntimeError("429 rate limited"),
            "ok",
        ])
        recovery.execute_with_recovery(fn)

        # Should have slept for rate_limit backoff
        assert mock_sleep.called

    def test_passes_args_and_kwargs(self, recovery):
        """Arguments and keyword arguments are passed through."""
        fn = MagicMock(return_value="result")
        recovery.execute_with_recovery(fn, "a", "b", x=1, y=2)
        fn.assert_called_once_with("a", "b", x=1, y=2)


class TestOnToolError:
    def test_classifies_error(self, recovery):
        """on_tool_error adds error_category to context."""
        ctx = recovery.on_tool_error({"error": ConnectionError("refused")})
        assert ctx["error_category"] == ErrorCategory.NETWORK

    def test_suggests_retry_for_recoverable(self, recovery):
        """on_tool_error suggests retry for network errors."""
        ctx = recovery.on_tool_error({"error": ConnectionError("refused")})
        assert ctx["retry_suggested"] is True
        assert ctx["retry_backoff"] == 1

    def test_no_retry_for_auth(self, recovery):
        """on_tool_error does not suggest retry for auth errors."""
        ctx = recovery.on_tool_error({"error": RuntimeError("401")})
        assert ctx["retry_suggested"] is False

    def test_handles_string_error(self, recovery):
        """on_tool_error handles string errors by wrapping in RuntimeError."""
        ctx = recovery.on_tool_error({"error": "429 rate limited"})
        assert ctx["error_category"] == ErrorCategory.RATE_LIMIT
        assert ctx["retry_suggested"] is True

    def test_no_error_in_context(self, recovery):
        """on_tool_error returns context as-is if no error."""
        ctx = recovery.on_tool_error({"tool": "read"})
        assert "error_category" not in ctx
