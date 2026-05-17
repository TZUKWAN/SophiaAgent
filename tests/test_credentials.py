"""Tests for CredentialPool: multi-API key rotation and failover."""
import time

import pytest

from sophia.credentials import CredentialPool
from sophia.hooks import HookEvent, HookManager


@pytest.fixture
def pool(tmp_path):
    """Create a CredentialPool with a temporary database."""
    db_path = str(tmp_path / "test_creds.db")
    return CredentialPool(db_path)


@pytest.fixture
def pool_with_hooks(tmp_path):
    """Create a CredentialPool with a HookManager attached."""
    db_path = str(tmp_path / "test_creds_hooks.db")
    hooks = HookManager()
    return CredentialPool(db_path, hooks=hooks), hooks


class TestCredentialPool:
    def test_add_credential(self, pool):
        """Add a credential and verify all fields are stored correctly."""
        cred_id = pool.add(
            provider="openai",
            api_key="sk-test-12345678abcdefgh",
            base_url="https://api.openai.com/v1",
            weight=2,
        )
        assert cred_id == 1

        # Verify via get_next
        cred = pool.get_next(provider="openai")
        assert cred is not None
        assert cred.id == cred_id
        assert cred.provider == "openai"
        assert cred.api_key == "sk-test-12345678abcdefgh"
        assert cred.base_url == "https://api.openai.com/v1"
        assert cred.weight == 2
        assert cred.status == "active"
        assert cred.error_count == 0

    def test_add_multiple_providers(self, pool):
        """Add credentials for different providers."""
        id1 = pool.add("openai", "key1", "https://api.openai.com/v1")
        id2 = pool.add("anthropic", "key2", "https://api.anthropic.com")
        id3 = pool.add("openai", "key3", "https://api.openai.com/v1")

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_get_next_round_robin(self, pool):
        """Get next cycles through credentials with equal weight."""
        pool.add("openai", "key-a", "https://api.openai.com/v1", weight=1)
        pool.add("openai", "key-b", "https://api.openai.com/v1", weight=1)

        first = pool.get_next(provider="openai")
        second = pool.get_next(provider="openai")
        third = pool.get_next(provider="openai")

        assert first is not None
        assert second is not None
        assert third is not None
        # Should cycle: a, b, a
        assert first.api_key != second.api_key
        assert first.api_key == third.api_key

    def test_get_next_weighted(self, pool):
        """Weighted round-robin: higher weight = more selections."""
        pool.add("openai", "key-light", "https://api.openai.com/v1", weight=1)
        pool.add("openai", "key-heavy", "https://api.openai.com/v1", weight=3)

        # Collect 4 selections (weight sum = 4)
        keys = [pool.get_next(provider="openai").api_key for _ in range(4)]
        heavy_count = keys.count("key-heavy")
        light_count = keys.count("key-light")

        assert heavy_count == 3
        assert light_count == 1

    def test_get_next_filter_by_provider(self, pool):
        """get_next only returns credentials matching the provider."""
        pool.add("openai", "openai-key", "https://api.openai.com/v1")
        pool.add("anthropic", "anthropic-key", "https://api.anthropic.com")

        cred = pool.get_next(provider="openai")
        assert cred.provider == "openai"

        cred = pool.get_next(provider="anthropic")
        assert cred.provider == "anthropic"

        # Non-existent provider
        cred = pool.get_next(provider="nonexistent")
        assert cred is None

    def test_get_next_skips_non_active(self, pool):
        """get_next only returns credentials with status='active'."""
        id1 = pool.add("openai", "key1", "https://api.openai.com/v1")
        id2 = pool.add("openai", "key2", "https://api.openai.com/v1")

        # Report 3 errors on first credential to mark it rate_limited
        pool.report_error(id1, "err1")
        pool.report_error(id1, "err2")
        pool.report_error(id1, "err3")

        # Should only return the second credential
        cred = pool.get_next(provider="openai")
        assert cred is not None
        assert cred.id == id2

    def test_report_success_resets_errors(self, pool):
        """report_success resets error_count to 0."""
        cred_id = pool.add("openai", "key", "https://api.openai.com/v1")

        pool.report_error(cred_id, "some error")
        cred = pool.get_next(provider="openai")
        assert cred.error_count == 1

        pool.report_success(cred_id)
        cred = pool.get_next(provider="openai")
        assert cred.error_count == 0

    def test_report_error_increments(self, pool):
        """report_error increments error_count."""
        cred_id = pool.add("openai", "key", "https://api.openai.com/v1")

        pool.report_error(cred_id, "err1")
        pool.report_error(cred_id, "err2")

        cred = pool.get_next(provider="openai")
        assert cred.error_count == 2

    def test_report_error_marks_rate_limited_after_three(self, pool):
        """After 3 errors, credential status becomes 'rate_limited'."""
        cred_id = pool.add("openai", "key", "https://api.openai.com/v1")

        pool.report_error(cred_id, "err1")
        pool.report_error(cred_id, "err2")
        # Should still be active after 2 errors
        creds = pool.list_all()
        assert creds[0]["status"] == "active"

        # Third error triggers rate_limited
        pool.report_error(cred_id, "err3")
        creds = pool.list_all()
        assert creds[0]["status"] == "rate_limited"
        assert creds[0]["error_count"] == 3

    def test_failover_gets_next_active(self, pool):
        """failover returns next active credential, skipping errored ones."""
        id1 = pool.add("openai", "key1", "https://api.openai.com/v1")
        id2 = pool.add("openai", "key2", "https://api.openai.com/v1")

        # Mark first as rate_limited
        pool.report_error(id1, "e1")
        pool.report_error(id1, "e2")
        pool.report_error(id1, "e3")

        cred = pool.failover(provider="openai")
        assert cred is not None
        assert cred.id == id2
        assert cred.status == "active"

    def test_failover_emits_hook(self, pool_with_hooks):
        """failover emits CREDENTIAL_FAILOVER hook."""
        pool, hooks = pool_with_hooks
        pool.add("openai", "key1", "https://api.openai.com/v1")

        events = []
        hooks.register(
            HookEvent.CREDENTIAL_FAILOVER,
            lambda ctx: (events.append(ctx), ctx)[1],
        )

        pool.failover(provider="openai")
        assert len(events) == 1
        assert "failover_credential_id" in events[0]

    def test_failover_none_when_all_limited(self, pool):
        """failover returns None when no active credentials remain."""
        id1 = pool.add("openai", "key1", "https://api.openai.com/v1")

        pool.report_error(id1, "e1")
        pool.report_error(id1, "e2")
        pool.report_error(id1, "e3")

        cred = pool.failover(provider="openai")
        assert cred is None

    def test_list_all_masks_api_keys(self, pool):
        """list_all returns masked API keys."""
        pool.add("openai", "sk-test-12345678abcdefgh", "https://api.openai.com/v1")
        pool.add("anthropic", "short", "https://api.anthropic.com")

        creds = pool.list_all()
        assert len(creds) == 2

        # Long key: first 4 + **** + last 4
        assert creds[0]["api_key"] == "sk-t****efgh"
        # Short key: fully masked
        assert creds[1]["api_key"] == "****"

    def test_list_all_returns_all_fields(self, pool):
        """list_all includes all expected fields."""
        pool.add("openai", "sk-test-12345678abcdefgh", "https://api.openai.com/v1")

        creds = pool.list_all()
        assert len(creds) == 1
        c = creds[0]
        assert "id" in c
        assert "provider" in c
        assert "base_url" in c
        assert "weight" in c
        assert "status" in c
        assert "error_count" in c
        assert "last_used" in c
        assert "last_error" in c

    def test_remove_credential(self, pool):
        """Remove a credential by ID."""
        cred_id = pool.add("openai", "key", "https://api.openai.com/v1")
        assert pool.remove(cred_id) is True
        assert pool.get_next(provider="openai") is None

    def test_remove_nonexistent(self, pool):
        """Removing a non-existent credential returns False."""
        assert pool.remove(9999) is False

    def test_reset_credential_status(self, pool):
        """Reset restores a rate_limited credential to active."""
        cred_id = pool.add("openai", "key", "https://api.openai.com/v1")

        # Trigger rate_limited
        pool.report_error(cred_id, "e1")
        pool.report_error(cred_id, "e2")
        pool.report_error(cred_id, "e3")

        creds = pool.list_all()
        assert creds[0]["status"] == "rate_limited"

        # Reset
        assert pool.reset(cred_id) is True

        cred = pool.get_next(provider="openai")
        assert cred is not None
        assert cred.status == "active"
        assert cred.error_count == 0

    def test_reset_nonexistent(self, pool):
        """Resetting a non-existent credential returns False."""
        assert pool.reset(9999) is False

    def test_report_error_emits_hook_on_rate_limited(self, pool_with_hooks):
        """report_error emits CREDENTIAL_FAILOVER when credential becomes rate_limited."""
        pool, hooks = pool_with_hooks
        cred_id = pool.add("openai", "key", "https://api.openai.com/v1")

        events = []
        hooks.register(
            HookEvent.CREDENTIAL_FAILOVER,
            lambda ctx: (events.append(ctx), ctx)[1],
        )

        # First two errors: no hook
        pool.report_error(cred_id, "e1")
        pool.report_error(cred_id, "e2")
        assert len(events) == 0

        # Third error triggers hook
        pool.report_error(cred_id, "e3")
        assert len(events) == 1
        assert events[0]["credential_id"] == cred_id
        assert events[0]["error_count"] == 3

    def test_get_next_updates_last_used(self, pool):
        """get_next updates the last_used timestamp."""
        pool.add("openai", "key", "https://api.openai.com/v1")
        before = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        pool.get_next(provider="openai")

        creds = pool.list_all()
        assert creds[0]["last_used"] is not None
        assert creds[0]["last_used"] >= before
