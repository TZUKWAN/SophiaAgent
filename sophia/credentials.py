"""Credential pool with multi-API key rotation and failover.

Supports weighted round-robin selection, automatic error tracking,
rate-limit detection, and failover via the hook system.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sophia.hooks import HookEvent, HookManager

logger = logging.getLogger(__name__)


@dataclass
class Credential:
    id: int
    provider: str
    api_key: str
    base_url: str
    weight: int            # for weighted round-robin
    status: str            # active | rate_limited | error | disabled
    error_count: int
    last_used: Optional[str]
    last_error: Optional[str]


class CredentialPool:
    CREDENTIALS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS credentials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT NOT NULL,
        api_key TEXT NOT NULL,
        base_url TEXT NOT NULL,
        weight INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'active',
        error_count INTEGER NOT NULL DEFAULT 0,
        last_used TEXT,
        last_error TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """

    def __init__(self, db_path: str, hooks: HookManager = None):
        self.db_path = db_path
        self.hooks = hooks
        self._round_robin_index = 0
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(self.CREDENTIALS_SCHEMA)

    def add(self, provider, api_key, base_url, weight=1) -> int:
        """Add a credential. Returns the credential ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO credentials (provider, api_key, base_url, weight) "
                "VALUES (?, ?, ?, ?)",
                (provider, api_key, base_url, weight),
            )
            conn.commit()
            cred_id = cursor.lastrowid
            logger.info("Added credential id=%d provider=%s", cred_id, provider)
            return cred_id

    def remove(self, cred_id) -> bool:
        """Remove a credential by ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM credentials WHERE id = ?", (cred_id,)
            )
            conn.commit()
            removed = cursor.rowcount > 0
            if removed:
                logger.info("Removed credential id=%d", cred_id)
            return removed

    def _row_to_credential(self, row) -> Credential:
        return Credential(
            id=row["id"],
            provider=row["provider"],
            api_key=row["api_key"],
            base_url=row["base_url"],
            weight=row["weight"],
            status=row["status"],
            error_count=row["error_count"],
            last_used=row["last_used"],
            last_error=row["last_error"],
        )

    def get_next(self, provider=None) -> Optional[Credential]:
        """Get next available credential using weighted round-robin.

        Only returns credentials with status='active'.
        Filters by provider if specified.
        """
        with self._connect() as conn:
            if provider:
                rows = conn.execute(
                    "SELECT * FROM credentials WHERE status = 'active' AND provider = ? "
                    "ORDER BY id",
                    (provider,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM credentials WHERE status = 'active' ORDER BY id"
                ).fetchall()

            if not rows:
                return None

            # Build weighted list: each credential appears `weight` times
            weighted = []
            for row in rows:
                cred = self._row_to_credential(row)
                for _ in range(cred.weight):
                    weighted.append(cred)

            if not weighted:
                return None

            # Weighted round-robin
            idx = self._round_robin_index % len(weighted)
            self._round_robin_index = idx + 1
            selected = weighted[idx]

            # Update last_used timestamp
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            conn.execute(
                "UPDATE credentials SET last_used = ? WHERE id = ?",
                (now, selected.id),
            )
            conn.commit()

            return selected

    def report_success(self, cred_id):
        """Report successful use of a credential. Resets error count."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE credentials SET error_count = 0, last_error = NULL WHERE id = ?",
                (cred_id,),
            )
            conn.commit()
            logger.debug("Credential id=%d: success reported, error_count reset", cred_id)

    def report_error(self, cred_id, error_msg=""):
        """Report an error. Increments error_count. If >= 3 errors, marks as rate_limited.

        Emits CREDENTIAL_FAILOVER if credential is disabled.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM credentials WHERE id = ?", (cred_id,)
            ).fetchone()

            if not row:
                logger.warning("report_error: credential id=%d not found", cred_id)
                return

            new_count = row["error_count"] + 1
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            if new_count >= 3:
                new_status = "rate_limited"
            else:
                new_status = row["status"]

            conn.execute(
                "UPDATE credentials SET error_count = ?, status = ?, last_error = ?, "
                "last_used = ? WHERE id = ?",
                (new_count, new_status, error_msg, now, cred_id),
            )
            conn.commit()

            logger.info(
                "Credential id=%d: error reported (count=%d, status=%s): %s",
                cred_id, new_count, new_status, error_msg,
            )

            if new_status == "rate_limited" and self.hooks:
                self.hooks.emit(HookEvent.CREDENTIAL_FAILOVER, {
                    "credential_id": cred_id,
                    "provider": row["provider"],
                    "error_count": new_count,
                    "error": error_msg,
                })

    def failover(self, provider=None) -> Optional[Credential]:
        """Get next available credential (skip current errored ones).

        Emits CREDENTIAL_FAILOVER hook.
        """
        next_cred = self.get_next(provider=provider)

        if self.hooks:
            self.hooks.emit(HookEvent.CREDENTIAL_FAILOVER, {
                "provider": provider,
                "failover_credential_id": next_cred.id if next_cred else None,
            })

        if next_cred:
            logger.info("Failover to credential id=%d", next_cred.id)
        else:
            logger.warning("Failover: no available credential for provider=%s", provider)

        return next_cred

    def list_all(self) -> List[Dict]:
        """List all credentials (masking API keys for security)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM credentials ORDER BY id"
            ).fetchall()

            result = []
            for row in rows:
                key = row["api_key"]
                if len(key) > 8:
                    masked = key[:4] + "****" + key[-4:]
                else:
                    masked = "****"
                result.append({
                    "id": row["id"],
                    "provider": row["provider"],
                    "api_key": masked,
                    "base_url": row["base_url"],
                    "weight": row["weight"],
                    "status": row["status"],
                    "error_count": row["error_count"],
                    "last_used": row["last_used"],
                    "last_error": row["last_error"],
                })
            return result

    def reset(self, cred_id) -> bool:
        """Reset error status of a credential back to active."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE credentials SET status = 'active', error_count = 0, "
                "last_error = NULL WHERE id = ?",
                (cred_id,),
            )
            conn.commit()
            reset = cursor.rowcount > 0
            if reset:
                logger.info("Reset credential id=%d to active", cred_id)
            return reset
