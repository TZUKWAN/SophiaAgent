"""Global seed manager for reproducibility across research engines.

Propagates a single seed to numpy, Python's `random`, PYTHONHASHSEED, and
torch (if installed). Engines that previously hardcoded `random_state=42`
should switch to `GlobalSeed.get_or_default()`.
"""
from __future__ import annotations

import contextlib
import os
import random
from typing import Optional

import numpy as np


class GlobalSeed:
    """Process-wide RNG seed. Single-process, not thread-safe (intentional)."""

    _current_seed: Optional[int] = None

    @classmethod
    def set(cls, seed: int) -> int:
        """Set the global seed and propagate to all RNGs we know about."""
        seed = int(seed)
        cls._current_seed = seed
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Optional libraries
        try:
            import torch  # type: ignore

            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        try:
            import tensorflow as tf  # type: ignore

            tf.random.set_seed(seed)
        except Exception:
            pass
        return seed

    @classmethod
    def get(cls) -> Optional[int]:
        return cls._current_seed

    @classmethod
    def reset(cls) -> None:
        cls._current_seed = None
        np.random.seed(None)
        random.seed()

    @classmethod
    def get_or_default(cls, default: int = 42) -> int:
        """Return the current seed, or the supplied default if unset.

        Engines that need a reproducible RNG (e.g. sklearn `random_state=`)
        should call this rather than hardcoding 42, so user-set seeds take
        precedence.
        """
        seed = cls._current_seed
        return int(seed) if seed is not None else int(default)

    @classmethod
    @contextlib.contextmanager
    def with_seed(cls, seed: int):
        """Temporarily activate `seed`, restoring the prior value on exit."""
        prior = cls._current_seed
        cls.set(seed)
        try:
            yield seed
        finally:
            if prior is None:
                cls.reset()
            else:
                cls.set(prior)
