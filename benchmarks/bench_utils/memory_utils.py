from __future__ import annotations

"""Helpers for memory related benchmark utilities."""

import math
import os
from typing import Optional

DEFAULT_MEMORY_BYTES = 64 * 1024 ** 3
"""Default memory budget (64 GiB) for statevector simulations."""

ENV_VAR = "QUASAR_STATEVECTOR_MAX_MEMORY_BYTES"
"""Environment variable to override the default statevector memory budget."""


def max_qubits_statevector(memory_bytes: Optional[int] = None) -> int:
    """Return the maximum number of qubits for dense statevectors.

    Parameters
    ----------
    memory_bytes:
        Number of bytes available for amplitudes. When ``None`` the value is
        taken from :data:`ENV_VAR` and falls back to
        :data:`DEFAULT_MEMORY_BYTES`.

    Returns
    -------
    int
        Maximum number of qubits that fit into memory assuming 16 bytes per
        amplitude.
    """

    if memory_bytes is None:
        env = os.getenv(ENV_VAR)
        memory_bytes = int(env) if env is not None else DEFAULT_MEMORY_BYTES
    if memory_bytes < 16:
        return 0
    return int(math.floor(math.log2(memory_bytes / 16)))


__all__ = ["max_qubits_statevector"]
