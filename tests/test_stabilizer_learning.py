"""Test stabilizer-learning preserves input state."""

from __future__ import annotations

import numpy as np
from quasar_convert import ConversionEngine

WINDOW_SIZES = [1, 2, 3, 4]


def plus_state(n: int) -> list[float]:
    """Return amplitudes of the n-qubit ``|+...+>`` state."""
    dim = 1 << n
    amp = 1 / np.sqrt(dim)
    return [amp] * dim


def test_stabilizer_learning_preserves_state() -> None:
    """Learning stabilizers should not mutate the provided state."""
    eng = ConversionEngine()
    for w in WINDOW_SIZES:
        state = plus_state(w)
        before = list(state)
        eng.learn_stabilizer(state)
        assert state == before
