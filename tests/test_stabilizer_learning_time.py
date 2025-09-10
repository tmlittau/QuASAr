"""Test stabilizer-learning runtime statistics."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pytest
from quasar_convert import ConversionEngine

WINDOW_SIZES = [1, 2, 3, 4]
BASELINE = {
    1: {"mean": 2.7e-05, "std": 3.9e-06},
    2: {"mean": 2.7e-05, "std": 3.4e-06},
    3: {"mean": 3.9e-05, "std": 2.3e-06},
    4: {"mean": 3.9e-05, "std": 2.6e-06},
}


def plus_state(n: int) -> list[float]:
    """Return amplitudes of the n-qubit ``|+...+>`` state."""
    dim = 1 << n
    amp = 1 / np.sqrt(dim)
    return [amp] * dim


def measure(window: int, repeats: int = 8, inner: int = 5000) -> Tuple[float, float]:
    """Return mean and std runtime for stabilizer learning on ``window`` qubits."""
    eng = ConversionEngine()
    state = plus_state(window)
    times = []
    for _ in range(repeats):
        start = time.process_time()
        for _ in range(inner):
            eng.learn_stabilizer(state)
        times.append((time.process_time() - start) / inner)
    return float(np.mean(times)), float(np.std(times, ddof=1))


def test_stabilizer_learning_time_matches_baseline() -> None:
    """Measured statistics should roughly match stored expectations."""
    for w in WINDOW_SIZES:
        mean, std = measure(w)
        assert mean == pytest.approx(BASELINE[w]["mean"], rel=1.0)
        assert std == pytest.approx(BASELINE[w]["std"], rel=1.0)
