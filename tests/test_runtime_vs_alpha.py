"""Verify runtime scales linearly with alpha.

The test uses ``time.sleep`` to simulate a workload whose runtime is
proportional to the input ``alpha``.  Measured runtimes are compared
against baseline values and required to increase with ``alpha``.
"""

from __future__ import annotations

import time

import pytest

ALPHAS = [0.5, 1, 2, 5]
BASELINE = {a: a * 0.01 for a in ALPHAS}


def run(alpha: float) -> float:
    """Return the runtime of a sleep-based workload for ``alpha``."""
    start = time.perf_counter()
    time.sleep(alpha * 0.01)
    return time.perf_counter() - start


def test_runtime_vs_alpha() -> None:
    """Runtimes should match the baseline curve and grow with ``alpha``."""
    runtimes = {a: run(a) for a in ALPHAS}
    for alpha, runtime in runtimes.items():
        assert runtime == pytest.approx(BASELINE[alpha], abs=0.005)
    assert all(runtimes[a] < runtimes[b] for a, b in zip(ALPHAS, ALPHAS[1:]))
