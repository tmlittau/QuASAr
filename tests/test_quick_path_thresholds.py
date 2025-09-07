"""Benchmark quick-path thresholds on representative circuits.

This test runs small circuits through the planner with and without
quick-path heuristics to verify that the shortcuts provide tangible
speedups.  It covers GHZ, QFT, W state preparation and a seeded
random circuit so the results are comparable across runs.
"""
from __future__ import annotations

import time

import pytest

from benchmarks.circuits import (
    ghz_circuit,
    qft_circuit,
    random_circuit,
    w_state_circuit,
)
from quasar import Scheduler
from quasar.planner import Planner
from quasar.cost import Backend


def _time_planning(circ, *, quick: bool) -> float:
    """Return planning time for ``circ`` with or without quick analysis."""
    if quick:
        planner = Planner(
            quick_max_qubits=10_000,
            quick_max_gates=1_000_000,
            quick_max_depth=10_000,
        )
    else:
        planner = Planner(quick_max_qubits=0, quick_max_gates=0, quick_max_depth=0)
    start = time.perf_counter()
    planner.plan(circ)
    return time.perf_counter() - start


@pytest.mark.parametrize(
    "circ",
    [
        ghz_circuit(5),
        qft_circuit(5),
        w_state_circuit(5),
        random_circuit(5, seed=123),
    ],
)
def test_quick_path_thresholds_match_performance(circ) -> None:
    """Quick-path planning should be faster on small circuits."""
    scheduler = Scheduler(
        quick_max_qubits=10,
        quick_max_gates=3000,
        quick_max_depth=300,
    )
    assert scheduler.should_use_quick_path(circ)
    quick_time = _time_planning(circ, quick=True)
    full_time = _time_planning(circ, quick=False)
    assert quick_time < full_time


def test_backend_still_checks_quick_path() -> None:
    scheduler = Scheduler(quick_max_qubits=1, quick_max_gates=1, quick_max_depth=1)
    circ = ghz_circuit(2)
    assert not scheduler.should_use_quick_path(circ, backend=Backend.STATEVECTOR)
    assert scheduler.should_use_quick_path(
        circ, backend=Backend.STATEVECTOR, force=True
    )
