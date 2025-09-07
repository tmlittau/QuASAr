"""Benchmark quick-path memory usage on representative circuits.

This test compares peak memory consumption when planning with and without
quick-path heuristics.  The goal is to ensure the shortcuts do not
increase memory usage relative to full planning, keeping the bias toward
lower memory.
"""
from __future__ import annotations

import tracemalloc

import pytest

from benchmarks.circuits import (
    ghz_circuit,
    qft_circuit,
    random_circuit,
    w_state_circuit,
)
from quasar import Scheduler
from quasar.planner import Planner


def _peak_memory(circ, *, quick: bool) -> int:
    """Return peak memory in bytes for planning ``circ``."""
    if quick:
        planner = Planner(
            quick_max_qubits=10_000,
            quick_max_gates=1_000_000,
            quick_max_depth=10_000,
        )
    else:
        planner = Planner(quick_max_qubits=0, quick_max_gates=0, quick_max_depth=0)
    tracemalloc.start()
    planner.plan(circ)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


@pytest.mark.parametrize(
    "circ",
    [
        ghz_circuit(5),
        qft_circuit(5),
        w_state_circuit(5),
        random_circuit(5, seed=123),
    ],
)
def test_quick_path_does_not_increase_memory(circ) -> None:
    """Quick-path planning should not consume more memory."""
    scheduler = Scheduler(
        quick_max_qubits=10,
        quick_max_gates=3000,
        quick_max_depth=300,
    )
    assert scheduler.should_use_quick_path(circ)
    quick_mem = _peak_memory(circ, quick=True)
    full_mem = _peak_memory(circ, quick=False)
    assert quick_mem <= full_mem
