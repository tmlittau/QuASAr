"""Verify cumulative runtime speedup from plan cache reuse."""
from __future__ import annotations

from time import perf_counter
from typing import List

import numpy as np
from benchmarks.circuits import ghz_circuit
from quasar.planner import Planner


def cumulative_speedups(runs: int = 5) -> List[float]:
    """Return cumulative cold/warm runtime ratios over repeated runs."""
    planner_cold = Planner()
    planner_warm = Planner()
    cold_times = []
    warm_times = []
    for _ in range(runs):
        circ = ghz_circuit(3)
        t0 = perf_counter(); planner_cold.plan(circ, use_cache=False); cold_times.append(perf_counter() - t0)
        t1 = perf_counter(); planner_warm.plan(circ, use_cache=True); warm_times.append(perf_counter() - t1)
    cum_cold = np.cumsum(cold_times)
    cum_warm = np.cumsum(warm_times)
    return (cum_cold / cum_warm).tolist()


def test_plan_cache_speedup() -> None:
    speedup = cumulative_speedups()
    assert speedup[-1] > 1.0
    assert all(a < b for a, b in zip(speedup, speedup[1:]))
