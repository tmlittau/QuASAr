"""Verify plan cache hit-rate growth for repeated circuit runs."""
from __future__ import annotations

from benchmarks.circuits import ghz_circuit
from quasar.planner import Planner
import pytest


def hit_rates(runs: int = 5):
    planner = Planner()
    rates = []
    for i in range(1, runs + 1):
        planner.plan(ghz_circuit(3))
        rates.append(planner.cache_hits / i)
    return rates


def test_plan_cache_hit_rate() -> None:
    expected = [0.0, 0.5, 2/3, 0.75, 0.8]
    observed = hit_rates(len(expected))
    assert observed == pytest.approx(expected, rel=1e-6)
    assert all(a < b for a, b in zip(observed, observed[1:]))
