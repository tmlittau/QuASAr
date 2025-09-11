"""Verify plan cache produces consistent results and tracks hits."""
from __future__ import annotations

from benchmarks.circuits import ghz_circuit
from quasar.planner import Planner


def test_plan_cache_consistency() -> None:
    """Cached planning should yield identical results and track hits."""
    planner = Planner()
    circ = ghz_circuit(3)
    cold = planner.plan(circ, use_cache=False)
    warm1 = planner.plan(circ, use_cache=True)
    warm2 = planner.plan(circ, use_cache=True)
    assert warm1.steps == cold.steps
    assert warm2.steps == cold.steps
    assert planner.cache_hits == 1
