"""Verify backend switch counting."""

from __future__ import annotations

import pytest

from quasar.circuit import Circuit, Gate
from quasar.simulation_engine import SimulationEngine
from quasar.cost import Backend
from quasar.planner import PlanResult, PlanStep

BASELINE_SWITCHES = 2


def measure() -> int:
    """Return the number of backend switches for a simple plan."""
    circuit = Circuit([Gate("H", [0]), Gate("T", [0]), Gate("H", [0])])
    steps = [
        PlanStep(0, 1, Backend.TABLEAU),
        PlanStep(1, 2, Backend.MPS),
        PlanStep(2, 3, Backend.TABLEAU),
    ]
    plan = PlanResult(table=[], final_backend=None, gates=circuit.gates, explicit_steps=steps)
    plan.explicit_conversions = []
    engine = SimulationEngine()
    engine.scheduler.run(circuit, plan, instrument=True)
    _, metrics = engine.scheduler.run(circuit, plan, instrument=True)
    return metrics.backend_switches


def test_backend_switches() -> None:
    switches = measure()
    assert switches == BASELINE_SWITCHES
