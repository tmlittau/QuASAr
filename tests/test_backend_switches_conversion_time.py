"""Verify backend switches and conversion time baseline."""

from __future__ import annotations

import pytest

from quasar.circuit import Circuit, Gate
from quasar.simulation_engine import SimulationEngine
from quasar.cost import Backend
from quasar.planner import PlanResult, PlanStep

BASELINE_SWITCHES = 2
BASELINE_CONVERSION_TIME = 0.0020


def measure() -> tuple[int, float]:
    """Return backend switches and total conversion time."""
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
    switches = metrics.backend_switches
    conversion_time = sum(metrics.conversion_durations)
    return switches, conversion_time


def test_backend_switches_conversion_time() -> None:
    switches, conv_time = measure()
    assert switches == BASELINE_SWITCHES
    assert conv_time == pytest.approx(BASELINE_CONVERSION_TIME, rel=0.5)
