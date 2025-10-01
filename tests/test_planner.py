"""Planner regression tests."""

from __future__ import annotations

from quasar.circuit import Circuit, Gate
from quasar.cost import Backend
from quasar.planner import PlanStep, Planner


def test_repeated_planning_does_not_rebuild_circuit_metadata(monkeypatch) -> None:
    """Repeated planning should not trigger another metadata rebuild."""

    circuit = Circuit(
        [
            Gate("H", [0]),
            Gate("CX", [0, 1]),
            Gate("T", [1]),
        ],
        use_classical_simplification=True,
    )
    planner = Planner()

    call_count = 0
    original_refresh = Circuit._refresh_metadata

    def spy(self: Circuit) -> None:
        nonlocal call_count
        call_count += 1
        original_refresh(self)

    monkeypatch.setattr(Circuit, "_refresh_metadata", spy)

    planner.plan(circuit)
    planner.plan(circuit)

    assert call_count == 0


def test_plan_records_wide_conversion_window() -> None:
    gates = [Gate("H", [q]) for q in range(5)]
    gates.append(Gate("CX", [4, 5]))
    gates.extend(Gate("X", [q]) for q in range(6))
    planner = Planner()
    steps = [
        PlanStep(start=0, end=6, backend=Backend.TABLEAU),
        PlanStep(start=6, end=len(gates), backend=Backend.MPS),
    ]

    layers = planner._conversions_for_steps(gates, steps)

    assert layers
    layer = layers[0]
    assert layer.primitive == "LW"
    assert layer.window == 5
