"""Tests for planner hybrid suffix and partial conversion heuristics."""

from __future__ import annotations

from quasar.circuit import Gate
from quasar.cost import Backend, CostEstimator
from quasar.planner import PlanStep, Planner


def test_partial_conversion_records_retained_boundary() -> None:
    """Switching to DD should retain dense qubits in the source backend."""

    planner = Planner(estimator=CostEstimator())

    gates = [
        Gate("CX", [0, 1]),
        Gate("CX", [1, 2]),
        Gate("CX", [2, 3]),
        Gate("RZ", [2], {"phi": 0.21}),
        Gate("RZ", [3], {"phi": 0.47}),
    ]

    steps = [
        PlanStep(start=0, end=3, backend=Backend.MPS),
        PlanStep(start=3, end=5, backend=Backend.DECISION_DIAGRAM),
    ]

    layers = planner._conversions_for_steps(gates, steps)
    assert len(layers) == 1

    layer = layers[0]
    assert layer.boundary == (2,)
    assert layer.retained == (3,)
    assert layer.full_boundary == (2, 3)
    assert layer.frontier == 1
    assert layer.rank <= 2
