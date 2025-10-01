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
    assert layer.residual_backend == Backend.MPS
    assert layer.converted_terms == layer.rank


def test_partial_conversion_prefers_high_entropy_qubits() -> None:
    """High-entropy boundary qubits should be extracted first."""

    planner = Planner(estimator=CostEstimator())
    gates = [
        Gate("CX", [0, 3]),
        Gate("CX", [1, 3]),
        Gate("CX", [2, 3]),
        Gate("H", [0]),
        Gate("CX", [0, 4]),
        Gate("CX", [0, 5]),
        Gate("CX", [1, 4]),
        Gate("T", [1]),
        Gate("RZ", [2], {"phi": 0.21}),
    ]

    steps = [
        PlanStep(start=0, end=3, backend=Backend.MPS),
        PlanStep(start=3, end=len(gates), backend=Backend.STATEVECTOR),
    ]

    layers = planner._conversions_for_steps(gates, steps)
    assert len(layers) == 1

    layer = layers[0]
    assert layer.boundary[0] == 0
    assert 2 in layer.retained
    assert layer.residual_backend == Backend.MPS
    assert layer.rank == layer.converted_terms
    assert layer.converted_terms <= 1 << layer.frontier


def test_subset_conversion_preserves_dense_rank() -> None:
    """Dense boundaries should preserve fidelity for the extracted subset."""

    planner = Planner(estimator=CostEstimator())
    gates = [
        Gate("H", [0]),
        Gate("RY", [1], {"theta": 0.3}),
        Gate("CX", [1, 2]),
        Gate("CX", [2, 3]),
        Gate("CX", [0, 4]),
        Gate("CX", [1, 4]),
        Gate("CX", [2, 4]),
    ]

    steps = [
        PlanStep(start=0, end=4, backend=Backend.MPS),
        PlanStep(start=4, end=len(gates), backend=Backend.STATEVECTOR),
    ]

    layers = planner._conversions_for_steps(gates, steps)
    assert len(layers) == 1

    layer = layers[0]
    assert layer.retained
    assert layer.rank == 1 << layer.frontier
    assert layer.converted_terms == layer.rank
