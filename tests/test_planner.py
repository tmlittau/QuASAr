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
    assert layer.frontier == 1
    assert layer.boundary == (0,)
    assert layer.retained == (1, 2, 3, 4, 5)
    assert layer.full_boundary == (0, 1, 2, 3, 4, 5)
    assert layer.residual_backend == Backend.TABLEAU
    assert layer.converted_terms is not None
    assert layer.rank == layer.converted_terms
    if layer.primitive == "LW":
        assert layer.window == 5
    else:
        # Hybrid suffix planning can trigger a full extraction when the
        # retained subset is small and a local window would not reduce the
        # conversion cost.  In that case the primitive switches to a full
        # extraction and no specific window is reported.
        assert layer.primitive == "Full"
        assert layer.window is None


def test_single_backend_choice_reflects_fragment_sparsity() -> None:
    """Planner's single-backend estimate should react to sparsity metrics."""

    planner = Planner()
    gates = [
        Gate("H", [0]),
        Gate("RY", [0], params={"theta": 0.2}),
        Gate("CX", [0, 1]),
        Gate("CX", [1, 2]),
        Gate("RY", [2], params={"theta": 0.4}),
        Gate("CX", [2, 3]),
        Gate("RY", [3], params={"theta": 0.6}),
        Gate("CX", [3, 4]),
        Gate("RY", [4], params={"theta": 0.8}),
    ]

    dense_backend, _ = planner._single_backend(
        gates,
        max_memory=None,
        sparsity=0.05,
        phase_rotation_diversity=40,
        amplitude_rotation_diversity=0,
        max_time=1500,
    )
    sparse_backend, _ = planner._single_backend(
        gates,
        max_memory=None,
        sparsity=0.9,
        phase_rotation_diversity=40,
        amplitude_rotation_diversity=0,
        max_time=1500,
    )

    assert dense_backend == Backend.STATEVECTOR
    assert sparse_backend == Backend.DECISION_DIAGRAM
