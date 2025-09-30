"""Planner regression tests."""

from __future__ import annotations

import math

import pytest

from quasar.circuit import Circuit, Gate
from quasar.planner import Planner


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


def test_dp_receives_single_backend_upper_bound(monkeypatch) -> None:
    """The DP passes should receive the single-backend upper bound when available."""

    circuit = Circuit(
        [
            Gate("H", [0]),
            Gate("CX", [0, 1]),
            Gate("T", [1]),
            Gate("MEASURE", [0]),
        ],
        use_classical_simplification=True,
    )

    baseline_planner = Planner(
        quick_max_qubits=0,
        quick_max_gates=0,
        quick_max_depth=0,
    )
    baseline_result = baseline_planner.plan(circuit, use_cache=False, explain=True)
    baseline_steps = [
        (step.start, step.end, step.backend) for step in baseline_result.steps
    ]
    baseline_backend = baseline_result.final_backend

    observed_bounds: dict[str, object] = {}
    original_dp = Planner._dp

    def spy(self: Planner, *args, **kwargs):  # type: ignore[override]
        stage = kwargs.get("stage")
        if stage in {"pre", "coarse"}:
            observed_bounds[stage] = kwargs.get("upper_bound")
        return original_dp(self, *args, **kwargs)

    planner = Planner(
        quick_max_qubits=0,
        quick_max_gates=0,
        quick_max_depth=0,
    )
    monkeypatch.setattr(Planner, "_dp", spy)

    result = planner.plan(circuit, use_cache=False, explain=True)

    assert result.diagnostics is not None
    single_cost = result.diagnostics.single_cost
    assert single_cost is not None
    assert math.isfinite(single_cost.time)
    assert math.isfinite(single_cost.memory)

    for stage in ("pre", "coarse"):
        assert stage in observed_bounds
        bound = observed_bounds[stage]
        assert bound is not None
        assert bound.time == pytest.approx(single_cost.time)
        assert bound.memory == pytest.approx(single_cost.memory)

    steps = [(step.start, step.end, step.backend) for step in result.steps]
    assert steps == baseline_steps
    assert result.final_backend == baseline_backend
