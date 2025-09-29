"""Regression tests covering DD selection for small Grover circuits."""

from __future__ import annotations

import pytest

import quasar.config as config

from benchmarks.bench_utils.circuits import layered_clifford_delayed_magic_circuit
from benchmarks.circuits import grover_circuit
from quasar.circuit import Circuit
from quasar.cost import Backend, CostEstimator
from quasar.method_selector import MethodSelector
from quasar.sparsity import sparsity_estimate
from quasar.symmetry import (
    amplitude_rotation_diversity,
    phase_rotation_diversity,
)


DELAYED_MAGIC_TEST_DEPTH = 40


def _gate_counts(circuit: Circuit) -> tuple[int, int, int]:
    num_gates = len(circuit.gates)
    num_meas = sum(
        1 for gate in circuit.gates if gate.gate.upper() in {"MEASURE", "RESET"}
    )
    num_1q = sum(
        1
        for gate in circuit.gates
        if len(gate.qubits) == 1 and gate.gate.upper() not in {"MEASURE", "RESET"}
    )
    num_2q = num_gates - num_1q - num_meas
    return num_1q, num_2q, num_meas


@pytest.mark.parametrize("width", [3, 4])
def test_small_grover_prefers_decision_diagrams(width: int) -> None:
    """Ensure DD heuristics evaluate and win for tiny Grover fragments."""

    circuit = grover_circuit(width, 1)
    estimator = CostEstimator()
    selector = MethodSelector(estimator)

    phase_div = phase_rotation_diversity(circuit)
    amp_div = amplitude_rotation_diversity(circuit)
    estimated_sparsity = sparsity_estimate(circuit)
    assert estimated_sparsity > 0.0

    num_1q, num_2q, num_meas = _gate_counts(circuit)
    chi = getattr(estimator, "chi_max", None) or 4
    mps_cost = estimator.mps(
        circuit.num_qubits,
        num_1q + num_meas,
        num_2q,
        chi=chi,
        svd=True,
    )

    for sparsity in (0.0, estimated_sparsity):
        diagnostics: dict[str, object] = {}
        backend, _ = selector.select(
            circuit.gates,
            circuit.num_qubits,
            sparsity=sparsity,
            phase_rotation_diversity=phase_div,
            amplitude_rotation_diversity=amp_div,
            diagnostics=diagnostics,
        )

        assert backend is Backend.DECISION_DIAGRAM

        backends = diagnostics["backends"]
        assert Backend.DECISION_DIAGRAM in backends
        dd_entry = backends[Backend.DECISION_DIAGRAM]
        assert dd_entry["feasible"] is True
        assert dd_entry.get("selected") is True
        dd_cost = dd_entry["cost"]

        assert dd_cost.time <= mps_cost.time


def test_layered_circuit_prefers_mps(monkeypatch) -> None:
    monkeypatch.setattr(
        config.DEFAULT,
        "mps_long_range_fraction_threshold",
        0.95,
        raising=False,
    )
    monkeypatch.setattr(
        config.DEFAULT,
        "mps_long_range_extent_threshold",
        0.95,
        raising=False,
    )
    circuit = layered_clifford_delayed_magic_circuit(
        12, depth=DELAYED_MAGIC_TEST_DEPTH
    )
    estimator = CostEstimator()
    selector = MethodSelector(estimator)

    phase_div = phase_rotation_diversity(circuit)
    amp_div = amplitude_rotation_diversity(circuit)
    estimated_sparsity = sparsity_estimate(circuit)

    diagnostics: dict[str, object] = {}
    backend, cost = selector.select(
        circuit.gates,
        circuit.num_qubits,
        sparsity=estimated_sparsity,
        phase_rotation_diversity=phase_div,
        amplitude_rotation_diversity=amp_div,
        diagnostics=diagnostics,
    )

    assert backend is Backend.MPS

    backends = diagnostics["backends"]
    assert Backend.MPS in backends
    assert Backend.STATEVECTOR in backends

    mps_entry = backends[Backend.MPS]
    sv_entry = backends[Backend.STATEVECTOR]

    assert mps_entry["feasible"] is True
    assert sv_entry["feasible"] is True
    assert mps_entry["cost"].time <= sv_entry["cost"].time
    assert mps_entry["cost"].memory <= sv_entry["cost"].memory
    assert diagnostics["selected_backend"] is Backend.MPS
    assert diagnostics["selected_cost"] == cost

