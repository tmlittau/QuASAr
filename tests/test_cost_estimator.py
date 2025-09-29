from __future__ import annotations

import pytest

from benchmarks.bench_utils.circuits import layered_clifford_delayed_magic_circuit

from quasar.cost import CostEstimator


DELAYED_MAGIC_TEST_DEPTH = 40


def test_bond_dimensions_respect_local_schmidt_cap() -> None:
    circuit = layered_clifford_delayed_magic_circuit(
        12, depth=DELAYED_MAGIC_TEST_DEPTH
    )
    estimator = CostEstimator()

    gates = list(circuit.gates)
    bonds = estimator.bond_dimensions(circuit.num_qubits, gates)
    assert estimator.max_schmidt_rank(circuit.num_qubits, gates) == 64

    local_caps = [
        2 ** min(i + 1, circuit.num_qubits - i - 1)
        for i in range(max(0, circuit.num_qubits - 1))
    ]

    assert bonds
    assert len(bonds) == len(local_caps)
    for idx, (bond, cap) in enumerate(zip(bonds, local_caps)):
        assert bond <= cap, f"Cut {idx} exceeds local Schmidt cap"


def _gate_counts(circuit) -> tuple[int, int, int]:
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


def test_scalar_chi_matches_per_cut_cap() -> None:
    circuit = layered_clifford_delayed_magic_circuit(
        12, depth=DELAYED_MAGIC_TEST_DEPTH
    )
    estimator = CostEstimator()

    num_1q, num_2q, num_meas = _gate_counts(circuit)

    chi = 64
    per_cut_caps = [
        2 ** min(i + 1, circuit.num_qubits - i - 1)
        for i in range(max(0, circuit.num_qubits - 1))
    ]
    per_cut_chi = [min(chi, cap) for cap in per_cut_caps]

    scalar_cost = estimator.mps(
        circuit.num_qubits,
        num_1q + num_meas,
        num_2q,
        chi=chi,
        svd=True,
    )
    list_cost = estimator.mps(
        circuit.num_qubits,
        num_1q + num_meas,
        num_2q,
        chi=per_cut_chi,
        svd=True,
    )

    assert scalar_cost.time == pytest.approx(list_cost.time)
    assert scalar_cost.memory == pytest.approx(list_cost.memory)
    assert scalar_cost.log_depth == pytest.approx(list_cost.log_depth)
    assert scalar_cost.conversion == pytest.approx(list_cost.conversion)
    assert scalar_cost.replay == pytest.approx(list_cost.replay)
