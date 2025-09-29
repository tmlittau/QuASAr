from __future__ import annotations

from benchmarks.bench_utils.circuits import layered_clifford_delayed_magic_circuit

from quasar.cost import CostEstimator


def test_bond_dimensions_respect_local_schmidt_cap() -> None:
    circuit = layered_clifford_delayed_magic_circuit(12)
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
