from __future__ import annotations

import pytest

from benchmarks.bench_utils.circuits import layered_clifford_delayed_magic_circuit

from quasar.circuit import Gate
from quasar.cost import Backend, CostEstimator


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


def test_bond_dimensions_remap_noncontiguous_qubits() -> None:
    estimator = CostEstimator()
    gates = [Gate("CX", [7, 8])]

    bonds = estimator.bond_dimensions(2, gates)

    assert bonds == [2]


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


def test_conversion_window_reflects_entanglement() -> None:
    estimator = CostEstimator()

    dense_details = estimator.conversion_candidates(
        Backend.STATEVECTOR,
        Backend.MPS,
        num_qubits=6,
        rank=64,
        frontier=6,
        compressed_terms=64,
    )
    assert dense_details["LW"].window == 6

    sparse_details = estimator.conversion_candidates(
        Backend.STATEVECTOR,
        Backend.MPS,
        num_qubits=6,
        rank=2,
        frontier=6,
        compressed_terms=2,
    )
    assert sparse_details["LW"].window == 4

    selected_primitive, selected_detail = min(
        dense_details.items(), key=lambda kv: kv[1].cost.time
    )
    estimate = estimator.conversion(
        Backend.STATEVECTOR,
        Backend.MPS,
        num_qubits=6,
        rank=64,
        frontier=6,
        compressed_terms=64,
    )
    assert estimate.primitive == selected_primitive
    assert estimate.window == selected_detail.window
    assert estimate.ingest_terms == 64


def test_staged_conversion_respects_cap_hint() -> None:
    estimator = CostEstimator()
    rank = 20
    cap_hint = 8
    details = estimator.conversion_candidates(
        Backend.STATEVECTOR,
        Backend.STATEVECTOR,
        num_qubits=6,
        rank=rank,
        frontier=6,
        chi_cap=cap_hint,
    )
    staged = details["ST"]
    assert staged.chi_cap == cap_hint
    assert staged.stages == 3
    stage_coeff = estimator.coeff["st_stage"]
    remaining = rank
    dims = []
    while remaining > 0:
        dims.append(min(cap_hint, remaining))
        remaining -= cap_hint
    expected_stage = stage_coeff * sum(dim**3 for dim in dims)
    assert staged.components["stage"] == pytest.approx(expected_stage)
    ingest = estimator.coeff["ingest_sv"] * (2**6)
    base = estimator.coeff.get("conversion_base", 0.0)
    assert staged.cost.time == pytest.approx(expected_stage + ingest + base)


def test_staged_conversion_single_stage_when_cap_suffices() -> None:
    estimator = CostEstimator()
    rank = 12
    cap_hint = 32
    details = estimator.conversion_candidates(
        Backend.STATEVECTOR,
        Backend.STATEVECTOR,
        num_qubits=5,
        rank=rank,
        frontier=5,
        chi_cap=cap_hint,
    )
    staged = details["ST"]
    assert staged.chi_cap == cap_hint
    assert staged.stages == 1
    stage_coeff = estimator.coeff["st_stage"]
    expected_stage = stage_coeff * (rank**3)
    assert staged.components["stage"] == pytest.approx(expected_stage)
