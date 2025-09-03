from quasar import Circuit, Backend


def test_clifford_tableau_selection():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "S", "qubits": [1]},
    ]
    circ = Circuit.from_dict(gates)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.TABLEAU


def test_small_statevector_selection():
    gates = [
        {"gate": "T", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [1]},
        {"gate": "CX", "qubits": [1, 2]},
    ]
    circ = Circuit.from_dict(gates)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.STATEVECTOR


def test_sparse_dd_selection():
    gates = [
        {"gate": "T", "qubits": list(range(20))},
    ]
    circ = Circuit.from_dict(gates)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.DECISION_DIAGRAM


def test_dense_statevector_selection():
    base = [
        {"gate": "T", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 2]},
    ]
    gates = base * 5  # 10 gates > 2**3
    circ = Circuit.from_dict(gates)
    # The partitioner may introduce a conversion from a lightweight method to
    # a dense statevector once the gate count grows. The last partition should
    # therefore use the statevector backend.
    part = circ.ssd.partitions[-1]
    assert part.backend == Backend.STATEVECTOR


def test_partition_multiple_backends():
    # Two identical Clifford subcircuits on qubits (0,1) and (2,3)
    gates = []
    for base in (0, 2):
        gates.extend(
            [
                {"gate": "H", "qubits": [base]},
                {"gate": "CX", "qubits": [base, base + 1]},
                {"gate": "S", "qubits": [base + 1]},
            ]
        )
    # Local non-Clifford chain on qubits (4,5,6) -> MPS
    gates.extend(
        [
            {"gate": "T", "qubits": [4]},
            {"gate": "CX", "qubits": [4, 5]},
            {"gate": "T", "qubits": [5]},
            {"gate": "CX", "qubits": [5, 6]},
            {"gate": "T", "qubits": [6]},
        ]
    )
    # Sparse non-local gates on qubits (7,9) -> decision diagram
    gates.extend(
        [
            {"gate": "T", "qubits": [7]},
            {"gate": "CX", "qubits": [7, 9]},
            {"gate": "T", "qubits": [9]},
        ]
    )
    # Dense, non-local two-qubit pattern on (10,12) -> statevector
    base = [
        {"gate": "T", "qubits": [10]},
        {"gate": "CX", "qubits": [10, 12]},
    ]
    gates.extend(base * 5)  # 10 gates > 2**2

    circ = Circuit.from_dict(gates)
    groups = circ.ssd.by_backend()

    assert set(groups.keys()) == {Backend.TABLEAU, Backend.STATEVECTOR}

    tableau = groups[Backend.TABLEAU][0]
    assert tableau.multiplicity == 2
    assert set(tableau.qubits) == {0, 1, 2, 3}

    state_parts = groups[Backend.STATEVECTOR]
    assert any({4, 5, 6}.issubset(set(p.qubits)) for p in state_parts)
    assert any({7, 9}.issubset(set(p.qubits)) for p in state_parts)
    assert any({10, 12}.issubset(set(p.qubits)) for p in state_parts)

