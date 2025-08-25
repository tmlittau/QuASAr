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


def test_local_mps_selection():
    gates = [
        {"gate": "T", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [1]},
        {"gate": "CX", "qubits": [1, 2]},
    ]
    circ = Circuit.from_dict(gates)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.MPS


def test_sparse_dd_selection():
    gates = [
        {"gate": "T", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 2]},
        {"gate": "T", "qubits": [2]},
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
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.STATEVECTOR
