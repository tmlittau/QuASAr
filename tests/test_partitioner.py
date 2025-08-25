from quasar import Circuit, Partitioner


def test_compress_equal_histories():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "H", "qubits": [2]},
    ]
    circ = Circuit.from_dict(gates)
    ssd = Partitioner().partition(circ)
    assert len(ssd.partitions) == 1
    part = ssd.partitions[0]
    assert part.multiplicity == 3
    qubits = {q for group in part.subsystems for q in group}
    assert qubits == {0, 1, 2}


def test_entangling_gate_merges_partitions():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "H", "qubits": [2]},
    ]
    circ = Circuit.from_dict(gates)
    ssd = Partitioner().partition(circ)
    assert len(ssd.partitions) == 2
    groups = [set(group) for p in ssd.partitions for group in p.subsystems]
    assert {0, 1} in groups
    assert {2} in groups
