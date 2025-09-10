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


def test_graph_cut_prefers_smaller_boundary():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [1]},
    ]
    circ = Circuit.from_dict(gates)
    ssd_h = Partitioner().partition(circ)
    ssd_g = Partitioner().partition(circ, graph_cut=True)
    assert len(ssd_h.conversions) == len(ssd_g.conversions) == 1
    assert len(ssd_g.conversions[0].boundary) <= len(ssd_h.conversions[0].boundary)


def test_graph_cut_avoids_unnecessary_conversion():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "H", "qubits": [2]},
        {"gate": "CX", "qubits": [2, 3]},
        {"gate": "T", "qubits": [2]},
    ]
    circ = Circuit.from_dict(gates)
    ssd_h = Partitioner().partition(circ)
    ssd_g = Partitioner().partition(circ, graph_cut=True)
    assert len(ssd_h.conversions) == 1
    assert ssd_h.conversions[0].boundary == (2,)
    assert len(ssd_g.conversions) == 0
    assert any(
        set(group) == {0, 1}
        for part in ssd_g.partitions
        for group in part.subsystems
    )
