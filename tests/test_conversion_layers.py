from quasar import Circuit, Partitioner, Backend


def test_conversion_layer_inserted():
    # Construct a circuit that starts Clifford-only and then introduces
    # non-Clifford gates on the same qubits, forcing a backend switch and
    # a conversion layer. The circuit size is deliberately large to ensure
    # heuristics trigger even on conservative thresholds.
    gates = []

    # Stage 1: Clifford entangling on 5 qubits
    for q in range(5):
        gates.append({"gate": "H", "qubits": [q]})
    for q in range(4):
        gates.append({"gate": "CX", "qubits": [q, q + 1]})

    # Stage 2: Non-Clifford gates on all qubits
    for q in range(5):
        gates.append({"gate": "T", "qubits": [q]})

    circ = Circuit.from_dict(gates)
    ssd = Partitioner().partition(circ)

    assert len(ssd.partitions) == 2
    assert len(ssd.conversions) == 1

    conv = ssd.conversions[0]
    assert conv.source == Backend.TABLEAU
    assert conv.target == Backend.MPS
    assert set(conv.boundary) == {0, 1, 2, 3, 4}
    assert conv.cost.time > 0

