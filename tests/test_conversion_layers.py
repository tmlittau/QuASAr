from quasar import Circuit, Partitioner, Backend
import quasar.config as config


def test_conversion_layer_inserted(monkeypatch):
    monkeypatch.setattr(config.DEFAULT, "dd_sparsity_threshold", 0.0)
    monkeypatch.setattr(config.DEFAULT, "dd_nnz_threshold", 10_000_000)
    monkeypatch.setattr(config.DEFAULT, "dd_phase_rotation_diversity_threshold", 1000)
    monkeypatch.setattr(config.DEFAULT, "dd_amplitude_rotation_diversity_threshold", 1000)
    monkeypatch.setattr(config.DEFAULT, "dd_metric_threshold", 0.0)
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
    assert conv.target == Backend.DECISION_DIAGRAM
    assert set(conv.boundary) == {0, 1, 2, 3, 4}
    assert conv.cost.time > 0

