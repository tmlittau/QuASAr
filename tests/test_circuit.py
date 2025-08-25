import json
from quasar import Circuit, SSD


def example_gates():
    return [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "RY", "qubits": [0], "params": {"theta": 0.7071}},
    ]


def test_circuit_from_dict():
    circ = Circuit.from_dict(example_gates())
    assert circ.num_qubits == 2
    assert len(circ.gates) == 3
    assert isinstance(circ.ssd, SSD)
    # Initial Clifford block followed by a non-Clifford gate triggers a
    # backend switch and thus two partitions.
    assert len(circ.ssd.partitions) == 2


def test_circuit_from_json(tmp_path):
    path = tmp_path / "circuit.json"
    with open(path, "w", encoding="utf8") as f:
        json.dump(example_gates(), f)
    circ = Circuit.from_json(str(path))
    assert circ.num_qubits == 2
    assert [g.gate for g in circ.gates] == ["H", "CX", "RY"]


def test_qubit_inference_from_one_based_indexing():
    circ = Circuit.from_dict([
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [1, 2]},
    ])
    assert circ.num_qubits == 2
