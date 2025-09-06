import json
import math

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


def test_sparsity_attribute():
    circ = Circuit.from_dict([
        {"gate": "H", "qubits": [0]},
        {"gate": "X", "qubits": [1]},
    ])
    assert circ.sparsity == 0.5


def test_classical_gate_simplification():
    circ = Circuit.from_dict([
        {"gate": "X", "qubits": [0]},
        {"gate": "Z", "qubits": [0]},
        {"gate": "H", "qubits": [0]},
        {"gate": "X", "qubits": [0]},
    ])
    assert [g.gate for g in circ.gates] == ["H", "X"]
    assert circ.classical_state == [None]


def test_controlled_classical_simplification():
    circ = Circuit.from_dict([
        {"gate": "X", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
    ])
    assert circ.gates == []
    assert circ.classical_state == [1, 1]


def test_rx_branching():
    circ = Circuit.from_dict([
        {"gate": "RX", "qubits": [0], "params": {"param0": math.pi}},
        {"gate": "RX", "qubits": [1], "params": {"param0": math.pi / 2}},
    ])
    assert [g.gate for g in circ.gates] == ["RX"]
    assert circ.classical_state == [1, None]
