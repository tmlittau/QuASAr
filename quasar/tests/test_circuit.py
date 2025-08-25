import json
from pathlib import Path

from quasar import Circuit, load_circuit


def test_from_dict_infers_qubits():
    data = [
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [1, 3]},
    ]
    circuit = Circuit.from_dict(data)
    assert circuit.num_qubits == 3
    assert len(circuit.gates) == 2


def test_from_json(tmp_path: Path):
    data = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
    ]
    path = tmp_path / "circ.json"
    path.write_text(json.dumps(data))
    circuit = Circuit.from_json(path)
    assert circuit.num_qubits == 2
    assert circuit.gates[1].name == "CX"


def test_load_circuit_accepts_path_or_data(tmp_path: Path):
    dict_data = [{"gate": "H", "qubits": [5]}]
    circ1 = load_circuit(dict_data)
    assert circ1.num_qubits == 1

    json_path = tmp_path / "circ.json"
    json_path.write_text(json.dumps(dict_data))
    circ2 = load_circuit(json_path)
    assert circ2.num_qubits == 1


def test_placeholders_return_defaults():
    circuit = Circuit.from_dict([])
    assert circuit.ssd is None
    assert circuit.cost == 0
