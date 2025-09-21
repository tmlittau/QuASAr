"""Tests for serialising :class:`~quasar.circuit.Circuit` objects."""

from __future__ import annotations

import json

from quasar.circuit import Circuit


def _simple_circuit() -> Circuit:
    return Circuit(
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "RZ", "qubits": [1], "params": {"theta": 1.57079632679}},
        ]
    )


def test_circuit_is_json_serialisable() -> None:
    circuit = _simple_circuit()

    encoded = json.dumps(circuit)
    payload = json.loads(encoded)

    assert payload["use_classical_simplification"] is True
    assert [gate["gate"] for gate in payload["gates"]] == ["H", "CX", "RZ"]


def test_circuit_round_trip_via_json(tmp_path) -> None:
    circuit = _simple_circuit()

    json_path = tmp_path / "circuit.json"
    text = circuit.to_json(json_path, indent=2)

    assert json_path.read_text() == text

    restored = Circuit.from_json(json_path)
    assert restored.to_dict(include_metadata=False) == circuit.to_dict(include_metadata=False)
