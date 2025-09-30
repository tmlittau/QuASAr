"""Unit tests for :mod:`quasar.analyzer`."""

from __future__ import annotations

import math

from quasar.analyzer import CircuitAnalyzer
from quasar.circuit import Circuit


def test_analyzer_reuses_topological_order(monkeypatch) -> None:
    """Repeated metric calls reuse the cached topological order."""

    circuit = Circuit(
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "RZ", "qubits": [1], "params": {"theta": math.pi / 2}},
        ],
        use_classical_simplification=False,
    )

    call_count = 0
    original_topological = circuit.topological

    def tracking_topological():
        nonlocal call_count
        call_count += 1
        return original_topological()

    monkeypatch.setattr(circuit, "topological", tracking_topological)

    analyzer = CircuitAnalyzer(circuit)

    first_distribution = analyzer.gate_distribution()
    second_distribution = analyzer.gate_distribution()
    assert first_distribution == second_distribution

    first_entanglement = analyzer.entanglement_metrics()
    second_entanglement = analyzer.entanglement_metrics()
    assert first_entanglement == second_entanglement

    first_rotations = analyzer.rotation_angle_stats()
    second_rotations = analyzer.rotation_angle_stats()
    assert first_rotations == second_rotations

    assert call_count == 1
