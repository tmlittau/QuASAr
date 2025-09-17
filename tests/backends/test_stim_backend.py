from __future__ import annotations

import pytest

stim = pytest.importorskip("stim")

from quasar.backends import StimBackend
from quasar.backends.stim_backend import direct_sum


def _ghz_tableau(num_qubits: int) -> stim.Tableau:
    circuit = stim.Circuit()
    circuit.append("H", [0])
    for target in range(1, num_qubits):
        circuit.append("CX", [0, target])
    return stim.Tableau.from_circuit(circuit)


def _bell_tableau() -> stim.Tableau:
    circuit = stim.Circuit()
    circuit.append("H", [0])
    circuit.append("CX", [0, 1])
    return stim.Tableau.from_circuit(circuit)


def test_direct_sum_matches_block_diagonal_tableau() -> None:
    """Direct sums should append independent subsystems without mutation."""

    ghz = _ghz_tableau(3).inverse()
    bell = _bell_tableau().inverse()
    combined = direct_sum(ghz, bell)

    reference = stim.Circuit()
    reference.append("H", [0])
    reference.append("CX", [0, 1])
    reference.append("CX", [0, 2])
    reference.append("H", [3])
    reference.append("CX", [3, 4])
    expected = stim.Tableau.from_circuit(reference).inverse()

    assert len(ghz) == 3
    assert len(bell) == 2
    assert len(combined) == 5
    assert combined == expected


def test_direct_sum_allows_cross_qubit_gate() -> None:
    """Merged tableaus should support Clifford gates across subsystems."""

    ghz = _ghz_tableau(3).inverse()
    bell = _bell_tableau().inverse()
    combined = direct_sum(ghz, bell)

    backend = StimBackend()
    backend.ingest(combined, num_qubits=5)
    backend.apply_gate("CX", (2, 3))

    cross = stim.Circuit()
    cross.append("H", [0])
    cross.append("CX", [0, 1])
    cross.append("CX", [0, 2])
    cross.append("H", [3])
    cross.append("CX", [3, 4])
    cross.append("CX", [2, 3])
    expected = stim.Tableau.from_circuit(cross).inverse()

    tableau = backend.simulator.current_inverse_tableau()
    assert tableau == expected
