import pytest

from quasar.circuit import Gate, Circuit
from quasar.sparsity import sparsity_estimate


def test_single_h_on_one_qubit():
    circ = Circuit([Gate("H", [0])])
    assert sparsity_estimate(circ) == 0.0


def test_h_on_first_of_two_qubits():
    circ = Circuit([Gate("H", [0]), Gate("X", [1])])
    assert sparsity_estimate(circ) == 0.5


def test_h_then_controlled_ry():
    circ = Circuit([Gate("H", [0]), Gate("CRY", [0, 1])])
    assert sparsity_estimate(circ) == pytest.approx(0.25)


def test_h_then_cx():
    circ = Circuit([Gate("H", [0]), Gate("CX", [0, 1])])
    assert sparsity_estimate(circ) == 0.5


def test_clamp_to_full_dimension():
    circ = Circuit([Gate("H", [0]), Gate("H", [0])])
    assert sparsity_estimate(circ) == 0.0
