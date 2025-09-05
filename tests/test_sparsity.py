import pytest
from qiskit import transpile
from qiskit.circuit.random import random_circuit as qiskit_random_circuit

from benchmarks.circuits import qft_circuit, w_state_circuit
from quasar.circuit import Circuit, Gate
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


def test_w_state_sparsity():
    assert w_state_circuit(5).sparsity > 0.8


def test_qft_sparsity():
    assert qft_circuit(5).sparsity < 0.2


def test_random_circuit_sparsity():
    qc = qiskit_random_circuit(5, depth=20, seed=123)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    circ = Circuit.from_qiskit(qc)
    assert circ.sparsity < 0.2
