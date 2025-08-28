import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector

from quasar.circuit import Circuit, Gate
from quasar.backends.statevector import StatevectorBackend
from benchmarks.circuits import (
    ghz_circuit,
    w_state_circuit,
    qft_circuit,
    qft_on_ghz_circuit,
    grover_circuit,
    bernstein_vazirani_circuit,
)


def _simulate(circ: Circuit, n: int) -> np.ndarray:
    backend = StatevectorBackend()
    backend.load(n)
    for g in circ.gates:
        backend.apply_gate(g.gate, g.qubits, g.params)
    return backend.state


def _assert_equivalent(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-8) -> None:
    idx = np.flatnonzero(expected)[0]
    phase = actual[idx] / expected[idx]
    assert np.allclose(actual / phase, expected, atol=atol)


def test_ghz_circuit_state():
    n = 3
    circ = ghz_circuit(n)
    state = _simulate(circ, n)
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1 / math.sqrt(2)
    expected[-1] = 1 / math.sqrt(2)
    _assert_equivalent(state, expected)


def test_w_state_circuit_state():
    n = 3
    circ = w_state_circuit(n)
    state = _simulate(circ, n)
    expected = np.zeros(2**n, dtype=complex)
    for i in range(n):
        expected[1 << i] = 1 / math.sqrt(n)
    _assert_equivalent(state, expected)


def test_qft_circuit_matches_qiskit():
    n = 3
    init = [Gate("X", [0])]
    circ = Circuit(init + list(qft_circuit(n).gates))
    state = _simulate(circ, n)

    qc = QuantumCircuit(n)
    qc.x(0)
    qc.append(QFT(n), range(n))
    expected = Statevector.from_instruction(qc).data
    _assert_equivalent(state, expected)


def test_qft_on_ghz_circuit_matches_qiskit():
    n = 3
    circ = qft_on_ghz_circuit(n)
    state = _simulate(circ, n)

    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(i - 1, i)
    qc.append(QFT(n), range(n))
    expected = Statevector.from_instruction(qc).data
    _assert_equivalent(state, expected)


def test_grover_circuit_state():
    n = 2
    circ = grover_circuit(n, 1)
    state = _simulate(circ, n)
    expected = np.zeros(2**n, dtype=complex)
    expected[-1] = 1
    _assert_equivalent(state, expected)


def test_bernstein_vazirani_circuit_state():
    n = 3
    secret = 0b101
    circ = bernstein_vazirani_circuit(n, secret)
    state = _simulate(circ, n + 1)
    expected = np.zeros(2 ** (n + 1), dtype=complex)
    idx0 = secret << 1
    idx1 = idx0 | 1
    expected[idx0] = 1 / math.sqrt(2)
    expected[idx1] = -1 / math.sqrt(2)
    _assert_equivalent(state, expected)
