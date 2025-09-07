import math
import numpy as np

from quasar.circuit import Circuit, Gate
from quasar.backends.statevector import StatevectorBackend
from benchmarks.circuits import (
    ghz_circuit,
    w_state_circuit,
    grover_circuit,
    bernstein_vazirani_circuit,
    qft_circuit,
)


def _simulate(circ: Circuit, n: int) -> np.ndarray:
    backend = StatevectorBackend()
    backend.load(n)
    for g in circ.gates:
        backend.apply_gate(g.gate, g.qubits, g.params)
    return backend.statevector()


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


def test_qft_circuit_respects_flag():
    circ = qft_circuit(2, use_classical_simplification=True)
    assert circ.use_classical_simplification is True
    circ2 = qft_circuit(2)
    assert circ2.use_classical_simplification is False


def test_w_state_circuit_state():
    n = 3
    circ = w_state_circuit(n)
    state = _simulate(circ, n)
    expected = np.zeros(2**n, dtype=complex)
    for i in range(n):
        expected[1 << i] = 1 / math.sqrt(n)
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
