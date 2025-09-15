import numpy as np
from benchmarks.circuits import hhl_circuit


def _extract_state(circuit):
    gate = circuit.gates[0]
    size = len(gate.params)
    vec = np.array([gate.params[f"param{i}"] for i in range(size)])
    return vec / np.linalg.norm(vec)


def test_hhl_2x2():
    A = np.array([[1, 0.5], [0.5, 1]], dtype=complex)
    b = np.array([1, 0], dtype=complex)
    circ = hhl_circuit(A, b)
    state = _extract_state(circ)
    classical = np.linalg.solve(A, b)
    classical /= np.linalg.norm(classical)
    assert np.allclose(state, classical, atol=1e-8) or np.allclose(state, -classical, atol=1e-8)


def test_hhl_4x4():
    A = np.diag([1, 2, 3, 4]).astype(complex)
    b = np.array([1, 0, 0, 0], dtype=complex)
    circ = hhl_circuit(A, b)
    state = _extract_state(circ)
    classical = np.linalg.solve(A, b)
    classical /= np.linalg.norm(classical)
    assert np.allclose(state, classical, atol=1e-8) or np.allclose(state, -classical, atol=1e-8)
