import math
import random

from benchmarks.circuits import qft_circuit, w_state_circuit
from quasar.circuit import Circuit, Gate


def random_circuit(n_qubits: int, depth: int, seed: int = 0) -> Circuit:
    random.seed(seed)
    gates = []
    for _ in range(depth):
        gate = random.choice(["RX", "RY", "RZ", "CRX", "CRY", "CRZ"])
        angle = random.random() * math.pi
        if gate.startswith("C"):
            q1 = random.randrange(n_qubits)
            q2 = random.randrange(n_qubits)
            while q2 == q1:
                q2 = random.randrange(n_qubits)
            gates.append(Gate(gate, [q1, q2], {"theta": angle}))
        else:
            q = random.randrange(n_qubits)
            gates.append(Gate(gate, [q], {"theta": angle}))
    return Circuit(gates)


def test_qft_has_high_symmetry():
    circ = qft_circuit(5)
    assert circ.symmetry > 0.7


def test_w_state_has_high_symmetry():
    circ = w_state_circuit(5)
    assert circ.symmetry > 0.3


def test_random_circuit_has_low_symmetry():
    circ = random_circuit(5, 20, seed=123)
    assert circ.symmetry < 0.2
