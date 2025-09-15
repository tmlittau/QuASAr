import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from quasar.circuit import Circuit, Gate


def _matrix_from_gates(gates, num_qubits):
    qc = QuantumCircuit(num_qubits)
    for gate in gates:
        name = gate.gate.upper()
        if name == "H":
            qc.h(gate.qubits[0])
        elif name == "T":
            qc.t(gate.qubits[0])
        elif name == "TDG":
            qc.tdg(gate.qubits[0])
        elif name == "CX":
            qc.cx(gate.qubits[0], gate.qubits[1])
        else:  # pragma: no cover - unsupported gate in test
            raise ValueError(f"Unsupported gate {name}")
    return Operator(qc).data


def _expected_mcx(controls, target, ancillas, num_qubits):
    qc = QuantumCircuit(num_qubits)
    qc.mcx(controls, target, ancillas, mode="v-chain")
    return Operator(qc).data


def _max_index(gates):
    return max((q for g in gates for q in g.qubits), default=-1)


def test_mcx_three_controls():
    controls = [0, 1, 2]
    target = 3
    circ = Circuit([Gate("MCX", controls + [target])], use_classical_simplification=False)
    assert all(g.gate.upper() not in {"MCX", "CCX"} for g in circ.gates)
    n = _max_index(circ.gates) + 1
    ancillas = list(range(target + 1, n))
    decomp = _matrix_from_gates(circ.gates, n)
    expected = _expected_mcx(controls, target, ancillas, n)
    assert np.allclose(decomp, expected)


def test_mcx_four_controls():
    controls = [0, 1, 2, 3]
    target = 4
    circ = Circuit([Gate("MCX", controls + [target])], use_classical_simplification=False)
    assert all(g.gate.upper() not in {"MCX", "CCX"} for g in circ.gates)
    n = _max_index(circ.gates) + 1
    ancillas = list(range(target + 1, n))
    decomp = _matrix_from_gates(circ.gates, n)
    expected = _expected_mcx(controls, target, ancillas, n)
    assert np.allclose(decomp, expected)
