import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from quasar.circuit import Gate, Circuit


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
        else:  # pragma: no cover - not expected in this test
            raise ValueError(f"Unsupported gate {name}")
    return Operator(qc).data


def test_ccz_decomposition_unitary() -> None:
    """Decomposed CCZ should match the native controlled-controlled-Z."""
    for n in range(3, 6):
        circ = Circuit([Gate("CCZ", [0, 1, 2])], use_classical_simplification=False)
        assert all(g.gate.upper() != "CCZ" for g in circ.gates)
        decomp = _matrix_from_gates(circ.gates, n)
        native_circuit = QuantumCircuit(n)
        native_circuit.ccz(0, 1, 2)
        native = Operator(native_circuit).data
        assert np.allclose(decomp, native)

