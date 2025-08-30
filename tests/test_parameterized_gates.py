import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import U1Gate, U2Gate, UGate

from quasar.circuit import Circuit
from quasar.backends import StatevectorBackend, MPSBackend, DecisionDiagramBackend


def _run_backend(cls, circuit):
    backend = cls()
    backend.load(circuit.num_qubits)
    for g in circuit.gates:
        backend.apply_gate(g.gate, g.qubits, g.params)
    return backend


def _to_little_endian(vec: np.ndarray) -> np.ndarray:
    n = int(np.log2(len(vec)))
    return vec.reshape([2] * n).transpose(*reversed(range(n))).reshape(-1)


def test_parameterized_gates():
    qc = QuantumCircuit(2)
    qc.rx(0.1, 0)
    qc.ry(0.2, 1)
    qc.rz(0.3, 0)
    qc.p(0.4, 1)
    qc.append(U1Gate(0.5), [1])
    qc.rzz(0.6, 0, 1)
    qc.append(U2Gate(0.7, 0.8), [0])
    qc.append(UGate(0.9, 1.0, 1.1), [1])

    expected = _to_little_endian(Statevector.from_instruction(qc).data)

    circuit = Circuit.from_qiskit(qc)

    sv = _run_backend(StatevectorBackend, circuit)
    np.testing.assert_allclose(sv.statevector(), expected, atol=1e-8)

    mps = _run_backend(MPSBackend, circuit)
    np.testing.assert_allclose(mps.statevector(), expected, atol=1e-8)

    dd = _run_backend(DecisionDiagramBackend, circuit)
    ssd = dd.extract_ssd()
    assert ssd.partitions[0].history == tuple(g.gate for g in circuit.gates)
