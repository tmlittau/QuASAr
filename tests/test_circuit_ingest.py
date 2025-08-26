import math

from qiskit import QuantumCircuit

from quasar import Circuit, SSD


def test_from_qiskit_basic():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.5, 0)
    circ = Circuit.from_qiskit(qc)
    assert circ.num_qubits == 2
    assert [g.gate for g in circ.gates] == ["H", "CX", "RY"]
    assert math.isclose(circ.gates[2].params["param0"], 0.5)
    assert isinstance(circ.ssd, SSD)


def _qasm_snippet():
    return """
OPENQASM 3.0;
include \"stdgates.inc\";
qubit[2] q;
h q[0];
cx q[0], q[1];
""".strip()


def test_from_qasm_string():
    qasm = _qasm_snippet()
    circ = Circuit.from_qasm(qasm)
    assert circ.num_qubits == 2
    assert [g.gate for g in circ.gates] == ["H", "CX"]


def test_from_qasm_file(tmp_path):
    qasm = _qasm_snippet()
    path = tmp_path / "circ.qasm"
    path.write_text(qasm)
    circ = Circuit.from_qasm(str(path))
    assert circ.num_qubits == 2
    assert [g.gate for g in circ.gates] == ["H", "CX"]
