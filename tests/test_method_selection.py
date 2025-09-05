from benchmarks.circuits import random_circuit
from quasar import Circuit, Backend, Scheduler


def _prepare(circ: Circuit):
    scheduler = Scheduler(
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
        force_single_backend_below=None,
    )
    plan = scheduler.prepare_run(circ)
    return scheduler, plan


def test_clifford_tableau_selection():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "S", "qubits": [1]},
    ]
    circ = Circuit.from_dict(gates)
    _prepare(circ)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.TABLEAU


def test_small_statevector_selection():
    gates = [
        {"gate": "T", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [1]},
        {"gate": "CX", "qubits": [1, 2]},
    ]
    circ = Circuit.from_dict(gates)
    _prepare(circ)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.DECISION_DIAGRAM


def test_sparse_dd_selection():
    gates = [
        {"gate": "T", "qubits": list(range(20))},
    ]
    circ = Circuit.from_dict(gates)
    _prepare(circ)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.DECISION_DIAGRAM


def test_dense_statevector_selection():
    circ = random_circuit(5, seed=123)
    _prepare(circ)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.STATEVECTOR

