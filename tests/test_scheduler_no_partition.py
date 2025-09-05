"""Ensure small circuits remain in a single partition.

These circuits mirror examples from ``tests/notebooks/quasar_api_demo.ipynb``
which use 2- and 3-qubit systems. They should fit into a single partition
when processed by the scheduler or simulation engine.
"""

from quasar import Circuit, SimulationEngine, Scheduler
from qiskit import QuantumCircuit
import numpy as np


def build_bell_circuit() -> Circuit:
    """Bell state on 2 qubits."""
    return Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
    ])


def build_three_qubit_ghz() -> Circuit:
    """3-qubit GHZ-like circuit from the demo notebook."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return Circuit.from_qiskit(qc)


def build_random_two_qubit_circuit() -> Circuit:
    """Random 2-qubit circuit mirroring the notebook example."""
    np.random.seed(0)
    qc = QuantumCircuit(2)
    qc.rx(0.3, 0)
    qc.ry(0.2, 1)
    qc.cz(0, 1)
    qc.rz(0.1, 0)
    qc.h(1)
    return Circuit.from_qiskit(qc)


def test_bell_circuit_single_partition():
    circuit = build_bell_circuit()
    engine = SimulationEngine()
    result = engine.simulate(circuit)
    assert len({p.backend for p in result.ssd.partitions}) == 1
    assert not result.ssd.conversions


def test_prepare_run_single_partition():
    circuit = build_bell_circuit()
    scheduler = Scheduler()
    scheduler.prepare_run(circuit)
    assert len({p.backend for p in circuit.ssd.partitions}) == 1


def test_three_qubit_ghz_single_partition():
    circuit = build_three_qubit_ghz()
    scheduler = Scheduler()
    plan = scheduler.prepare_run(circuit)
    ssd = scheduler.run(circuit, plan)
    assert len({p.backend for p in ssd.partitions}) == 1
    assert not ssd.conversions


def test_random_two_qubit_circuit_single_partition():
    circuit = build_random_two_qubit_circuit()
    engine = SimulationEngine()
    result = engine.simulate(circuit)
    assert len({p.backend for p in result.ssd.partitions}) == 1


def test_fifteen_qubit_circuit_single_backend():
    qc = QuantumCircuit(15)
    for i in range(14):
        qc.cx(i, i + 1)
    circuit = Circuit.from_qiskit(qc)
    engine = SimulationEngine(scheduler=Scheduler(quick_max_qubits=15))
    result = engine.simulate(circuit)
    assert len(result.ssd.partitions) == 1
    assert not result.ssd.conversions
