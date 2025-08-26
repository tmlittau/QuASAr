import pytest
import numpy as np
import stim
import mqt.ddsim as ddsim
from mqt.core.ir import QuantumComputation

from quasar.backends import (
    StatevectorBackend,
    MPSBackend,
    StimBackend,
    DecisionDiagramBackend,
)


def _exercise_backend(backend_cls):
    backend = backend_cls()
    backend.load(2)
    backend.apply_gate("H", [0])
    backend.apply_gate("CX", [0, 1])
    ssd = backend.extract_ssd()
    assert ssd.partitions[0].backend == backend.backend
    assert ssd.partitions[0].history == ("H", "CX")


def test_statevector_backend():
    _exercise_backend(StatevectorBackend)


def test_mps_backend():
    _exercise_backend(MPSBackend)


def test_stim_backend():
    _exercise_backend(StimBackend)


def test_decision_diagram_backend():
    _exercise_backend(DecisionDiagramBackend)


def test_statevector_ingest():
    state = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    backend = StatevectorBackend()
    backend.ingest_state(state)
    assert backend.num_qubits == 2
    assert np.allclose(backend.state, state)


def test_mps_ingest():
    tensors = [np.zeros((1, 2, 1), dtype=complex) for _ in range(2)]
    tensors[0][0, 0, 0] = 1.0
    tensors[1][0, 1, 0] = 1.0
    backend = MPSBackend()
    backend.ingest_state(tensors)
    assert backend.num_qubits == 2
    assert np.allclose(backend.tensors[1], tensors[1])


def test_stim_ingest():
    circuit = stim.Circuit("H 0\nCX 0 1")
    tableau = stim.Tableau.from_circuit(circuit)
    backend = StimBackend()
    backend.ingest_state(tableau)
    assert backend.num_qubits == 2
    assert backend.simulator.current_inverse_tableau() == tableau


def test_decision_diagram_ingest():
    qc = QuantumComputation(2)
    qc.h(0)
    qc.cx(0, 1)
    sim = ddsim.CircuitSimulator(qc)
    state = sim.get_constructed_dd()
    backend = DecisionDiagramBackend()
    backend.load(2)
    backend.ingest_state(state)
    assert backend.state == state
    assert backend.num_qubits == 2
