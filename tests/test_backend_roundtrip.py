import numpy as np
import pytest

from quasar.backends import (
    StatevectorBackend,
    MPSBackend,
    StimBackend,
    DecisionDiagramBackend,
)


def _prepare_backend(backend):
    backend.load(2)
    backend.apply_gate("H", [0])
    backend.apply_gate("CX", [0, 1])
    return backend


def test_statevector_roundtrip():
    b1 = _prepare_backend(StatevectorBackend())
    state = b1.extract_ssd().partitions[0].state
    b2 = StatevectorBackend()
    b2.ingest(state)
    assert np.allclose(b1.statevector(), b2.statevector())


def test_mps_roundtrip():
    b1 = _prepare_backend(MPSBackend())
    state = b1.extract_ssd().partitions[0].state
    assert isinstance(state, (list, tuple))
    b2 = MPSBackend()
    b2.ingest(state)
    assert np.allclose(b1.statevector(), b2.statevector())


def test_stim_roundtrip():
    b1 = _prepare_backend(StimBackend())
    tableau = b1.extract_ssd().partitions[0].state
    expected = b1.statevector()
    b2 = StimBackend()
    b2.ingest(tableau)
    np.testing.assert_allclose(b2.statevector(), expected)


def test_decision_diagram_roundtrip():
    b1 = _prepare_backend(DecisionDiagramBackend())
    state = b1.extract_ssd().partitions[0].state
    b2 = DecisionDiagramBackend()
    b2.ingest(state)
    state2 = b2.extract_ssd().partitions[0].state
    assert state2[0] == state[0]
    assert state2[1] is state[1]
