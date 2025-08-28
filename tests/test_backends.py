import pytest

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
    assert ssd.extract_state(ssd.partitions[0]) is not None


def test_statevector_backend():
    _exercise_backend(StatevectorBackend)


def test_mps_backend():
    _exercise_backend(MPSBackend)


def test_stim_backend():
    _exercise_backend(StimBackend)


def test_decision_diagram_backend():
    _exercise_backend(DecisionDiagramBackend)
