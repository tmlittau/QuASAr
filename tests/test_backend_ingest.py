import numpy as np

from quasar.backends import StatevectorBackend, MPSBackend


def _build_reference_state():
    ref = StatevectorBackend()
    ref.load(2)
    ref.apply_gate("H", [0])
    ref.apply_gate("CX", [0, 1])
    ref.apply_gate("S", [0])
    ref.apply_gate("H", [1])
    return ref.statevector()


def test_statevector_to_mps_roundtrip():
    ref_state = _build_reference_state()

    sv = StatevectorBackend()
    sv.load(2)
    sv.apply_gate("H", [0])
    sv.apply_gate("CX", [0, 1])

    mps = MPSBackend()
    mps.ingest(sv.statevector())
    mps.apply_gate("S", [0])
    mps.apply_gate("H", [1])

    assert np.allclose(mps.statevector(), ref_state)


def test_mps_to_statevector_roundtrip():
    ref_state = _build_reference_state()

    mps = MPSBackend()
    mps.load(2)
    mps.apply_gate("H", [0])
    mps.apply_gate("CX", [0, 1])

    sv = StatevectorBackend()
    sv.ingest(mps.statevector())
    sv.apply_gate("S", [0])
    sv.apply_gate("H", [1])

    assert np.allclose(sv.statevector(), ref_state)
