import numpy as np

from quasar.backends import StatevectorBackend, MPSBackend


def _mps_to_state(tensors):
    state = tensors[0]
    for t in tensors[1:]:
        state = np.tensordot(state, t, axes=(2, 0))
    return state.reshape(-1)


def _build_reference_state():
    ref = StatevectorBackend()
    ref.load(2)
    ref.apply_gate("H", [0])
    ref.apply_gate("CX", [0, 1])
    ref.apply_gate("S", [0])
    ref.apply_gate("H", [1])
    return ref.state.copy()


def test_statevector_to_mps_roundtrip():
    ref_state = _build_reference_state()

    sv = StatevectorBackend()
    sv.load(2)
    sv.apply_gate("H", [0])
    sv.apply_gate("CX", [0, 1])

    mps = MPSBackend()
    mps.ingest(sv.state)
    mps.apply_gate("S", [0])
    mps.apply_gate("H", [1])

    result = _mps_to_state(mps.tensors)
    assert np.allclose(result, ref_state)


def test_mps_to_statevector_roundtrip():
    ref_state = _build_reference_state()

    mps = MPSBackend()
    mps.load(2)
    mps.apply_gate("H", [0])
    mps.apply_gate("CX", [0, 1])

    sv = StatevectorBackend()
    sv.ingest(_mps_to_state(mps.tensors))
    sv.apply_gate("S", [0])
    sv.apply_gate("H", [1])

    assert np.allclose(sv.state, ref_state)
