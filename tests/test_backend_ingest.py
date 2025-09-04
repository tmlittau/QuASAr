import numpy as np
import stim

from quasar.backends import StatevectorBackend, MPSBackend, StimBackend


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


def test_backends_ingest_with_mapping_preserves_register():
    ref_sv = StatevectorBackend()
    ref_sv.load(13)
    ref_sv.apply_gate("X", [12])
    ref_state = ref_sv.statevector()

    sv = StatevectorBackend()
    sv.ingest([0, 1], num_qubits=13, mapping=[12])
    assert np.allclose(sv.statevector(), ref_state)

    mps = MPSBackend()
    mps.ingest([0, 1], num_qubits=13, mapping=[12])
    assert np.allclose(mps.statevector(), ref_state)

    ref_stim = StimBackend()
    ref_stim.load(13)
    ref_stim.apply_gate("X", [12])
    ref_stim_state = ref_stim.statevector()

    sim = stim.TableauSimulator()
    sim.x(0)
    stim_b = StimBackend()
    stim_b.ingest(sim, num_qubits=13, mapping=[12])
    assert np.allclose(stim_b.statevector(), ref_stim_state)
