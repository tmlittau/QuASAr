import numpy as np
from quasar.backends import MPSBackend, StatevectorBackend


def test_mps_controlled_gates_match_statevector_backend():
    mps = MPSBackend()
    mps.load(2)
    sv = StatevectorBackend()
    sv.load(2)
    # create superposition on control qubit
    mps.apply_gate("H", [0])
    sv.apply_gate("H", [0])
    # apply a controlled RY rotation
    theta = 0.432
    cry_params = {"param0": theta}
    mps.apply_gate("CRY", [0, 1], cry_params)
    sv.apply_gate("CRY", [0, 1], cry_params)
    # apply a controlled RZ rotation
    phi = 0.123
    crz_params = {"param0": phi}
    mps.apply_gate("CRZ", [0, 1], crz_params)
    sv.apply_gate("CRZ", [0, 1], crz_params)
    np.testing.assert_allclose(mps.statevector(), sv.statevector(), atol=1e-12)


def test_run_benchmark_caches_state(monkeypatch):
    backend = MPSBackend()
    backend.load(1)
    backend.prepare_benchmark()
    backend.apply_gate("H", [0])

    calls = {"count": 0}

    def fake_run(self):
        calls["count"] += 1
        return np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])

    monkeypatch.setattr(MPSBackend, "_run", fake_run)

    state = backend.run_benchmark()
    assert calls["count"] == 1

    ssd = backend.extract_ssd()
    assert calls["count"] == 1
    np.testing.assert_allclose(state, ssd.partitions[0].state)

