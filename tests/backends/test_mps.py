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


def test_mps_benchmark_uses_cached_state(monkeypatch):
    backend = MPSBackend()
    backend.load(1)
    backend.prepare_benchmark()
    backend.apply_gate("H", [0])

    def fail_apply(*_: object, **__: object) -> None:  # pragma: no cover - should not run
        raise AssertionError("apply_gate called during run_benchmark")

    monkeypatch.setattr(backend, "apply_gate", fail_apply)

    original_run = backend._run
    run_calls = {"n": 0}

    def run_spy() -> object:
        run_calls["n"] += 1
        return original_run()

    monkeypatch.setattr(backend, "_run", run_spy)
    state = backend.run_benchmark()
    assert run_calls["n"] == 1

    def fail_run() -> object:  # pragma: no cover - should not run
        raise AssertionError("_run invoked despite cached state")

    monkeypatch.setattr(backend, "_run", fail_run)

    ssd = backend.extract_ssd()
    assert ssd.partitions[0].state is state

    conv_calls = {"n": 0}
    original_conv = backend._mps_to_statevector

    def conv_spy(mps: object) -> np.ndarray:
        conv_calls["n"] += 1
        return original_conv(mps)

    monkeypatch.setattr(backend, "_mps_to_statevector", conv_spy)

    vec1 = backend.statevector()
    vec2 = backend.statevector()
    assert conv_calls["n"] == 1
    np.testing.assert_allclose(vec1, vec2)

