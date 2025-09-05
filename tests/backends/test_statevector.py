import numpy as np
import pytest
from quasar.backends import StatevectorBackend


def test_statevector_benchmark_uses_cached_state(monkeypatch):
    backend = StatevectorBackend()
    backend.load(1)
    backend.prepare_benchmark()
    backend.apply_gate("H", [0])
    state = backend.run_benchmark()

    def fail_run():  # pragma: no cover - should not be called
        raise AssertionError("_run invoked despite cached state")

    monkeypatch.setattr(backend, "_run", fail_run)

    ssd = backend.extract_ssd()
    np.testing.assert_allclose(ssd.partitions[0].state, state)

    vec = backend.statevector()
    np.testing.assert_allclose(vec, state)
