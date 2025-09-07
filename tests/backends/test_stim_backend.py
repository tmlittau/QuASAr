"""Tests for the Stim backend initialisation."""

from quasar.backends.stim_backend import StimBackend
from quasar.ssd import SSD


def test_run_benchmark_returns_ssd_without_statevector(monkeypatch) -> None:
    """``run_benchmark`` should return an SSD without dense extraction."""
    backend = StimBackend()
    backend.load(1)
    backend.prepare_benchmark()
    backend.apply_gate("H", [0])

    assert backend.simulator is not None  # for mypy
    called = False
    original_sv = backend.simulator.state_vector

    def spy_state_vector(self, *args, **kwargs):
        nonlocal called
        called = True
        return original_sv(*args, **kwargs)

    monkeypatch.setattr(
        type(backend.simulator), "state_vector", spy_state_vector, raising=False
    )

    result = backend.run_benchmark(return_state=True)
    assert not called
    assert isinstance(result, SSD)
    assert result.partitions[0].history == ("H",)


def test_load_and_apply_highest_qubit() -> None:
    """Loading three qubits allows operations on the highest index."""
    backend = StimBackend()
    backend.load(3)
    backend.apply_gate("X", [2])
    assert backend.simulator is not None
    assert backend.simulator.num_qubits == 3

