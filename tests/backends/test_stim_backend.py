"""Tests for the Stim backend initialisation."""

from quasar.backends.stim_backend import StimBackend


def test_run_benchmark_caches_tableau_without_statevector(monkeypatch) -> None:
    """``run_benchmark`` should execute queued ops without dense extraction."""
    backend = StimBackend()
    backend.load(1)
    backend.prepare_benchmark()
    backend.apply_gate("H", [0])

    # Spy on the simulator's state_vector to ensure it is not called
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

    result = backend.run_benchmark()
    assert result is None
    assert backend._benchmark_tableau is not None
    assert not called

    run_called = False

    def fake_run() -> None:
        nonlocal run_called
        run_called = True

    backend.run = fake_run  # type: ignore[assignment]
    ssd = backend.extract_ssd()
    assert not run_called
    assert ssd.partitions[0].history == ("H",)


def test_load_and_apply_highest_qubit() -> None:
    """Loading three qubits allows operations on the highest index."""
    backend = StimBackend()
    backend.load(3)
    backend.apply_gate("X", [2])
    assert backend.simulator is not None
    assert backend.simulator.num_qubits == 3

