import pytest
from quasar.backends import DecisionDiagramBackend


def test_mqt_dd_benchmark_uses_cached_state(monkeypatch):
    backend = DecisionDiagramBackend()
    backend.load(1)
    backend.prepare_benchmark()
    backend.apply_gate("H", [0])

    def fail_statevector():  # pragma: no cover - should not be called
        raise AssertionError("statevector called during run_benchmark")

    monkeypatch.setattr(backend, "statevector", fail_statevector)
    original_extract = backend.extract_ssd

    def fail_extract():  # pragma: no cover - should not be called
        raise AssertionError("extract_ssd called during run_benchmark")

    monkeypatch.setattr(backend, "extract_ssd", fail_extract)

    run_calls = {"n": 0}
    original_run = backend.run

    def run_spy() -> None:
        run_calls["n"] += 1
        return original_run()

    monkeypatch.setattr(backend, "run", run_spy)

    result = backend.run_benchmark()
    assert result is None
    assert run_calls["n"] == 1
    state = backend._benchmark_state
    assert state is backend.state

    monkeypatch.setattr(backend, "extract_ssd", original_extract)

    def fail_run() -> None:  # pragma: no cover - should not be called
        raise AssertionError("run invoked despite cached state")

    monkeypatch.setattr(backend, "run", fail_run)
    ssd = backend.extract_ssd()
    assert ssd.partitions[0].state[1] is state
