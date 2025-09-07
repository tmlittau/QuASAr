from __future__ import annotations

from quasar.circuit import Circuit
from quasar.scheduler import Scheduler
from quasar.cost import Backend


class DummyBackend:
    def load(self, n):  # pragma: no cover - trivial
        pass

    def apply_gate(self, gate, qubits, params):  # pragma: no cover - trivial
        pass

    def extract_ssd(self):  # pragma: no cover - trivial
        return None


def test_run_skips_instrumentation_by_default(monkeypatch):
    scheduler = Scheduler(
        backends={
            Backend.STATEVECTOR: DummyBackend(),
            Backend.TABLEAU: DummyBackend(),
        }
    )
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    plan = scheduler.prepare_run(circuit)

    perf_called = {"value": False}
    tm_called = {"value": False}

    def fake_perf_counter():
        perf_called["value"] = True
        return 0.0

    def fake_start():
        tm_called["value"] = True

    monkeypatch.setattr("quasar.scheduler.time.perf_counter", fake_perf_counter)
    monkeypatch.setattr("quasar.scheduler.tracemalloc.start", fake_start)
    monkeypatch.setattr(
        "quasar.scheduler.tracemalloc.get_traced_memory", lambda: (0, 0)
    )
    monkeypatch.setattr("quasar.scheduler.tracemalloc.reset_peak", lambda: None)
    monkeypatch.setattr("quasar.scheduler.tracemalloc.stop", lambda: None)

    scheduler.run(circuit, plan)

    assert perf_called["value"] is False
    assert tm_called["value"] is False


def test_run_reports_instrumentation(monkeypatch):
    scheduler = Scheduler(
        backends={
            Backend.STATEVECTOR: DummyBackend(),
            Backend.TABLEAU: DummyBackend(),
        }
    )
    circuit = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "X", "qubits": [0]},
    ])
    plan = scheduler.prepare_run(circuit)

    times = iter([0.0, 1.0, 2.0, 3.0])

    monkeypatch.setattr(
        "quasar.scheduler.time.perf_counter", lambda: next(times)
    )
    monkeypatch.setattr("quasar.scheduler.tracemalloc.start", lambda: None)
    monkeypatch.setattr("quasar.scheduler.tracemalloc.stop", lambda: None)
    monkeypatch.setattr(
        "quasar.scheduler.tracemalloc.get_traced_memory", lambda: (0, 0)
    )
    monkeypatch.setattr("quasar.scheduler.tracemalloc.reset_peak", lambda: None)

    _, run_cost = scheduler.run(circuit, plan, instrument=True)
    assert run_cost.time > 0
