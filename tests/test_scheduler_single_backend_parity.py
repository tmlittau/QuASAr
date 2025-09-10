from quasar.circuit import Circuit
from quasar.scheduler import Scheduler


def test_single_backend_auto_and_fixed_runtime_parity(monkeypatch):
    scheduler = Scheduler()
    auto_circ = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "X", "qubits": [0]},
    ])
    fixed_circ = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "X", "qubits": [0]},
    ])

    backend = scheduler.select_backend(auto_circ)

    times = iter(range(100))
    monkeypatch.setattr("quasar.scheduler.time.perf_counter", lambda: next(times))
    monkeypatch.setattr("quasar.scheduler.tracemalloc.start", lambda: None)
    monkeypatch.setattr("quasar.scheduler.tracemalloc.stop", lambda: None)
    monkeypatch.setattr("quasar.scheduler.tracemalloc.reset_peak", lambda: None)
    monkeypatch.setattr(
        "quasar.scheduler.tracemalloc.get_traced_memory", lambda: (0, 0)
    )

    _, auto_metrics = scheduler.run(auto_circ, instrument=True)
    _, fixed_metrics = scheduler.run(fixed_circ, instrument=True, backend=backend)

    assert auto_metrics.cost.time == fixed_metrics.cost.time
