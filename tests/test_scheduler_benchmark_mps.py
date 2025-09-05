import numpy as np

from quasar.circuit import Circuit
from quasar.scheduler import Scheduler
from quasar.cost import Backend
from quasar.backends import MPSBackend


def test_scheduler_uses_run_benchmark(monkeypatch):
    calls = {"count": 0}

    def fake_run_benchmark(self):
        calls["count"] += 1
        self._benchmark_state = np.array([1.0, 0.0])
        return self._benchmark_state

    def fake_run(self):  # pragma: no cover - should not be called
        raise AssertionError("_run should not be invoked")

    monkeypatch.setattr(MPSBackend, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(MPSBackend, "_run", fake_run)

    scheduler = Scheduler(backends={Backend.MPS: MPSBackend()})
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    plan = scheduler.prepare_run(circuit, backend=Backend.MPS)
    scheduler.run(circuit, plan, instrument=True)

    assert calls["count"] == 1
