from types import SimpleNamespace

from quasar.circuit import Circuit
from quasar.cost import Backend
from quasar.planner import Planner, PlanStep
from quasar.scheduler import Scheduler
from quasar.backends import StatevectorBackend, MPSBackend
from quasar_convert import ConversionEngine


class CountingEngine(ConversionEngine):
    def __init__(self):
        super().__init__()
        self.bridge_calls = 0
        self.conv_calls = 0

    def build_bridge_tensor(self, left, right):  # type: ignore[override]
        self.bridge_calls += 1
        return super().build_bridge_tensor(left, right)

    def convert_boundary_to_statevector(self, ssd):  # type: ignore[override]
        self.conv_calls += 1
        return super().convert_boundary_to_statevector(ssd)


class BridgePlanner(Planner):
    def plan(self, circuit, *, backend=None, **kwargs):  # type: ignore[override]
        steps = [
            PlanStep(0, 1, Backend.STATEVECTOR),
            PlanStep(1, 2, Backend.MPS),
            PlanStep(2, 3, Backend.STATEVECTOR),
        ]
        return SimpleNamespace(steps=steps)


def test_scheduler_ssd_cache_hits():
    circuit = Circuit(
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "H", "qubits": [1]},
            {"gate": "CX", "qubits": [0, 1]},
        ],
        use_classical_simplification=False,
    )
    engine = CountingEngine()
    scheduler = Scheduler(
        conversion_engine=engine,
        planner=BridgePlanner(),
        backends={
            Backend.STATEVECTOR: StatevectorBackend(),
            Backend.MPS: MPSBackend(),
        },
    )
    plan = scheduler.prepare_run(circuit)
    scheduler.run(circuit, plan, instrument=True)

    plan = scheduler.prepare_run(circuit)
    scheduler.run(circuit, plan, instrument=True)

    assert engine.bridge_calls == 1
    assert engine.conv_calls == 1
    assert scheduler.ssd_cache.hits >= 2
