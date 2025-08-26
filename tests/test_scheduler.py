from quasar import Circuit, Scheduler, Planner
from quasar_convert import ConversionEngine
from quasar.cost import Backend
import time


class CountingConversionEngine(ConversionEngine):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def convert_boundary_to_statevector(self, ssd):  # type: ignore[override]
        self.calls += 1
        return super().convert_boundary_to_statevector(ssd)

    def convert_boundary_to_tableau(self, ssd):  # type: ignore[override]
        self.calls += 1
        return super().convert_boundary_to_tableau(ssd)

    def convert_boundary_to_dd(self, ssd):  # type: ignore[override]
        self.calls += 1
        return super().convert_boundary_to_dd(ssd)


def build_switch_circuit():
    return Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
    ])


def test_scheduler_triggers_conversion():
    engine = CountingConversionEngine()
    scheduler = Scheduler(conversion_engine=engine)
    circuit = build_switch_circuit()
    scheduler.run(circuit)
    assert engine.calls == 1


class CountingPlanner(Planner):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def plan(self, circuit):
        self.calls += 1
        return super().plan(circuit)


def test_scheduler_reoptimises_when_requested():
    planner = CountingPlanner()
    scheduler = Scheduler(planner=planner, conversion_engine=CountingConversionEngine())
    circuit = build_switch_circuit()

    triggered = {"done": False}

    def monitor(step, cost):
        if not triggered["done"]:
            triggered["done"] = True
            return True
        return False

    scheduler.run(circuit, monitor=monitor)
    assert planner.calls >= 2


class SleepBackend:
    def load(self, n):
        pass

    def apply_gate(self, gate, qubits, params):
        time.sleep(0.2)

    def extract_ssd(self):  # pragma: no cover - not used
        return None

    def ingest(self, ssd):  # pragma: no cover - not used
        pass


def test_parallel_execution_on_independent_subcircuits():
    circuit = Circuit([
        {"gate": "T", "qubits": [0]},
        {"gate": "T", "qubits": [1]},
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
    ])
    scheduler = Scheduler(
        backends={
            Backend.STATEVECTOR: SleepBackend(),
            Backend.DECISION_DIAGRAM: SleepBackend(),
        },
        planner=Planner(),
    )
    start = time.time()
    scheduler.run(circuit)
    duration = time.time() - start
    assert duration < 0.7
