from quasar import Circuit, Scheduler, Planner
from quasar.planner import PlanStep
from quasar_convert import ConversionEngine
from quasar.cost import Backend
from quasar import SSD
import time
from types import SimpleNamespace
from quasar.backends import StimBackend


class CountingConversionEngine(ConversionEngine):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def _bump(self):
        self.calls += 1

    def convert_boundary_to_statevector(self, ssd):  # type: ignore[override]
        self._bump()
        return super().convert_boundary_to_statevector(ssd)

    def convert_boundary_to_tableau(self, ssd):  # type: ignore[override]
        self._bump()
        if hasattr(ConversionEngine, "convert_boundary_to_tableau"):
            return super().convert_boundary_to_tableau(ssd)
        raise AttributeError("convert_boundary_to_tableau not available")

    def convert_boundary_to_dd(self, ssd):  # type: ignore[override]
        self._bump()
        if hasattr(ConversionEngine, "convert_boundary_to_dd"):
            return super().convert_boundary_to_dd(ssd)
        raise AttributeError("convert_boundary_to_dd not available")

    def extract_ssd(self, *args, **kwargs):  # type: ignore[override]
        self._bump()
        return super().extract_ssd(*args, **kwargs)

    def build_bridge_tensor(self, *args, **kwargs):  # type: ignore[override]
        self._bump()
        return super().build_bridge_tensor(*args, **kwargs)

    def extract_local_window(self, *args, **kwargs):  # type: ignore[override]
        self._bump()
        dim = 1 << len(args[1]) if args else 1
        vec = [0j] * dim
        if dim:
            vec[0] = 1.0 + 0j
        return vec


class PrimitiveTrackingEngine(ConversionEngine):
    def __init__(self):
        super().__init__()
        self.local_windows = 0
        self.dense_calls = 0

    def extract_local_window(self, state, qubits):  # type: ignore[override]
        self.local_windows += 1
        dim = 1 << len(qubits)
        win = [0j] * dim
        if dim:
            win[0] = 1.0 + 0j
        return win

    def convert_boundary_to_statevector(self, ssd):  # type: ignore[override]
        self.dense_calls += 1
        return super().convert_boundary_to_statevector(ssd)


class TwoStepPlanner(Planner):
    def plan(self, circuit):  # type: ignore[override]
        steps = [
            PlanStep(0, 5, Backend.TABLEAU),
            PlanStep(5, 6, Backend.MPS),
        ]
        return SimpleNamespace(steps=steps)


class DummyBackend:
    def __init__(self):
        self.backend = Backend.MPS

    def load(self, n):
        self.num_qubits = n

    def ingest(self, state):
        self.state = state

    def apply_gate(self, gate, qubits, params):
        pass

    def extract_ssd(self):
        return SSD([])


def build_switch_circuit():
    return Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "T", "qubits": [0]},
    ])


def build_switch_circuit_rz():
    return Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "RZ", "qubits": [0], "params": {"param0": 0.78539816339}},
    ])


def test_scheduler_triggers_conversion():
    engine = CountingConversionEngine()
    scheduler = Scheduler(conversion_engine=engine)
    circuit = build_switch_circuit()
    scheduler.run(circuit)
    assert engine.calls == 1


def test_scheduler_uses_non_dense_primitive():
    engine = PrimitiveTrackingEngine()
    scheduler = Scheduler(
        conversion_engine=engine,
        planner=TwoStepPlanner(),
        backends={Backend.TABLEAU: StimBackend(), Backend.MPS: DummyBackend()},
    )
    circuit = build_switch_circuit_rz()
    scheduler.run(circuit)
    assert engine.local_windows == 1
    assert engine.dense_calls == 0


def test_scheduler_returns_final_ssd():
    scheduler = Scheduler()
    circuit = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "X", "qubits": [0]},
    ])
    result = scheduler.run(circuit)
    assert isinstance(result, SSD)
    assert result.partitions[0].history == ("H", "X")


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
