from quasar import Circuit, Scheduler
from quasar.planner import PlanStep, Planner
from quasar_convert import ConversionEngine
from quasar.cost import Backend, CostEstimator
from quasar import SSD
import time
from types import SimpleNamespace
from quasar.backends import StimBackend, StatevectorBackend, MPSBackend
import pytest


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
    def plan(self, circuit, *, backend=None, **kwargs):  # type: ignore[override]
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
    scheduler = Scheduler(
        conversion_engine=engine,
        planner=Planner(quick_max_qubits=None, quick_max_gates=None, quick_max_depth=None),
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
    )
    circuit = build_switch_circuit()
    plan = scheduler.planner.plan(circuit)
    scheduler.run(circuit)
    assert engine.calls > 0


def test_scheduler_uses_non_dense_primitive():
    engine = PrimitiveTrackingEngine()
    scheduler = Scheduler(
        conversion_engine=engine,
        planner=TwoStepPlanner(),
        backends={Backend.TABLEAU: StimBackend(), Backend.MPS: DummyBackend()},
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
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


def test_scheduler_quick_path_skips_planner_and_engine():
    class TableauCountingBackend:
        backend = Backend.TABLEAU
        load_calls = 0
        apply_calls = 0

        def load(self, n):  # pragma: no cover - trivial
            type(self).load_calls += 1

        def apply_gate(self, gate, qubits, params):  # pragma: no cover - trivial
            type(self).apply_calls += 1

        def extract_ssd(self):  # pragma: no cover - trivial
            return SSD([])

    class StatevectorCountingBackend:
        backend = Backend.STATEVECTOR
        load_calls = 0
        apply_calls = 0

        def load(self, n):  # pragma: no cover - trivial
            type(self).load_calls += 1

        def apply_gate(self, gate, qubits, params):  # pragma: no cover - trivial
            type(self).apply_calls += 1

        def extract_ssd(self):  # pragma: no cover - trivial
            return SSD([])

    scheduler = Scheduler(
        backends={
            Backend.TABLEAU: TableauCountingBackend(),
            Backend.STATEVECTOR: StatevectorCountingBackend(),
        }
    )
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    scheduler.run(circuit)
    assert TableauCountingBackend.load_calls == 1
    assert StatevectorCountingBackend.load_calls == 0
    assert scheduler.planner is None
    assert scheduler.conversion_engine is None


class CountingPlanner(Planner):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def plan(self, circuit, *, backend=None, **kwargs):  # type: ignore[override]
        self.calls += 1
        return super().plan(circuit, backend=backend, **kwargs)


@pytest.mark.skip(reason="takes too long for CI")
def test_scheduler_reoptimises_when_requested():
    planner = CountingPlanner()
    scheduler = Scheduler(
        planner=planner,
        conversion_engine=CountingConversionEngine(),
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
    )
    circuit = build_switch_circuit()

    triggered = {"done": False}

    def monitor(step, observed, estimated):
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
        time.sleep(0.05)

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
    scheduler_p = Scheduler(
        backends={Backend.DECISION_DIAGRAM: SleepBackend()},
        planner=Planner(),
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
        parallel_backends=[Backend.DECISION_DIAGRAM],
    )
    start = time.time()
    scheduler_p.run(circuit)
    parallel_duration = time.time() - start

    scheduler_s = Scheduler(
        backends={Backend.DECISION_DIAGRAM: SleepBackend()},
        planner=Planner(),
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
        parallel_backends=[],
    )
    start = time.time()
    scheduler_s.run(circuit)
    serial_duration = time.time() - start

    assert serial_duration > parallel_duration
    assert serial_duration - parallel_duration > 0.05


class BridgeTrackingEngine(ConversionEngine):
    def __init__(self):
        super().__init__()
        self.bridge_calls = 0

    def build_bridge_tensor(self, *args, **kwargs):  # type: ignore[override]
        self.bridge_calls += 1
        return super().build_bridge_tensor(*args, **kwargs)


class BridgePlanner(Planner):
    def plan(self, circuit, *, backend=None, **kwargs):  # type: ignore[override]
        steps = [
            PlanStep(0, 1, Backend.STATEVECTOR),
            PlanStep(1, 2, Backend.MPS),
            PlanStep(2, 3, Backend.STATEVECTOR),
        ]
        return SimpleNamespace(steps=steps)


def test_cross_backend_gate_uses_bridge_tensor():
    circuit = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "CX", "qubits": [0, 1]},
    ])
    engine = BridgeTrackingEngine()
    scheduler = Scheduler(
        conversion_engine=engine,
        planner=BridgePlanner(),
        backends={Backend.STATEVECTOR: StatevectorBackend(), Backend.MPS: DummyBackend()},
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
    )
    result = scheduler.run(circuit)
    assert engine.bridge_calls == 1
    assert result.conversions[-1].primitive == "BRIDGE"


def test_partition_histories_multiple_backends():
    circuit = Circuit([
        # Two identical Clifford subcircuits -> tableau backend
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "S", "qubits": [1]},
        {"gate": "H", "qubits": [2]},
        {"gate": "CX", "qubits": [2, 3]},
        {"gate": "S", "qubits": [3]},
        # Local non-Clifford chain -> MPS backend
        {"gate": "T", "qubits": [4]},
        {"gate": "CX", "qubits": [4, 5]},
        {"gate": "T", "qubits": [5]},
        {"gate": "CX", "qubits": [5, 6]},
        {"gate": "T", "qubits": [6]},
        # Sparse non-local gates -> decision diagram backend
        {"gate": "T", "qubits": [7]},
        {"gate": "CX", "qubits": [7, 9]},
        {"gate": "T", "qubits": [9]},
    ])
    groups = circuit.ssd.by_backend()
    assert set(groups.keys()) == {
        Backend.TABLEAU,
        Backend.MPS,
        Backend.DECISION_DIAGRAM,
    }
    tableau = groups[Backend.TABLEAU][0]
    assert tableau.multiplicity == 2
    assert set(tableau.qubits) == {0, 1, 2, 3}
    assert tableau.history == ("H", "CX", "S")
    mps_histories = [p.history for p in groups[Backend.MPS]]
    assert ("T", "CX", "T", "CX", "T") in mps_histories
    dd = groups[Backend.DECISION_DIAGRAM][0]
    assert dd.history == ("CX", "T")


class B2BFallbackEngine(ConversionEngine):
    def __init__(self):
        super().__init__()
        self.dense_calls = 0

    def convert(self, ssd):  # type: ignore[override]
        return SimpleNamespace(primitive=SimpleNamespace(name="B2B"), cost=0.0)

    def convert_boundary_to_statevector(self, ssd):  # type: ignore[override]
        self.dense_calls += 1
        return super().convert_boundary_to_statevector(ssd)


class FailingOnceBackend(DummyBackend):
    calls = 0

    def ingest(self, state):  # type: ignore[override]
        type(self).calls += 1
        if type(self).calls == 1:
            raise RuntimeError("fail")
        self.state = state


class SVThenMPSPlanner(Planner):
    def plan(self, circuit, *, backend=None, **kwargs):  # type: ignore[override]
        steps = [
            PlanStep(0, 1, Backend.STATEVECTOR),
            PlanStep(1, 2, Backend.MPS),
        ]
        return SimpleNamespace(steps=steps)


def test_conversion_fallback_path():
    engine = B2BFallbackEngine()
    scheduler = Scheduler(
        conversion_engine=engine,
        planner=SVThenMPSPlanner(),
        backends={Backend.STATEVECTOR: StatevectorBackend(), Backend.MPS: FailingOnceBackend()},
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
    )
    circuit = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [0]},
    ])
    scheduler.run(circuit)
    assert engine.dense_calls == 1
    assert FailingOnceBackend.calls == 2


class AutoTwoStepPlanner(Planner):
    def __init__(self, estimator):
        super().__init__(estimator=estimator)
        self.calls = 0

    def plan(self, circuit, *, backend=None, **kwargs):  # type: ignore[override]
        self.calls += 1
        if not circuit.gates:
            steps = []
        elif len(circuit.gates) > 1:
            steps = [
                PlanStep(0, 1, Backend.STATEVECTOR),
                PlanStep(1, len(circuit.gates), Backend.STATEVECTOR),
            ]
        else:
            steps = [PlanStep(0, 1, Backend.STATEVECTOR)]
        return SimpleNamespace(steps=steps)


def test_scheduler_auto_reoptimises_on_cost_mismatch():
    coeff = {k: 0.0 for k in CostEstimator().coeff}
    est = CostEstimator(coeff)
    planner = AutoTwoStepPlanner(est)
    scheduler = Scheduler(
        planner=planner,
        conversion_engine=CountingConversionEngine(),
        backends={Backend.STATEVECTOR: SleepBackend()},
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
    )
    circuit = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [0]},
    ])
    scheduler.run(circuit)
    assert planner.calls >= 2
