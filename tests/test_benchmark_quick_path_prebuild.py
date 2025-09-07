import pytest

from benchmarks.runner import BenchmarkRunner
from quasar.cost import Backend


class DummyGate:
    def __init__(self) -> None:
        self.gate = "X"
        self.qubits = [0]
        self.params = ()


class DummyCircuit:
    num_qubits = 1
    depth = 1
    gates = [DummyGate()]
    ssd = "ssd"


class DummyPlanner:
    def __init__(self) -> None:
        self.called = False

    def plan(self, circuit, *, backend=None) -> None:  # pragma: no cover - trivial
        self.called = True


class DummyBackend:
    name = "dummy"
    loaded = 0
    applied = 0
    extracted = 0

    def load(self, n) -> None:  # pragma: no cover - trivial
        type(self).loaded += 1

    def apply_gate(self, gate, qubits, params) -> None:  # pragma: no cover - trivial
        type(self).applied += 1

    def extract_ssd(self):  # pragma: no cover - trivial
        type(self).extracted += 1
        return "ssd"


class DummyScheduler:
    def __init__(self) -> None:
        self.planner = DummyPlanner()
        self.backends = {Backend.STATEVECTOR: DummyBackend()}
        self.ran = False

    def should_use_quick_path(self, circuit, *, backend=None, force: bool = False):  # pragma: no cover - trivial
        return True

    def select_backend(self, circuit, *, backend=None):  # pragma: no cover - trivial
        return Backend.STATEVECTOR

    def run(self, circuit, *, backend=None):  # pragma: no cover - trivial
        self.ran = True
        return None


def test_quick_path_prebuilds_backend():
    runner = BenchmarkRunner()
    sched = DummyScheduler()
    circuit = DummyCircuit()
    record = runner.run_quasar(circuit, sched)
    assert not sched.planner.called
    assert not sched.ran
    assert DummyBackend.loaded == 1
    assert DummyBackend.applied == 1
    assert DummyBackend.extracted == 1
    assert record["prepare_time"] == pytest.approx(0.0, abs=0.01)
    assert record["run_time"] == pytest.approx(0.0, abs=0.01)
    assert record["backend"] == "STATEVECTOR"
    assert not record["failed"]

