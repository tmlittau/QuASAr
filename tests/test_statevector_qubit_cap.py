import pytest
from benchmarks.runner import BenchmarkRunner
from quasar.cost import Backend
from quasar.planner import PlanResult, PlanStep
from quasar.ssd import SSD
from types import SimpleNamespace


class DummySimBackend:
    backend = Backend.STATEVECTOR

    def load(self, num_qubits):
        pass

    def apply_gate(self, gate, qubits, params):
        pass

    def extract_ssd(self):
        return SSD([])


class SimpleScheduler:
    def __init__(self):
        self.backends = {Backend.STATEVECTOR: DummySimBackend()}

        class Planner:
            def plan(self, circuit, *, backend=None):
                return PlanResult(
                    table=[],
                    final_backend=backend or Backend.STATEVECTOR,
                    gates=circuit.gates,
                    explicit_steps=[
                        PlanStep(0, len(circuit.gates), backend or Backend.STATEVECTOR)
                    ],
                    explicit_conversions=[],
                    step_costs=[],
                )

        self.planner = Planner()

    def prepare_run(self, circuit, plan=None, *, backend=None):
        return plan if plan is not None else self.planner.plan(circuit, backend=backend)

    def run(self, circuit, plan, *, monitor=None, instrument=False):
        if instrument:
            from quasar.cost import Cost

            return SSD([]), Cost(time=0.0, memory=0.0)
        return SSD([])


class WideCircuit:
    def __init__(self, width):
        self.num_qubits = width
        self.gates = [SimpleNamespace(gate="x", qubits=(0,), params=())]
        self.ssd = SSD([])

    def simplify_classical_controls(self):
        return self.gates


def test_run_quasar_multiple_skips_when_over_cap():
    runner = BenchmarkRunner()
    circuit = WideCircuit(4)
    scheduler = SimpleScheduler()
    with pytest.warns(UserWarning, match="exceeds statevector limit"):
        record = runner.run_quasar_multiple(
            circuit,
            scheduler,
            repetitions=1,
            memory_bytes=16 * (2**3),
        )
    assert record["unsupported"] is True
    assert record["repetitions"] == 0
    assert record["backend"] == Backend.STATEVECTOR.name


def test_run_quasar_quick_skips_when_over_cap():
    runner = BenchmarkRunner()
    circuit = WideCircuit(4)

    class QuickScheduler:
        def __init__(self):
            self.backends = {Backend.STATEVECTOR: DummySimBackend()}

        def should_use_quick_path(self, circuit, backend=None, force=False):
            return True

        def select_backend(self, circuit, backend=None):
            return backend or Backend.STATEVECTOR

    scheduler = QuickScheduler()
    with pytest.warns(UserWarning, match="exceeds statevector limit"):
        record = runner.run_quasar(
            circuit,
            scheduler,
            backend=Backend.STATEVECTOR,
            quick=True,
            memory_bytes=16 * (2**3),
        )
    assert record["unsupported"] is True
    assert record["backend"] == Backend.STATEVECTOR.name
    assert record["failed"] is False
