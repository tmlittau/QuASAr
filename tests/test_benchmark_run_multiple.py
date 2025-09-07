import math
import time
from unittest.mock import patch

import pytest

from benchmarks.circuits import ghz_circuit
from benchmarks.runner import BenchmarkRunner
from quasar import SimulationEngine
from quasar.ssd import SSD, SSDPartition
from quasar.cost import Backend, Cost
from quasar.planner import PlanResult, PlanStep
from types import SimpleNamespace


class DummyBackend:
    name = "dummy"

    def run(self, circuit, **_):
        return None


def test_run_multiple_aggregates_statistics():
    runner = BenchmarkRunner()
    # perf_counter is called once for start_time and twice per run
    side_effect = [
        0.0,  # start of run_multiple
        # run 1
        0.0, 1.0,
        # run 2
        1.0, 3.0,
        # run 3
        3.0, 6.0,
    ]
    with patch("benchmarks.runner.time.perf_counter", side_effect=side_effect):
        record = runner.run_multiple(None, DummyBackend(), repetitions=3)
    assert len(runner.results) == 1
    assert runner.results[0] == record
    assert record["repetitions"] == 3
    assert record["run_time_mean"] == 2.0
    assert math.isclose(record["run_time_std"], math.sqrt(2 / 3))
    assert record["backend"] == "dummy"


class SleepBackend:
    name = "sleep"

    def __init__(self, durations):
        self.durations = durations

    def run(self, circuit, **_):
        t = self.durations.pop(0)
        time.sleep(t)
        return None


def test_run_multiple_timeout_records_failure():
    runner = BenchmarkRunner()
    backend = SleepBackend([0.01, 0.2])
    record = runner.run_multiple(
        None, backend, repetitions=2, run_timeout=0.05
    )
    assert record["repetitions"] == 1
    assert "failed_runs" in record and len(record["failed_runs"]) == 1
    assert "timed out" in record["failed_runs"][0]
    assert "comment" in record and "excluded" in record["comment"]
    assert record["backend"] == "sleep"


class FlakyBackend:
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def run(self, circuit, **_):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return None


def test_run_multiple_skips_failed_runs():
    runner = BenchmarkRunner()
    backend = FlakyBackend()
    side_effect = [
        0.0,  # start of run_multiple
        # run 1 (fails)
        0.0,
        # run 2
        1.0, 2.0,
        # run 3
        2.0, 5.0,
    ]
    with patch("benchmarks.runner.time.perf_counter", side_effect=side_effect):
        record = runner.run_multiple(None, backend, repetitions=3)
    assert record["repetitions"] == 2
    assert "failed_runs" in record and len(record["failed_runs"]) == 1
    assert "failed" in record["failed_runs"][0]
    assert "comment" in record and "excluded" in record["comment"]
    assert record["run_time_mean"] == 2.0
    assert record["run_time_std"] == 1.0
    assert record["backend"] == "flaky"


class UnsupportedBackend:
    name = "unsupported"

    def run(self, circuit, **_):
        raise NotImplementedError("nyi")


def test_run_multiple_records_unsupported():
    runner = BenchmarkRunner()
    record = runner.run_multiple(None, UnsupportedBackend(), repetitions=3)
    assert record["unsupported"] is True
    assert record["repetitions"] == 0
    assert "nyi" in record["comment"]
    assert record["framework"] == "unsupported"
    assert record["backend"] == "unsupported"


class DummyScheduler:
    def __init__(self):
        self.plan_calls: list = []
        self.run_calls: list = []
        self.run_times = [0.0, 1.0, 2.0, 3.0]

        class Planner:
            def __init__(self, outer):
                self.outer = outer

            def plan(self, circuit, *, backend=None):
                self.outer.plan_calls.append(backend)
                return PlanResult(
                    table=[],
                    final_backend=backend,
                    gates=[],
                    explicit_steps=[],
                    explicit_conversions=[],
                    step_costs=[],
                )

        self.planner = Planner(self)

    def prepare_run(self, circuit, plan=None, *, backend=None):
        return (
            plan
            if plan is not None
            else PlanResult(
                table=[],
                final_backend=backend,
                gates=[],
                explicit_steps=[],
                explicit_conversions=[],
                step_costs=[],
            )
        )

    def run(self, circuit, plan, *, monitor=None, instrument=False):
        self.run_calls.append((plan.final_backend, instrument))
        runtime = self.run_times.pop(0)
        ssd = SSD(
            [
                SSDPartition(
                    subsystems=((0,),),
                    backend=plan.final_backend or Backend.STATEVECTOR,
                )
            ]
        )
        if instrument:
            return ssd, Cost(time=runtime, memory=0.0)
        return ssd


def test_run_quasar_multiple_aggregates_statistics():
    runner = BenchmarkRunner()
    scheduler = DummyScheduler()
    side_effect = [
        0.0,  # start of run_quasar_multiple
        0.0, 1.0,  # run 1
        1.0, 3.0,  # run 2
        3.0, 6.0,  # run 3
    ]
    with patch("benchmarks.runner.time.perf_counter", side_effect=side_effect):
        record = runner.run_quasar_multiple(
            None, scheduler, repetitions=3, backend=Backend.TABLEAU
        )
    assert len(runner.results) == 1
    assert runner.results[0] == record
    assert record["repetitions"] == 3
    assert record["run_time_mean"] == 2.0
    assert math.isclose(record["run_time_std"], math.sqrt(2 / 3))
    assert scheduler.plan_calls == [Backend.TABLEAU]
    assert scheduler.run_calls == [(Backend.TABLEAU, False)] * 3
    assert record["backend"] == Backend.TABLEAU.name


class DirectScheduler:
    def __init__(self):
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
        self.run_calls: list[bool] = []
        self.backends = {Backend.STATEVECTOR: DummySimBackend()}

    def prepare_run(self, circuit, plan=None, *, backend=None):
        return plan if plan is not None else self.planner.plan(circuit, backend=backend)

    def run(self, circuit, plan, *, monitor=None, instrument=False):
        self.run_calls.append(instrument)
        ssd = SSD([
            SSDPartition(subsystems=((0,),), backend=plan.final_backend or Backend.STATEVECTOR)
        ])
        if instrument:
            return ssd, Cost(time=0.0, memory=0.0)
        return ssd


class DummySimBackend:
    backend = Backend.STATEVECTOR

    def load(self, num_qubits):
        pass

    def apply_gate(self, gate, qubits, params):
        pass

    def extract_ssd(self):
        return SSD([
            SSDPartition(subsystems=((0,),), backend=Backend.STATEVECTOR)
        ])


class DummyCircuit:
    def __init__(self):
        self.num_qubits = 1
        self.gates = [SimpleNamespace(gate="x", qubits=(0,), params=())]
        self.ssd = SSD([])

    def simplify_classical_controls(self):
        return self.gates


def test_run_quasar_multiple_direct_backend_path():
    runner = BenchmarkRunner()
    circuit = DummyCircuit()
    scheduler = DirectScheduler()
    side_effect = [0.0, 0.0, 1.0, 1.0, 3.0]
    with patch("benchmarks.runner.time.perf_counter", side_effect=side_effect):
        record = runner.run_quasar_multiple(circuit, scheduler, repetitions=2)
    assert record["repetitions"] == 2
    assert math.isclose(record["run_time_mean"], 1.5)
    assert math.isclose(record["run_time_std"], math.sqrt(0.25))
    assert scheduler.run_calls == []
    assert record["backend"] == Backend.STATEVECTOR.name


class EstimatorScheduler(DirectScheduler):
    def __init__(self):
        super().__init__()

        class Estimator:
            def __init__(self):
                self.coeff = {"changed": False}

        self.planner.estimator = Estimator()

    def run(self, circuit, plan, *, monitor=None, instrument=False):
        self.run_calls.append(instrument)
        if instrument:
            self.planner.estimator.coeff["changed"] = True
            ssd = SSD([
                SSDPartition(
                    subsystems=((0,),),
                    backend=plan.final_backend or Backend.STATEVECTOR,
                )
            ])
            return ssd, Cost(time=0.0, memory=0.0)
        return SSD([
            SSDPartition(
                subsystems=((0,),),
                backend=plan.final_backend or Backend.STATEVECTOR,
            )
        ])


def test_run_quasar_multiple_skips_instrumentation_keeps_estimator_coeff():
    runner = BenchmarkRunner()
    circuit = DummyCircuit()
    scheduler = EstimatorScheduler()
    record = runner.run_quasar_multiple(circuit, scheduler, repetitions=1)
    assert record["repetitions"] == 1
    assert scheduler.run_calls == []
    assert scheduler.planner.estimator.coeff["changed"] is False


class PlannerErrorScheduler:
    def __init__(self):
        class Planner:
            def plan(self, circuit, *, backend=None):
                raise RuntimeError("plan boom")

        self.planner = Planner()

    def prepare_run(self, circuit, plan=None, *, backend=None):  # pragma: no cover - unused
        return plan

    def run(self, circuit, plan, *, monitor=None, instrument=False):  # pragma: no cover - unused
        ssd = SSD([
            SSDPartition(subsystems=((0,),), backend=plan.final_backend or Backend.STATEVECTOR)
        ])
        return ssd, Cost(time=0.0, memory=0.0)


def test_run_quasar_returns_failure_record_on_planner_error():
    runner = BenchmarkRunner()
    scheduler = PlannerErrorScheduler()
    record = runner.run_quasar(None, scheduler)
    assert record["failed"] is True
    assert "plan boom" in record["error"]
    assert record["backend"] is None


class RunErrorScheduler:
    def __init__(self):
        class Planner:
            def plan(self, circuit, *, backend=None):
                return PlanResult(table=[], final_backend=backend, gates=[], explicit_steps=[], explicit_conversions=[], step_costs=[])

        self.planner = Planner()

    def prepare_run(self, circuit, plan=None, *, backend=None):
        return plan

    def run(self, circuit, plan, *, monitor=None, instrument=False):
        if not instrument:
            raise ValueError("run boom")
        return SSD([
            SSDPartition(subsystems=((0,),), backend=plan.final_backend or Backend.STATEVECTOR)
        ]), Cost(time=0.0, memory=0.0)


class RunFailScheduler:
    def __init__(self):
        class Planner:
            def plan(self, circuit, *, backend=None):
                return PlanResult(table=[], final_backend=backend, gates=[], explicit_steps=[], explicit_conversions=[], step_costs=[])

        self.planner = Planner()

    def prepare_run(self, circuit, plan=None, *, backend=None):
        return plan

    def run(self, circuit, plan, *, monitor=None, instrument=False):
        raise ValueError("run boom")


def test_run_quasar_returns_failure_record_on_run_error():
    runner = BenchmarkRunner()
    scheduler = RunFailScheduler()
    record = runner.run_quasar(None, scheduler)
    assert record["failed"] is True
    assert "run boom" in record["error"]
    assert record["backend"] is None


def test_run_quasar_multiple_raises_runtime_error_with_failures():
    runner = BenchmarkRunner()
    scheduler = RunErrorScheduler()
    with pytest.raises(RuntimeError) as exc:
        runner.run_quasar_multiple(None, scheduler, repetitions=2)
    assert "run boom" in str(exc.value)


