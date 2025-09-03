import math
import time
from unittest.mock import patch

import pytest

from benchmarks.backends import StatevectorAdapter
from benchmarks.circuits import ghz_circuit
from benchmarks.runner import BenchmarkRunner
from quasar import SimulationEngine
from quasar.ssd import SSD, SSDPartition
from quasar.cost import Backend


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
        self.plan_calls = []
        self.run_calls = []

        class Planner:
            def __init__(self, outer):
                self.outer = outer

            def plan(self, circuit, *, backend=None):
                self.outer.plan_calls.append(backend)

        self.planner = Planner(self)

    def run(self, circuit, *, backend=None):
        self.run_calls.append(backend)
        return SSD([
            SSDPartition(subsystems=((0,),), backend=backend or Backend.STATEVECTOR)
        ])


def test_run_quasar_multiple_aggregates_statistics():
    runner = BenchmarkRunner()
    scheduler = DummyScheduler()
    side_effect = [
        0.0,  # start of run_quasar_multiple
        # run 1
        0.0, 0.0, 0.0, 1.0,
        # run 2
        1.0, 1.0, 1.0, 3.0,
        # run 3
        3.0, 3.0, 3.0, 6.0,
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
    assert scheduler.plan_calls == [Backend.TABLEAU] * 3
    assert scheduler.run_calls == [Backend.TABLEAU] * 3
    assert record["backend"] == Backend.TABLEAU.name


class PlannerErrorScheduler:
    def __init__(self):
        class Planner:
            def plan(self, circuit, *, backend=None):
                raise RuntimeError("plan boom")

        self.planner = Planner()

    def run(self, circuit, *, backend=None):
        return SSD([
            SSDPartition(subsystems=((0,),), backend=backend or Backend.STATEVECTOR)
        ])


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
                pass

        self.planner = Planner()

    def run(self, circuit, *, backend=None):
        raise ValueError("run boom")


def test_run_quasar_returns_failure_record_on_run_error():
    runner = BenchmarkRunner()
    scheduler = RunErrorScheduler()
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


def test_statevector_and_quasar_runtime_agree():
    circuit = ghz_circuit(3)
    runner = BenchmarkRunner()

    direct = runner.run_multiple(
        circuit, StatevectorAdapter(), repetitions=3, statevector=False
    )
    quasar = runner.run_quasar_multiple(
        circuit, SimulationEngine(), backend=Backend.STATEVECTOR, repetitions=3
    )

    assert abs(direct["run_time_mean"] - quasar["run_time_mean"]) < 0.01


def test_statevector_adapter_returns_ssd():
    circuit = ghz_circuit(2)
    runner = BenchmarkRunner()
    record = runner.run(circuit, StatevectorAdapter(), statevector=False)
    assert isinstance(record["result"], SSD)
