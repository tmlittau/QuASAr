from __future__ import annotations

"""Regression tests for :mod:`benchmarks.bench_utils.runner`."""

import math

from benchmarks.bench_utils.runner import BenchmarkRunner
from quasar.cost import Backend


class _DummyConversion:
    def __init__(self) -> None:
        self.boundary = (0,)
        self.rank = 1
        self.frontier = 1
        self.primitive = "dummy"


class _DummyPartition:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.multiplicity = 1
        self.subsystems = (0,)


class _DummyPlan:
    def __init__(self, backend: Backend) -> None:
        self.conversions = [_DummyConversion()]
        self.partitions = [_DummyPartition(backend)]
        self.steps = []
        self.gates = []


class _DummyResult:
    def __init__(self, backend: Backend) -> None:
        self.partitions = [_DummyPartition(backend)]
        self.hierarchy = None


class _DummyCost:
    def __init__(self, *, time: float, memory: float) -> None:
        self.time = time
        self.memory = memory


class _DummyPlanner:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    def plan(self, circuit, backend=None, max_memory=None):  # noqa: ANN001
        return _DummyPlan(self.backend)


class _DummyScheduler:
    def __init__(self, backend: Backend, *, time: float, memory: float) -> None:
        self._backend = backend
        self._cost = _DummyCost(time=time, memory=memory)
        self.backends = {backend: object()}
        self.planner = _DummyPlanner(backend)

    def should_use_quick_path(self, *_, **__):  # noqa: D401, ANN001
        """Always request the quick path."""

        return True

    def select_backend(self, *_, **__):  # noqa: ANN001
        return None

    def prepare_run(self, circuit, plan=None, **__):  # noqa: ANN001
        return plan if plan is not None else _DummyPlan(self._backend)

    def run(self, circuit, plan, instrument=True):  # noqa: ANN001
        result = _DummyResult(self._backend)
        if instrument:
            return result, self._cost
        return result


class _DummyEngine:
    def __init__(self, backend: Backend, *, time: float, memory: float) -> None:
        self.scheduler = _DummyScheduler(backend, time=time, memory=memory)
        self.planner = self.scheduler.planner


class _DummyCircuit:
    def __init__(self, qubits: int) -> None:
        self.num_qubits = qubits
        self.gates = []
        self.ssd = None


def test_run_quasar_falls_back_when_quick_backend_unavailable() -> None:
    circuit = _DummyCircuit(qubits=3)
    engine = _DummyEngine(Backend.MPS, time=1.25, memory=512)
    runner = BenchmarkRunner()

    record = runner.run_quasar(circuit, engine, quick=True)

    assert record["failed"] is False
    assert record["backend"] == Backend.MPS.name
    assert math.isclose(record["run_time"], 1.25)
    assert record["run_peak_memory"] == 512


def test_run_quasar_multiple_records_metrics_on_quick_fallback() -> None:
    circuit = _DummyCircuit(qubits=3)
    engine = _DummyEngine(Backend.MPS, time=2.0, memory=256)
    runner = BenchmarkRunner()

    summary = runner.run_quasar_multiple(circuit, engine, repetitions=2, quick=True)

    assert summary["repetitions"] == 2
    assert summary.get("failed_runs") is None
    assert summary["backend"] == Backend.MPS.name
    assert summary["run_time_mean"] > 0.0
    assert summary["run_peak_memory_mean"] is not None

