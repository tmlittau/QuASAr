"""Scheduler parallel execution tests."""

from __future__ import annotations

import time

import numpy as np

from benchmarks.parallel_circuits import many_ghz_subsystems
from quasar.circuit import Circuit
from quasar.cost import Backend, Cost
from quasar.planner import PlanResult, PlanStep, Planner
from quasar.scheduler import Scheduler, _clone_backend_instance, merge_subsystems
from quasar.ssd import ConversionLayer


def test_scheduler_parallelizes_independent_ghz_blocks(monkeypatch) -> None:
    """Planner and scheduler should expose and exploit parallel GHZ blocks."""

    num_blocks = 3
    block_size = 4
    circuit = many_ghz_subsystems(num_groups=num_blocks, group_size=block_size)

    planner = Planner()
    plan = planner.plan(circuit, backend=Backend.STATEVECTOR)

    # The planner should emit a single step whose parallel metadata mirrors the
    # GHZ blocks from the circuit construction.
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert len(step.parallel) == num_blocks
    expected_groups = tuple(
        tuple(range(block * block_size, (block + 1) * block_size))
        for block in range(num_blocks)
    )
    assert step.parallel == expected_groups

    # Provide dummy metadata so ``Scheduler.run`` executes the generic path that
    # honours ``step.parallel`` rather than the quick single-step shortcut.
    plan.step_costs = [Cost(time=0.0, memory=0.0)]
    plan.explicit_conversions = [
        ConversionLayer(
            boundary=(),
            source=step.backend,
            target=step.backend,
            rank=1,
            frontier=0,
            primitive="Full",
            cost=Cost(time=0.0, memory=0.0),
        )
    ]

    instances: list["RecordingBackend"] = []

    class RecordingBackend:
        backend = Backend.STATEVECTOR

        def __init__(self) -> None:
            instances.append(self)
            self.applied: list[tuple[str, tuple[int, ...]]] = []
            self.num_qubits: int | None = None

        def load(self, num_qubits: int) -> None:
            self.num_qubits = num_qubits

        def apply_gate(self, name: str, qubits: list[int], params: dict | None) -> None:
            self.applied.append((name, tuple(qubits)))

        def extract_ssd(self):  # pragma: no cover - not exercised in this test
            return None

    executor_records: list[dict[str, list[list[tuple[object, tuple[int, ...]]]]]] = []

    class RecordingExecutor:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            record = {"summaries": []}
            executor_records.append(record)
            self._record = record
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def map(self, func, iterable):
            jobs = list(iterable)
            summary: list[tuple[object, tuple[int, ...]]] = []
            for job in jobs:
                sim, gate_list = job
                qubits = tuple(sorted({q for gate in gate_list for q in gate.qubits}))
                summary.append((sim, qubits))
            self._record["summaries"].append(summary)
            for job in jobs:
                func(job)
            return []

    monkeypatch.setattr("quasar.scheduler.ThreadPoolExecutor", RecordingExecutor)

    scheduler = Scheduler(
        planner=planner,
        backends={Backend.STATEVECTOR: RecordingBackend()},
        parallel_backends=[Backend.STATEVECTOR],
    )

    scheduler.run(circuit, plan=plan)

    # ThreadPoolExecutor should be used exactly once with one job per block.
    assert executor_records, "ThreadPoolExecutor was not invoked"
    assert len(executor_records[0]["summaries"]) == 1
    summary = executor_records[0]["summaries"][0]
    assert len(summary) == num_blocks

    sims_seen = {entry[0] for entry in summary}
    assert len(instances) == num_blocks + 1  # base instance + one per block
    assert instances[0] not in sims_seen
    assert sims_seen == set(instances[1:])

    observed_groups = sorted(entry[1] for entry in summary)
    assert observed_groups == sorted(expected_groups)


def test_parallel_single_qubit_merge_creates_bridge_state() -> None:
    """Merging two single-qubit subsystems should yield the expected tensor."""

    scheduler = Scheduler()
    template = scheduler.backends[Backend.STATEVECTOR]

    left = _clone_backend_instance(template)
    left.load(1)
    left.apply_gate("H", [0], None)

    right = _clone_backend_instance(template)
    right.load(1)
    right.apply_gate("X", [0], None)

    gate = Circuit([
        {"gate": "CX", "qubits": [0, 1]},
    ], use_classical_simplification=False).gates[0]

    merged, merged_qubits = merge_subsystems(
        left,
        right,
        gate,
        left_qubits=(0,),
        right_qubits=(1,),
    )

    assert merged_qubits == (0, 1)
    bridge_state = np.asarray(merged.statevector(), dtype=complex)
    expected = np.zeros(4, dtype=complex)
    expected[2] = expected[3] = 1 / np.sqrt(2)
    assert np.allclose(bridge_state, expected)


def test_disjoint_plan_steps_run_concurrently(monkeypatch) -> None:
    """Disjoint plan steps without conversions should execute in parallel."""

    delay = 0.05

    class SleepingStatevectorBackend:
        backend = Backend.STATEVECTOR

        def load(self, num_qubits: int) -> None:  # pragma: no cover - trivial
            _ = num_qubits

        def apply_gate(self, name, qubits, params) -> None:  # pragma: no cover
            _ = (name, qubits, params)
            time.sleep(delay)

        def extract_ssd(self):  # pragma: no cover - not used
            return None

    class SleepingMPSBackend(SleepingStatevectorBackend):
        backend = Backend.MPS

    def make_plan() -> tuple[Circuit, PlanResult]:
        gates = [
            {"gate": "H", "qubits": [0]},
            {"gate": "X", "qubits": [0]},
            {"gate": "Z", "qubits": [0]},
            {"gate": "H", "qubits": [1]},
            {"gate": "X", "qubits": [1]},
            {"gate": "Z", "qubits": [1]},
        ]
        circuit = Circuit(gates, use_classical_simplification=False)
        plan = PlanResult(
            table=[],
            final_backend=None,
            gates=list(circuit.gates),
            explicit_steps=[
                PlanStep(0, 3, Backend.STATEVECTOR),
                PlanStep(3, 6, Backend.MPS),
            ],
            explicit_conversions=[],
        )
        plan.step_costs = [
            Cost(time=delay * 3, memory=0.0),
            Cost(time=delay * 3, memory=0.0),
        ]
        return circuit, plan

    def run_scheduler(scheduler: Scheduler) -> float:
        circuit, plan = make_plan()
        start = time.perf_counter()
        scheduler.run(circuit, plan=plan)
        return time.perf_counter() - start

    backends = {
        Backend.STATEVECTOR: SleepingStatevectorBackend(),
        Backend.MPS: SleepingMPSBackend(),
    }

    scheduler_parallel = Scheduler(backends=backends)
    parallel_time = run_scheduler(scheduler_parallel)

    scheduler_sequential = Scheduler(backends=backends)

    def no_parallel(self, steps, qubits, start):
        return [start]

    monkeypatch.setattr(
        scheduler_sequential,
        "_collect_disjoint_steps",
        no_parallel.__get__(scheduler_sequential, Scheduler),
    )
    sequential_time = run_scheduler(scheduler_sequential)

    assert parallel_time < sequential_time
