"""
Exhaustively enumerate partition/backend combinations for small circuits
and compare against QuASAr's planner.

This script generates small random circuits with at most eight qubits,
computes the oracle optimal plan by enumerating all possible contiguous
partitions and backend assignments and compares the result to the plan
chosen by the dynamic programming planner.

The output table reports the plan quality gap and CPU overhead of the
enumeration approach relative to the planner.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from quasar import Circuit, Planner
from quasar.cost import Cost, CostEstimator, Backend
from quasar.planner import (
    PlanStep,
    _supported_backends,
    _simulation_cost,
    _add_cost,
    _better,
)
from quasar.circuit import Gate


def _boundaries(gates: List[Gate]) -> List[set[int]]:
    """Compute boundary qubit sets for each cut position."""

    n = len(gates)
    prefix: List[set[int]] = [set() for _ in range(n + 1)]
    running: set[int] = set()
    for i, gate in enumerate(gates, start=1):
        running |= set(gate.qubits)
        prefix[i] = running.copy()

    future: List[set[int]] = [set() for _ in range(n + 1)]
    running.clear()
    for idx in range(n - 1, -1, -1):
        running |= set(gates[idx].qubits)
        future[idx] = running.copy()

    return [prefix[i] & future[i] for i in range(n + 1)]


def enumerate_oracle(
    gates: List[Gate], estimator: CostEstimator
) -> Tuple[Cost, List[PlanStep]]:
    """Enumerate all partition/backend combinations to find the optimal plan."""

    bounds = _boundaries(gates)
    n = len(gates)

    from functools import lru_cache

    @lru_cache(None)
    def backtrack(
        start: int, prev_backend: Optional[Backend]
    ) -> Tuple[Cost, Tuple[PlanStep, ...]]:
        if start == n:
            return Cost(0.0, 0.0), ()
        best_cost = Cost(float("inf"), float("inf"))
        best_plan: Tuple[PlanStep, ...] = ()
        for end in range(start + 1, n + 1):
            segment = gates[start:end]
            qubits = {q for g in segment for q in g.qubits}
            num_qubits = len(qubits)
            num_gates = end - start
            for backend in _supported_backends(segment):
                sim_cost = _simulation_cost(
                    estimator, backend, num_qubits, num_gates
                )
                conv_cost = Cost(0.0, 0.0)
                if prev_backend is not None and prev_backend != backend:
                    boundary = bounds[start]
                    if boundary:
                        rank = 2 ** len(boundary)
                        frontier = len(boundary)
                        conv_cost = estimator.conversion(
                            prev_backend,
                            backend,
                            num_qubits=len(boundary),
                            rank=rank,
                            frontier=frontier,
                        ).cost
                rem_cost, rem_plan = backtrack(end, backend)
                total = _add_cost(_add_cost(sim_cost, conv_cost), rem_cost)
                if _better(total, best_cost):
                    best_cost = total
                    best_plan = (PlanStep(start, end, backend),) + rem_plan
        return best_cost, best_plan

    return backtrack(0, None)


def plan_cost(gates: List[Gate], steps: List[PlanStep], estimator: CostEstimator) -> Cost:
    """Compute total cost for a given plan."""

    bounds = _boundaries(gates)
    total = Cost(0.0, 0.0)
    prev: Optional[Backend] = None
    for step in steps:
        segment = gates[step.start : step.end]
        qubits = {q for g in segment for q in g.qubits}
        num_qubits = len(qubits)
        num_gates = step.end - step.start
        sim_cost = _simulation_cost(estimator, step.backend, num_qubits, num_gates)
        conv_cost = Cost(0.0, 0.0)
        if prev is not None and prev != step.backend:
            boundary = bounds[step.start]
            if boundary:
                rank = 2 ** len(boundary)
                frontier = len(boundary)
                conv_cost = estimator.conversion(
                    prev,
                    step.backend,
                    num_qubits=len(boundary),
                    rank=rank,
                    frontier=frontier,
                ).cost
        total = _add_cost(total, _add_cost(conv_cost, sim_cost))
        prev = step.backend
    return total


def random_circuit(num_qubits: int, num_gates: int, seed: int) -> Circuit:
    """Generate a small random circuit."""

    rng = random.Random(seed)
    single = ["H", "X", "Y", "Z", "T", "S"]
    gates = []
    for _ in range(num_gates):
        if num_qubits > 1 and rng.random() < 0.3:
            q1 = rng.randrange(num_qubits)
            q2 = rng.randrange(num_qubits - 1)
            if q2 >= q1:
                q2 += 1
            gates.append({"gate": "CX", "qubits": [q1, q2]})
        else:
            gate = rng.choice(single)
            q = rng.randrange(num_qubits)
            gates.append({"gate": gate, "qubits": [q]})
    return Circuit.from_dict(gates)


def main() -> None:
    estimator = CostEstimator()
    qubit_sizes = [2, 4, 6, 8]
    gate_count = 6
    results = []

    for n in qubit_sizes:
        circ = random_circuit(n, gate_count, seed=n)
        gates = circ.gates

        start = time.perf_counter()
        oracle_cost, oracle_plan = enumerate_oracle(gates, estimator)
        enum_time = time.perf_counter() - start

        planner = Planner(
            quick_max_qubits=None, quick_max_gates=None, quick_max_depth=None
        )
        start = time.perf_counter()
        plan_result = planner.plan(circ)
        plan_time = time.perf_counter() - start
        quasar_cost = plan_cost(gates, plan_result.steps, estimator)

        gap = (
            (quasar_cost.time - oracle_cost.time) / oracle_cost.time
            if oracle_cost.time
            else 0.0
        )
        overhead = enum_time / plan_time if plan_time > 0 else float("inf")
        results.append(
            {
                "qubits": n,
                "gates": len(gates),
                "oracle": oracle_cost.time,
                "quasar": quasar_cost.time,
                "gap": gap,
                "planner_time": plan_time,
                "enum_time": enum_time,
                "overhead": overhead,
            }
        )

    header = f"{'q':>3} {'gates':>5} {'oracle':>10} {'quasar':>10} {'gap%':>7} {'plan_t':>8} {'enum_t':>8} {'overhd':>8}"
    print(header)
    for r in results:
        print(
            f"{r['qubits']:>3} {r['gates']:>5} {r['oracle']:>10.1f} {r['quasar']:>10.1f} "
            f"{r['gap']*100:>6.2f}% {r['planner_time']:>8.4f} {r['enum_time']:>8.4f} {r['overhead']:>8.2f}"
        )


if __name__ == "__main__":
    main()
