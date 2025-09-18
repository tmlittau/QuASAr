from __future__ import annotations
import math

from docs.utils.partitioning_analysis import (
    BoundarySpec,
    FragmentStats,
    aggregate_partitioned_plan,
    aggregate_single_backend_plan,
    evaluate_fragment_backends,
)
from quasar.cost import Backend, CostEstimator


def test_evaluate_fragment_backends_tableau_preferred():
    stats = FragmentStats(
        num_qubits=3,
        num_1q_gates=4,
        num_2q_gates=2,
        num_measurements=1,
        is_clifford=True,
    )
    backend, diagnostics = evaluate_fragment_backends(stats)
    assert backend == Backend.TABLEAU
    table_entry = diagnostics["backends"][Backend.TABLEAU]
    assert table_entry["feasible"]
    assert table_entry["selected"]


def test_evaluate_fragment_backends_surface_dd_metric():
    stats = FragmentStats(
        num_qubits=12,
        num_1q_gates=5,
        num_2q_gates=4,
        num_measurements=1,
        is_clifford=False,
    )
    backend, diagnostics = evaluate_fragment_backends(
        stats,
        sparsity=0.95,
        phase_rotation_diversity=2,
        amplitude_rotation_diversity=1,
    )
    dd_entry = diagnostics["backends"][Backend.DECISION_DIAGRAM]
    assert dd_entry["feasible"]
    assert "metric" in dd_entry
    assert dd_entry["metric"] >= dd_entry["dd_metric_threshold"]
    assert dd_entry["selected"] == (backend == Backend.DECISION_DIAGRAM)


def test_evaluate_fragment_backends_local_mps_reports_chi():
    stats = FragmentStats(
        num_qubits=6,
        num_1q_gates=10,
        num_2q_gates=5,
        is_local=True,
    )
    backend, diagnostics = evaluate_fragment_backends(stats, max_memory=10**9)
    mps_entry = diagnostics["backends"][Backend.MPS]
    assert mps_entry["feasible"]
    assert "chi" in mps_entry
    assert backend in {Backend.MPS, Backend.DECISION_DIAGRAM, Backend.STATEVECTOR, Backend.TABLEAU}


def test_aggregate_single_backend_plan_sums_costs():
    estimator = CostEstimator()
    cost_a = estimator.statevector(3, 4, 1, 0)
    cost_b = estimator.statevector(4, 6, 2, 1)
    total = aggregate_single_backend_plan(
        [
            (Backend.STATEVECTOR, cost_a),
            (Backend.STATEVECTOR, cost_b),
        ]
    )
    assert math.isclose(total.time, cost_a.time + cost_b.time)
    assert total.memory == max(cost_a.memory, cost_b.memory)


def test_aggregate_partitioned_plan_includes_conversions():
    estimator = CostEstimator()
    fragments = [
        (Backend.STATEVECTOR, estimator.statevector(3, 4, 1, 0)),
        (Backend.MPS, estimator.mps(4, 5, 2, chi=4)),
    ]
    boundary = BoundarySpec(num_qubits=2, rank=2, frontier=2)
    plan = aggregate_partitioned_plan(fragments, [boundary], estimator=estimator)
    conversions = plan["conversions"]
    assert len(conversions) == 1
    step = conversions[0]
    assert step["primitive"] in {"B2B", "LW", "ST", "Full"}
    conversion_cost = step["cost"]
    assert plan["total_cost"].time >= sum(cost.time for _, cost in fragments)
    assert plan["total_cost"].conversion >= conversion_cost.time
    assert plan["total_cost"].memory >= max(cost.memory for _, cost in fragments)
