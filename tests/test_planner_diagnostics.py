from __future__ import annotations

from quasar.planner import Planner
from quasar.scheduler import Scheduler
from quasar.cost import Cost
from benchmarks.circuits import adder_circuit


def make_planner() -> Planner:
    return Planner(
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
    )


def test_planner_collects_diagnostics() -> None:
    circuit = adder_circuit(1)
    planner = make_planner()

    result = planner.plan(circuit, use_cache=False, explain=True)

    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.strategy is not None
    assert isinstance(diagnostics.single_cost, Cost)
    assert isinstance(diagnostics.pre_cost, Cost)
    assert isinstance(diagnostics.dp_cost, Cost)
    conversions = diagnostics.conversion_estimates
    assert isinstance(conversions, list)
    if conversions:
        estimate = conversions[0]
        assert estimate.stage in {"pre", "coarse", "refine"}
        assert estimate.source is not None
        assert estimate.boundary
        assert isinstance(estimate.cost, Cost)
        assert estimate.feasible in {True, False}


def test_scheduler_prepares_diagnostics() -> None:
    circuit = adder_circuit(1)
    scheduler = Scheduler(
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
    )

    plan = scheduler.prepare_run(circuit, explain=True)

    diagnostics = plan.diagnostics
    assert diagnostics is not None
    assert isinstance(diagnostics.conversion_estimates, list)
    assert diagnostics.single_cost is not None
