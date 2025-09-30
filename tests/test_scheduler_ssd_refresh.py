from __future__ import annotations

from typing import Any

from quasar.circuit import Circuit, Gate
from quasar.cost import Backend, Cost
from quasar.planner import PlanResult, PlanStep
from quasar.scheduler import Scheduler
from quasar.ssd import ConversionLayer


def _dependency_edges(
    graph: Any,
) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    return [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("kind") == "dependency"
    ]


def test_prepare_run_quick_path_refreshes_ssd_metadata() -> None:
    """Quick-path execution should keep the SSD descriptor in sync."""

    circuit = Circuit(
        [
            Gate("H", [0]),
            Gate("CX", [0, 1]),
        ],
        use_classical_simplification=False,
    )

    scheduler = Scheduler()
    plan = scheduler.prepare_run(circuit, backend=Backend.STATEVECTOR)

    assert len(plan.explicit_steps) == 1
    ssd = circuit.ssd
    assert len(ssd.partitions) == 1

    expected_fingerprint = tuple(part.fingerprint for part in ssd.partitions)
    assert ssd.fingerprint == expected_fingerprint

    graph = circuit.to_networkx_ssd()
    assert graph.graph["fingerprint"] == ssd.fingerprint
    assert _dependency_edges(graph) == []


def test_prepare_run_multi_backend_updates_dependencies() -> None:
    """Merged multi-backend plans must refresh SSD dependencies and fingerprint."""

    gates = [
        Gate("H", [0]),
        Gate("CX", [0, 1]),
    ]
    circuit = Circuit(gates, use_classical_simplification=False)

    conversion = ConversionLayer(
        boundary=(0,),
        source=Backend.STATEVECTOR,
        target=Backend.TABLEAU,
        rank=2,
        frontier=1,
        primitive="B2B",
        cost=Cost(time=0.0, memory=0.0),
    )

    plan = PlanResult(
        table=[],
        final_backend=Backend.TABLEAU,
        gates=gates,
        explicit_steps=[
            PlanStep(start=0, end=1, backend=Backend.STATEVECTOR),
            PlanStep(start=1, end=2, backend=Backend.TABLEAU),
        ],
        explicit_conversions=[conversion],
        step_costs=[Cost(time=0.0, memory=0.0), Cost(time=0.0, memory=0.0)],
    )

    scheduler = Scheduler()
    scheduler.prepare_run(circuit, plan=plan)

    ssd = circuit.ssd
    assert len(ssd.partitions) == 2

    expected_fingerprint = tuple(part.fingerprint for part in ssd.partitions)
    assert ssd.fingerprint == expected_fingerprint

    graph = circuit.to_networkx_ssd()
    assert graph.graph["fingerprint"] == ssd.fingerprint

    deps = _dependency_edges(graph)
    assert (("partition", 0), ("partition", 1)) in deps
    assert len(deps) == 1
