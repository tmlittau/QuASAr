"""Tests for circuits composed of independent parallel subsystems."""

from __future__ import annotations

import pytest

from benchmarks.parallel_circuits import many_ghz_subsystems
from quasar.cost import Backend
from quasar.method_selector import MethodSelector
from quasar.planner import Planner


def test_many_ghz_subsystems_parallel_groups() -> None:
    """Planner should expose disjoint GHZ groups as parallel partitions."""

    num_groups = 4
    group_size = 5
    circuit = many_ghz_subsystems(num_groups=num_groups, group_size=group_size)

    # Verify that gates never span multiple qubit blocks.
    for gate in circuit.gates:
        group_index = gate.qubits[0] // group_size if gate.qubits else 0
        assert all(q // group_size == group_index for q in gate.qubits)

    planner = Planner()
    plan = planner.plan(circuit, backend=Backend.TABLEAU)

    # The forced Tableau backend should treat all groups as a single step with
    # explicit parallel metadata describing each independent subsystem.
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert len(step.parallel) == num_groups
    expected_groups = tuple(
        tuple(range(group * group_size, (group + 1) * group_size))
        for group in range(num_groups)
    )
    assert step.parallel == expected_groups
    assert step.start == 0
    assert step.end == len(circuit.gates)


def test_parallel_groups_with_batched_planner() -> None:
    """Batched DP planning should still emit cached parallel metadata."""

    num_groups = 3
    group_size = 4
    circuit = many_ghz_subsystems(num_groups=num_groups, group_size=group_size)

    planner = Planner(batch_size=3)
    plan = planner.plan(circuit, backend=Backend.TABLEAU)

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert len(step.parallel) == num_groups
    expected_groups = tuple(
        tuple(range(group * group_size, (group + 1) * group_size))
        for group in range(num_groups)
    )
    assert step.parallel == expected_groups
    assert step.start == 0
    assert step.end == len(circuit.gates)


def test_method_selector_respects_parallel_memory_limits() -> None:
    """Selector should base memory on the largest independent subsystem."""

    num_groups = 6
    group_size = 5
    circuit = many_ghz_subsystems(num_groups=num_groups, group_size=group_size)
    selector = MethodSelector()

    def _counts(gates):
        total = len(gates)
        meas = sum(1 for gate in gates if gate.gate.upper() in {"MEASURE", "RESET"})
        singles = sum(
            1
            for gate in gates
            if len(gate.qubits) == 1 and gate.gate.upper() not in {"MEASURE", "RESET"}
        )
        return singles, total - singles - meas, meas

    total_1q, total_2q, total_meas = _counts(circuit.gates)
    combined_cost = selector.estimator.statevector(
        circuit.num_qubits, total_1q, total_2q, total_meas
    )

    single_circuit = many_ghz_subsystems(num_groups=1, group_size=group_size)
    single_1q, single_2q, single_meas = _counts(single_circuit.gates)
    single_cost = selector.estimator.statevector(
        single_circuit.num_qubits, single_1q, single_2q, single_meas
    )

    assert combined_cost.memory > single_cost.memory

    memory_limit = single_cost.memory * 10
    assert memory_limit < combined_cost.memory

    diagnostics: dict[str, object] = {}
    backend, cost = selector.select(
        circuit.gates,
        circuit.num_qubits,
        max_memory=memory_limit,
        allow_tableau=False,
        diagnostics=diagnostics,
    )

    assert backend in {Backend.STATEVECTOR, Backend.MPS}
    backends = diagnostics["backends"]
    sv_entry = backends[Backend.STATEVECTOR]
    assert sv_entry["feasible"] is True
    assert sv_entry["cost"].memory == pytest.approx(single_cost.memory)
