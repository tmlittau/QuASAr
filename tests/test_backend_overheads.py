"""Calibration smoke test ensuring backend baselines influence planning."""

from __future__ import annotations

import pytest

from quasar.calibration import benchmark_dd_baseline, benchmark_mps_baseline
from quasar.circuit import Circuit, Gate
from quasar.cost import Backend, CostEstimator
from quasar.method_selector import MethodSelector
from quasar.planner import Planner


def test_backend_overheads_drive_planning_choice() -> None:
    """Calibrate backend baselines and ensure the planner honours them."""

    mps_coeff = benchmark_mps_baseline(num_qubits=2)
    dd_coeff = benchmark_dd_baseline(num_qubits=2)
    coeff = {**mps_coeff, **dd_coeff}
    estimator = CostEstimator(coeff=coeff)

    # Baseline costs should be reflected directly in zero-gate estimates.
    mps_idle = estimator.mps(num_qubits=1, num_1q_gates=0, num_2q_gates=0, chi=1)
    assert mps_idle.time == pytest.approx(mps_coeff["mps_base_time"])
    assert mps_idle.memory - estimator.coeff["mps_mem"] == pytest.approx(
        mps_coeff["mps_base_mem"]
    )

    dd_idle = estimator.decision_diagram(num_gates=0, frontier=1)
    assert dd_idle.time == pytest.approx(dd_coeff["dd_base_time"])
    assert dd_idle.memory == pytest.approx(dd_coeff["dd_base_mem"])

    gates = [Gate("H", [0]), Gate("CX", [0, 1]), Gate("T", [0])]
    circuit = Circuit(gates, use_classical_simplification=False)
    circuit.sparsity = 1.0
    circuit.phase_rotation_diversity = 0
    circuit.amplitude_rotation_diversity = 0

    num_meas = sum(1 for gate in gates if gate.gate.upper() in {"MEASURE", "RESET"})
    num_1q = sum(
        1 for gate in gates if len(gate.qubits) == 1 and gate.gate.upper() not in {"MEASURE", "RESET"}
    )
    num_2q = len(gates) - num_1q - num_meas

    dd_cost = estimator.decision_diagram(num_gates=len(gates), frontier=circuit.num_qubits)
    mps_cost = estimator.mps(
        circuit.num_qubits,
        num_1q + num_meas,
        num_2q,
        chi=4,
        svd=True,
    )
    assert dd_cost.memory <= mps_cost.memory
    assert dd_cost.time <= mps_cost.time

    selector = MethodSelector(estimator)
    planner = Planner(
        estimator=estimator,
        selector=selector,
        quick_max_qubits=0,
        quick_max_gates=0,
        quick_max_depth=0,
    )

    expected_backend, _ = selector.select(
        circuit.gates,
        circuit.num_qubits,
        sparsity=circuit.sparsity,
        phase_rotation_diversity=circuit.phase_rotation_diversity,
        amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
    )
    assert expected_backend == Backend.DECISION_DIAGRAM

    plan = planner.plan(circuit)
    assert plan.final_backend == Backend.DECISION_DIAGRAM
    assert all(step.backend == Backend.DECISION_DIAGRAM for step in plan.steps)
