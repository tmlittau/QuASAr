from __future__ import annotations

import types

import numpy as np
import pytest

from benchmarks.bench_utils.circuits import layered_clifford_delayed_magic_circuit

from quasar.circuit import Circuit, Gate
from quasar.cost import Backend, CostEstimator
from quasar.planner import Planner
import quasar.config as config


@pytest.fixture(autouse=True)
def _relax_mps_thresholds(monkeypatch):
    monkeypatch.setattr(
        config.DEFAULT,
        "mps_long_range_fraction_threshold",
        0.95,
        raising=False,
    )
    monkeypatch.setattr(
        config.DEFAULT,
        "mps_long_range_extent_threshold",
        0.95,
        raising=False,
    )


DELAYED_MAGIC_TEST_DEPTH = 40


def random_clifford_t_circuit(
    num_qubits: int, *, depth_multiplier: int = 3, seed: int = 97
) -> Circuit:
    """Replica of the ``random_clifford_t`` workload used in benchmarks."""

    depth = depth_multiplier * num_qubits
    rng = np.random.default_rng(seed + num_qubits)
    gates: list[Gate] = []

    for _ in range(depth):
        for qubit in range(num_qubits):
            gate = rng.choice(["H", "S"])
            gates.append(Gate(gate, [qubit]))
        if num_qubits > 1:
            a, b = rng.choice(num_qubits, 2, replace=False)
            two_gate = "CX" if rng.random() < 0.5 else "CZ"
            gates.append(Gate(two_gate, [int(a), int(b)]))
        target = int(rng.integers(num_qubits)) if num_qubits else 0
        gates.append(Gate("T", [target]))

    return Circuit(gates, use_classical_simplification=False)


def calibrated_estimator() -> CostEstimator:
    """Return an estimator where MPS beats dense simulation for the workload."""

    estimator = CostEstimator()
    estimator.coeff.update(
        {
            "mps_gate_1q": 0.001,
            "mps_gate_2q": 0.001,
            "mps_trunc": 0.001,
            "mps_base_time": 0.0,
            "mps_long_range_weight": 0.05,
            "mps_long_range_extent_weight": 0.05,
            "sv_gate_1q": 1.0,
            "sv_gate_2q": 1.0,
            "sv_base_time": 0.5,
        }
    )
    return estimator


def test_planner_prefers_mps_for_random_clifford_when_faster() -> None:
    circuit = random_clifford_t_circuit(8)
    estimator = calibrated_estimator()

    def fixed_chi_for_constraints(
        self: CostEstimator,
        num_qubits: int,
        gates: list[Gate],
        fidelity: float,
        max_memory: float | None = None,
    ) -> int:
        return 8

    estimator.chi_for_constraints = types.MethodType(
        fixed_chi_for_constraints, estimator
    )

    planner = Planner(
        estimator=estimator,
        quick_max_qubits=64,
        quick_max_gates=512,
        quick_max_depth=512,
    )

    plan = planner.plan(circuit, explain=True, target_accuracy=0.8)
    assert plan.final_backend == Backend.MPS
    assert plan.diagnostics is not None

    single = plan.diagnostics.backend_selection["single"]
    assert single["selected_backend"] == Backend.MPS

    metrics = single["metrics"]
    assert metrics["mps_long_range_fraction"] > 0.0

    backends = single["backends"]
    assert backends[Backend.MPS]["feasible"] is True
    assert backends[Backend.STATEVECTOR]["feasible"] is True

    mps_time = backends[Backend.MPS]["cost"].time
    sv_time = backends[Backend.STATEVECTOR]["cost"].time
    assert mps_time < sv_time


def test_forced_mps_respects_memory_cap_for_delayed_magic() -> None:
    circuit = layered_clifford_delayed_magic_circuit(
        12, depth=DELAYED_MAGIC_TEST_DEPTH
    )
    estimator = CostEstimator()
    planner = Planner(
        estimator=estimator,
        max_memory=250_000_000,
        quick_max_qubits=None,
        quick_max_gates=None,
        quick_max_depth=None,
    )

    plan = planner.plan(circuit, backend=Backend.MPS, use_cache=False)

    assert plan.final_backend == Backend.MPS

