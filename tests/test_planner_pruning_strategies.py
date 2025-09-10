"""Tests for planner pruning strategies."""

from quasar.circuit import Circuit
from quasar.cost import Backend, Cost, CostEstimator, ConversionEstimate
from quasar.planner import Planner


class StubEstimator(CostEstimator):
    """Minimal estimator returning controlled costs.

    The stub deliberately reports nearly identical costs for multiple
    backends to trigger epsilon-dominance merging in the planner.
    """

    chi_max = None

    def statevector(self, *args, **kwargs) -> Cost:  # type: ignore[override]
        return Cost(time=1.0, memory=1.0)

    def mps(self, *args, **kwargs) -> Cost:  # type: ignore[override]
        return Cost(time=1.01, memory=1.0)

    def decision_diagram(self, *args, **kwargs) -> Cost:  # type: ignore[override]
        return Cost(time=0.99, memory=1.0)

    def tableau(self, *args, **kwargs) -> Cost:  # type: ignore[override]
        return Cost(time=1.0, memory=1.0)

    def conversion(self, *args, **kwargs) -> ConversionEstimate:  # type: ignore[override]
        return ConversionEstimate("b2b", Cost(0.0, 0.0))

    def parallel_time_overhead(self, groups: int) -> float:  # type: ignore[override]
        return 0.0

    def parallel_memory_overhead(self, groups: int) -> float:  # type: ignore[override]
        return 0.0


def _single_qubit_circuit() -> Circuit:
    return Circuit.from_dict(
        [{"gate": "T", "qubits": [0]}], use_classical_simplification=False
    )


def test_dp_epsilon_dominance_merging():
    circuit = _single_qubit_circuit()
    planner = Planner(estimator=StubEstimator(), epsilon=0.05)
    result = planner._dp(
        circuit.gates,
        epsilon=0.05,
        sparsity=1.0,
        allow_tableau=False,
    )
    assert len(result.table[-1]) == 1


def test_dp_branch_and_bound_prunes_all():
    circuit = _single_qubit_circuit()
    planner = Planner(estimator=StubEstimator())
    result = planner._dp(
        circuit.gates,
        upper_bound=Cost(time=0.5, memory=0.5),
        sparsity=1.0,
        allow_tableau=False,
    )
    assert result.final_backend is None
    assert not result.table[-1]


def test_dp_sliding_horizon_limits_segments():
    gates = []
    for i in range(8):
        if i % 2 == 0:
            gates.append({"gate": "CX", "qubits": [0, 1]})
        else:
            gates.append({"gate": "T", "qubits": [0]})
    circuit = Circuit.from_dict(gates, use_classical_simplification=False)
    planner = Planner(horizon=3)
    result = planner._dp(
        circuit.gates,
        horizon=3,
        sparsity=1.0,
        allow_tableau=False,
    )
    assert result.final_backend is not None
    assert all(step.end - step.start <= 3 for step in result.steps)

