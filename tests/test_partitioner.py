from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from quasar.circuit import Circuit, Gate
from quasar.cost import Backend, Cost, ConversionEstimate, CostEstimator
from quasar.partitioner import Partitioner
from quasar.sparsity import sparsity_estimate
from quasar.symmetry import (
    amplitude_rotation_diversity as rot_amp,
    phase_rotation_diversity as rot_phase,
)


@dataclass
class SimpleEstimator:
    """Estimator returning constant costs for all backends."""

    time: float = 1.0
    memory: float = 1.0
    coeff: dict[str, float] = field(default_factory=dict)

    _baseline_estimator: CostEstimator = field(default_factory=CostEstimator, init=False, repr=False)

    def conversion(  # type: ignore[no-untyped-def]
        self,
        source,
        target,
        num_qubits,
        rank,
        frontier,
        compressed_terms=None,
        window=None,
        **_kwargs,
    ) -> ConversionEstimate:
        return ConversionEstimate(
            "FAKE",
            Cost(time=0.0, memory=0.0, log_depth=float(frontier)),
            window=window,
        )

    def tableau(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=self.time, memory=self.memory)

    def mps(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=self.time, memory=self.memory)

    def decision_diagram(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=self.time, memory=self.memory)

    def extended_stabilizer(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=self.time, memory=self.memory)

    def statevector(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=self.time, memory=self.memory)

    def derive_conversion_window(self, num_qubits, *, rank, compressed_terms=None, bond_dimension=None):  # type: ignore[no-untyped-def]
        return min(num_qubits, 4)

    def bond_dimensions(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        return self._baseline_estimator.bond_dimensions(num_qubits, gates)

    def max_schmidt_rank(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        return self._baseline_estimator.max_schmidt_rank(num_qubits, gates)


class MetricsAssertingSelector:
    """Selector that validates analysis metrics passed by the partitioner."""

    def __init__(self, estimator: SimpleEstimator) -> None:
        self.estimator = estimator
        self.calls = 0

    def select(  # type: ignore[no-untyped-def]
        self,
        gates,
        num_qubits,
        *,
        sparsity,
        phase_rotation_diversity,
        amplitude_rotation_diversity,
        **kwargs,
    ):
        fragment = Circuit(list(gates), use_classical_simplification=False)
        expected_sparsity = sparsity_estimate(fragment)
        expected_phase = rot_phase(fragment)
        expected_amplitude = rot_amp(fragment)
        assert sparsity == pytest.approx(expected_sparsity)
        assert phase_rotation_diversity == expected_phase
        assert amplitude_rotation_diversity == expected_amplitude
        self.calls += 1
        return Backend.STATEVECTOR, Cost(time=1.0, memory=1.0)


class LinearSelector:
    """Selector returning a simple cost proportional to gate arity."""

    def select(  # type: ignore[no-untyped-def]
        self,
        gates,
        num_qubits,
        *,
        sparsity,
        phase_rotation_diversity,
        amplitude_rotation_diversity,
        **kwargs,
    ):
        num_1q = sum(1 for g in gates if len(g.qubits) == 1)
        num_multi = sum(1 for g in gates if len(g.qubits) > 1)
        time = float(num_1q + 2 * num_multi)
        memory = float(max(1, num_qubits))
        return Backend.STATEVECTOR, Cost(time=time, memory=memory)


def test_partitioner_metrics_match_circuit_analysis() -> None:
    estimator = SimpleEstimator()
    selector = MetricsAssertingSelector(estimator)
    partitioner = Partitioner(estimator=estimator, selector=selector)
    circuit = Circuit(
        [
            Gate("H", [0]),
            Gate("T", [0]),
            Gate("CX", [0, 1]),
            Gate("RZ", [1], {"theta": 0.125}),
            Gate("RY", [1], {"theta": 0.5}),
            Gate("CRX", [1, 2], {"theta": 0.75}),
            Gate("MEASURE", [2]),
        ],
        use_classical_simplification=False,
    )

    partitioner.partition(circuit, debug=True)

    assert selector.calls > 0


def test_graph_cut_widens_segments_for_entanglement() -> None:
    estimator = SimpleEstimator()
    selector = LinearSelector()
    partitioner = Partitioner(
        estimator=estimator,
        selector=selector,
        graph_cut_candidate_limit=2,
        graph_cut_neighbor_radius=1,
    )

    future_gate = Gate("T", [3])
    future = {3}

    weak_fragment = [
        Gate("H", [0]),
        Gate("CX", [0, 3]),
        Gate("H", [1]),
    ]

    entangled_fragment = [
        Gate("H", [0]),
        Gate("H", [1]),
        Gate("H", [2]),
        Gate("CX", [0, 3]),
        Gate("CX", [1, 3]),
        Gate("CX", [2, 3]),
    ]

    weak_idx, _ = partitioner._select_cut_point(weak_fragment, future_gate, future)
    entangled_idx, _ = partitioner._select_cut_point(
        entangled_fragment, future_gate, future
    )

    assert entangled_idx > weak_idx


def test_partitioner_handles_independent_subcircuits() -> None:
    """Independent subcircuits should be represented as parallel subsystems."""

    gates = [
        Gate("H", [0]),
        Gate("H", [2]),
        Gate("CX", [0, 1]),
        Gate("CX", [2, 3]),
        Gate("T", [0]),
        Gate("T", [2]),
        Gate("S", [1]),
        Gate("S", [3]),
    ]
    circuit = Circuit(gates, use_classical_simplification=False)

    partitioner = Partitioner()
    ssd = partitioner.partition(circuit)

    assert len(ssd.partitions) == 1
    partition = ssd.partitions[0]
    assert partition.multiplicity == 2
    assert partition.subsystems == ((0, 1), (2, 3))

    def _count_ops(sequence: list[Gate]) -> tuple[int, int, int]:
        num_meas = sum(1 for gate in sequence if gate.gate.upper() in {"MEASURE", "RESET"})
        num_1q = sum(
            1
            for gate in sequence
            if len(gate.qubits) == 1 and gate.gate.upper() not in {"MEASURE", "RESET"}
        )
        num_2q = len(sequence) - num_1q - num_meas
        return num_1q, num_2q, num_meas

    subsystem = partition.subsystems[0]
    subsystem_set = set(subsystem)
    subsystem_gates = [gate for gate in gates if set(gate.qubits).issubset(subsystem_set)]
    sub_1q, sub_2q, sub_meas = _count_ops(subsystem_gates)
    num_qubits = len(subsystem)
    backend = partition.backend
    if backend == Backend.TABLEAU:
        expected_subsystem_cost = partitioner.estimator.tableau(
            num_qubits, len(subsystem_gates), num_meas=sub_meas
        )
    elif backend == Backend.EXTENDED_STABILIZER:
        num_t = sum(1 for gate in subsystem_gates if gate.gate.upper() in {"T", "TDG"})
        num_clifford = max(0, len(subsystem_gates) - num_t - sub_meas)
        expected_subsystem_cost = partitioner.estimator.extended_stabilizer(
            num_qubits,
            num_clifford,
            num_t,
            num_meas=sub_meas,
            depth=len(subsystem_gates),
        )
    elif backend == Backend.MPS:
        expected_subsystem_cost = partitioner.estimator.mps(
            num_qubits,
            sub_1q + sub_meas,
            sub_2q,
            chi=4,
            svd=True,
        )
    elif backend == Backend.DECISION_DIAGRAM:
        expected_subsystem_cost = partitioner.estimator.decision_diagram(
            num_gates=len(subsystem_gates),
            frontier=num_qubits,
        )
    else:
        expected_subsystem_cost = partitioner.estimator.statevector(
            num_qubits, sub_1q, sub_2q, sub_meas
        )

    assert partition.cost.memory == pytest.approx(expected_subsystem_cost.memory)
    assert partition.cost.time == pytest.approx(expected_subsystem_cost.time)

    total_1q, total_2q, total_meas = _count_ops(gates)
    naive_cost = partitioner.estimator.statevector(4, total_1q, total_2q, total_meas)

    assert partition.cost.memory * partition.multiplicity < naive_cost.memory
