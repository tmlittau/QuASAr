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
