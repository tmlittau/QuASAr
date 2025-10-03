from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

from quasar.circuit import Circuit, Gate
from quasar.cost import Backend, Cost, ConversionEstimate, CostEstimator
from quasar.partitioner import Partitioner
from quasar.ssd import PartitionTraceEntry


class DummySelector:
    """Return pre-programmed backend choices for each ``select`` call."""

    def __init__(self, results: Iterable[Tuple[Backend, Cost]]):
        self._results: List[Tuple[Backend, Cost]] = list(results)
        self._index = 0

    def select(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self._index < len(self._results):
            backend, cost = self._results[self._index]
        else:
            backend, cost = self._results[-1]
        self._index += 1
        return backend, cost


@dataclass
class FakeEstimator:
    """Minimal cost estimator returning deterministic values."""

    conversion_time: float = 3.0
    conversion_memory: float = 7.0
    coeff: dict[str, float] = field(default_factory=dict)
    baseline: CostEstimator = field(default_factory=CostEstimator, init=False, repr=False)

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
            Cost(
                time=self.conversion_time * num_qubits,
                memory=self.conversion_memory * rank,
                log_depth=float(frontier),
            ),
            window=window,
        )

    def tableau(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=1.0, memory=2.0)

    def mps(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=1.0, memory=2.0)

    def decision_diagram(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=1.0, memory=2.0)

    def extended_stabilizer(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=1.0, memory=2.0)

    def statevector(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=1.0, memory=2.0)

    def derive_conversion_window(self, num_qubits, *, rank, compressed_terms=None, bond_dimension=None):  # type: ignore[no-untyped-def]
        return min(num_qubits, 4)

    def bond_dimensions(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        return self.baseline.bond_dimensions(num_qubits, gates)

    def max_schmidt_rank(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        return self.baseline.max_schmidt_rank(num_qubits, gates)


@dataclass
class CountingEstimator(FakeEstimator):
    """Estimator tracking entanglement queries."""

    bond_queries: int = 0
    rank_queries: int = 0

    def bond_dimensions(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        self.bond_queries += 1
        return super().bond_dimensions(num_qubits, gates)

    def max_schmidt_rank(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        self.rank_queries += 1
        return super().max_schmidt_rank(num_qubits, gates)


@dataclass
class PrimitiveSwitchEstimator(FakeEstimator):
    """Estimator selecting primitives based on rank."""

    threshold: int = 2
    bond_queries: int = 0
    rank_queries: int = 0
    last_rank: int | None = None
    last_frontier: int | None = None

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
        self.last_rank = rank
        self.last_frontier = frontier
        primitive = "ST" if rank <= self.threshold else "B2B"
        return ConversionEstimate(
            primitive,
            Cost(
                time=self.conversion_time * max(rank, 1),
                memory=self.conversion_memory * max(frontier, 1),
                log_depth=float(frontier),
            ),
            window=window,
        )

    def bond_dimensions(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        self.bond_queries += 1
        return super().bond_dimensions(num_qubits, gates)

    def max_schmidt_rank(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        self.rank_queries += 1
        return super().max_schmidt_rank(num_qubits, gates)

    def tableau(self, _num_qubits, num_gates, **_kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=50.0 * num_gates, memory=10.0)

    def mps(  # type: ignore[no-untyped-def]
        self,
        _num_qubits,
        num_1q,
        num_2q,
        *_args,
        **_kwargs,
    ) -> Cost:
        operations = max(num_1q + num_2q, 1)
        return Cost(time=float(operations), memory=1.0)


@dataclass
class LinearEstimator:
    """Estimator with linear gate-dependent costs for backlog testing."""

    prefix_rate: float
    suffix_rate: float
    conversion_cost: float
    coeff: dict[str, float] = field(default_factory=dict)
    baseline: CostEstimator = field(default_factory=CostEstimator, init=False, repr=False)

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
            Cost(time=self.conversion_cost, memory=1.0, log_depth=float(frontier)),
            window=window,
        )

    def tableau(self, _num_qubits, num_gates, **_kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=self.prefix_rate * num_gates, memory=1.0)

    def mps(  # type: ignore[no-untyped-def]
        self,
        _num_qubits,
        num_1q,
        num_2q,
        *_args,
        **_kwargs,
    ) -> Cost:
        return Cost(time=self.suffix_rate * (num_1q + num_2q), memory=1.0)

    def decision_diagram(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=1000.0, memory=1.0)

    def extended_stabilizer(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=1000.0, memory=1.0)

    def statevector(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return Cost(time=1000.0, memory=1.0)

    def derive_conversion_window(self, num_qubits, *, rank, compressed_terms=None, bond_dimension=None):  # type: ignore[no-untyped-def]
        return min(num_qubits, 4)

    def bond_dimensions(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        return self.baseline.bond_dimensions(num_qubits, gates)

    def max_schmidt_rank(self, num_qubits, gates):  # type: ignore[no-untyped-def]
        return self.baseline.max_schmidt_rank(num_qubits, gates)


def build_circuit(*gates: Gate) -> Circuit:
    return Circuit(list(gates), use_classical_simplification=False)


def assert_trace_entry(
    entry: PartitionTraceEntry,
    *,
    reason: str,
    applied: bool,
    source: Backend | None,
    target: Backend,
    boundary_size: int,
    primitive: str = "FAKE",
    window: int | None = None,
    break_even: int | None = None,
):
    assert entry.reason == reason
    assert entry.applied is applied
    assert entry.from_backend == source
    assert entry.to_backend == target
    assert entry.boundary_size == boundary_size
    assert entry.break_even_horizon == break_even
    if boundary_size:
        assert 1 <= entry.rank <= 2**boundary_size
        assert 0 <= entry.frontier <= boundary_size
        if entry.rank == 1:
            assert entry.frontier == 0
        assert entry.primitive == primitive
        assert entry.cost is not None
        assert entry.cost.time > 0
        if window is not None:
            assert entry.window == window
    else:
        assert entry.rank == 1
        assert entry.frontier == 0
        assert entry.primitive in (None, primitive)


def test_trace_records_statevector_lock() -> None:
    estimator = FakeEstimator()
    selector = DummySelector(
        [
            (Backend.STATEVECTOR, Cost(time=1.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=2.0)),
        ]
    )
    partitioner = Partitioner(
        estimator=estimator, selector=selector, target_accuracy=0.999
    )
    circuit = build_circuit(Gate("H", [0]), Gate("CX", [0, 1]))

    ssd = partitioner.partition(circuit, debug=True)

    assert len(ssd.trace) == 1
    entry = ssd.trace[0]
    assert_trace_entry(
        entry,
        reason="statevector_lock",
        applied=False,
        source=Backend.STATEVECTOR,
        target=Backend.MPS,
        boundary_size=1,
    )


def test_trace_marks_single_qubit_preamble_switch() -> None:
    estimator = FakeEstimator()
    selector = DummySelector(
        [
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=2.0)),
        ]
    )
    partitioner = Partitioner(
        estimator=estimator, selector=selector, target_accuracy=0.999
    )
    circuit = build_circuit(Gate("H", [0]), Gate("T", [0]))

    ssd = partitioner.partition(circuit, debug=True)

    assert len(ssd.trace) == 1
    entry = ssd.trace[0]
    assert_trace_entry(
        entry,
        reason="single_qubit_preamble",
        applied=True,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
    )


def test_trace_records_deferred_switch_candidate() -> None:
    estimator = FakeEstimator()
    selector = DummySelector(
        [
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=2.0)),
        ]
    )
    partitioner = Partitioner(
        estimator=estimator, selector=selector, target_accuracy=0.999
    )
    circuit = build_circuit(Gate("CX", [0, 1]), Gate("T", [0]))

    ssd = partitioner.partition(circuit, debug=True)

    assert len(ssd.trace) == 1
    entry = ssd.trace[0]
    assert_trace_entry(
        entry,
        reason="deferred_switch_candidate",
        applied=False,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
    )
    assert not ssd.conversions


def test_deferred_switch_materialises_when_cost_favourable() -> None:
    estimator = LinearEstimator(prefix_rate=10.0, suffix_rate=1.0, conversion_cost=12.0)
    selector = DummySelector(
        [
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
        ]
    )
    partitioner = Partitioner(estimator=estimator, selector=selector)
    circuit = build_circuit(
        Gate("CX", [0, 1]),
        Gate("T", [0]),
        Gate("T", [0]),
    )

    ssd = partitioner.partition(circuit, debug=True)

    assert len(ssd.conversions) == 1
    assert len(ssd.trace) == 3
    first_candidate, second_candidate, applied = ssd.trace
    assert_trace_entry(
        first_candidate,
        reason="deferred_switch_candidate",
        applied=False,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
        break_even=2,
    )
    assert first_candidate.gate_index == 1
    assert_trace_entry(
        second_candidate,
        reason="deferred_switch_candidate",
        applied=False,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
        break_even=2,
    )
    assert second_candidate.gate_index == 2
    assert_trace_entry(
        applied,
        reason="deferred_backend_switch",
        applied=True,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
        break_even=2,
    )
    assert applied.gate_index == 2
    conversion = ssd.conversions[0]
    assert conversion.source == Backend.TABLEAU
    assert conversion.target == Backend.MPS
    backends = [part.backend for part in ssd.partitions]
    assert Backend.TABLEAU in backends
    assert Backend.MPS in backends


def test_deferred_switch_projection_estimates_break_even_horizon() -> None:
    estimator = LinearEstimator(prefix_rate=8.0, suffix_rate=1.0, conversion_cost=20.0)
    selector = DummySelector(
        [
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
        ]
    )
    partitioner = Partitioner(estimator=estimator, selector=selector)
    circuit = build_circuit(
        Gate("CX", [0, 1]),
        Gate("T", [0]),
        Gate("T", [0]),
        Gate("T", [0]),
        Gate("T", [0]),
    )

    ssd = partitioner.partition(circuit, debug=True)

    assert len(ssd.conversions) == 1
    # Three candidates followed by an applied entry once the savings amortise the conversion.
    assert len(ssd.trace) == 4
    first, second, third, applied = ssd.trace
    assert_trace_entry(
        first,
        reason="deferred_switch_candidate",
        applied=False,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
        break_even=3,
    )
    assert_trace_entry(
        second,
        reason="deferred_switch_candidate",
        applied=False,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
        break_even=3,
    )
    assert_trace_entry(
        third,
        reason="deferred_switch_candidate",
        applied=False,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
        break_even=3,
    )
    assert_trace_entry(
        applied,
        reason="deferred_backend_switch",
        applied=True,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
        break_even=3,
    )


def test_deferred_switch_abandons_when_lookahead_insufficient() -> None:
    estimator = LinearEstimator(prefix_rate=8.0, suffix_rate=1.0, conversion_cost=50.0)
    selector = DummySelector(
        [
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=1.0)),
        ]
    )
    partitioner = Partitioner(estimator=estimator, selector=selector)
    circuit = build_circuit(
        Gate("CX", [0, 1]),
        Gate("T", [0]),
        Gate("T", [0]),
        Gate("T", [0]),
    )

    ssd = partitioner.partition(circuit, debug=True)

    assert not ssd.conversions
    assert len(ssd.trace) == 3
    for entry in ssd.trace:
        assert_trace_entry(
            entry,
            reason="deferred_switch_candidate",
            applied=False,
            source=Backend.TABLEAU,
            target=Backend.MPS,
            boundary_size=1,
            break_even=None,
        )
    assert all(part.backend == Backend.TABLEAU for part in ssd.partitions)


def test_entanglement_bounds_cached_for_deferred_switch() -> None:
    estimator = CountingEstimator()
    selector = DummySelector(
        [
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=2.0)),
            (Backend.MPS, Cost(time=2.0, memory=2.0)),
        ]
    )
    partitioner = Partitioner(estimator=estimator, selector=selector)
    circuit = build_circuit(
        Gate("H", [0]),
        Gate("CX", [0, 1]),
        Gate("H", [2]),
        Gate("CX", [1, 2]),
        Gate("CZ", [1, 2]),
    )

    ssd = partitioner.partition(circuit, debug=True)

    reasons = [entry.reason for entry in ssd.trace]
    assert reasons.count("deferred_switch_candidate") >= 2
    assert estimator.bond_queries <= 2


def test_conversion_primitive_respects_entanglement_bound() -> None:
    selector = DummySelector(
        [
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=2.0)),
        ]
    )
    estimator = PrimitiveSwitchEstimator(threshold=2, conversion_memory=1.0)
    partitioner = Partitioner(
        estimator=estimator, selector=selector, target_accuracy=0.999
    )
    circuit = build_circuit(
        Gate("H", [0]),
        Gate("CX", [0, 1]),
        Gate("H", [2]),
        Gate("CX", [1, 2]),
    )

    ssd = partitioner.partition(circuit, debug=True)

    assert ssd.conversions
    layer = ssd.conversions[0]
    assert layer.primitive == "ST"
    assert layer.rank == 1
    assert layer.frontier == 0
    assert estimator.last_rank == 1
    assert estimator.last_frontier == 0

    dense_selector = DummySelector(
        [
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.TABLEAU, Cost(time=1.0, memory=1.0)),
            (Backend.MPS, Cost(time=2.0, memory=2.0)),
        ]
    )
    dense_estimator = PrimitiveSwitchEstimator(threshold=2, conversion_memory=1.0)
    dense_partitioner = Partitioner(
        estimator=dense_estimator, selector=dense_selector, target_accuracy=0.999
    )
    dense_circuit = build_circuit(
        Gate("H", [0]),
        Gate("H", [1]),
        Gate("CX", [0, 2]),
        Gate("CX", [1, 3]),
        Gate("CX", [2, 3]),
    )

    dense_ssd = dense_partitioner.partition(dense_circuit, debug=True)

    assert dense_ssd.conversions
    dense_layer = dense_ssd.conversions[0]
    assert dense_layer.primitive == "B2B"
    assert dense_layer.rank >= 4
    assert dense_layer.frontier >= 2
    assert dense_estimator.last_rank is not None and dense_estimator.last_rank >= 4
    assert dense_estimator.last_frontier is not None and dense_estimator.last_frontier >= 2

