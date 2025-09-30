from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from quasar.circuit import Circuit, Gate
from quasar.cost import Backend, Cost, ConversionEstimate
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

    def conversion(  # type: ignore[no-untyped-def]
        self,
        source,
        target,
        num_qubits,
        rank,
        frontier,
        **_kwargs,
    ) -> ConversionEstimate:
        return ConversionEstimate(
            "FAKE",
            Cost(
                time=self.conversion_time * num_qubits,
                memory=self.conversion_memory * rank,
                log_depth=float(frontier),
            ),
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


@dataclass
class LinearEstimator:
    """Estimator with linear gate-dependent costs for backlog testing."""

    prefix_rate: float
    suffix_rate: float
    conversion_cost: float

    def conversion(  # type: ignore[no-untyped-def]
        self,
        source,
        target,
        num_qubits,
        rank,
        frontier,
        **_kwargs,
    ) -> ConversionEstimate:
        return ConversionEstimate(
            "FAKE",
            Cost(time=self.conversion_cost, memory=1.0, log_depth=float(frontier)),
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
):
    assert entry.reason == reason
    assert entry.applied is applied
    assert entry.from_backend == source
    assert entry.to_backend == target
    assert entry.boundary_size == boundary_size
    if boundary_size:
        assert entry.rank == 2**boundary_size
        assert entry.frontier == boundary_size
        assert entry.primitive == primitive
        assert entry.cost is not None
        assert entry.cost.time > 0
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
    partitioner = Partitioner(estimator=estimator, selector=selector)
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
    partitioner = Partitioner(estimator=estimator, selector=selector)
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
    partitioner = Partitioner(estimator=estimator, selector=selector)
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
    )
    assert first_candidate.gate_index == 1
    assert_trace_entry(
        second_candidate,
        reason="deferred_switch_candidate",
        applied=False,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
    )
    assert second_candidate.gate_index == 2
    assert_trace_entry(
        applied,
        reason="deferred_backend_switch",
        applied=True,
        source=Backend.TABLEAU,
        target=Backend.MPS,
        boundary_size=1,
    )
    assert applied.gate_index == 2
    conversion = ssd.conversions[0]
    assert conversion.source == Backend.TABLEAU
    assert conversion.target == Backend.MPS
    backends = [part.backend for part in ssd.partitions]
    assert Backend.TABLEAU in backends
    assert Backend.MPS in backends

