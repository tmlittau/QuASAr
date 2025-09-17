"""Tests for hybrid circuits requiring multiple simulation backends."""

from __future__ import annotations

from benchmarks.circuits import CLIFFORD_GATES
from benchmarks.partition_circuits import mixed_backend_subsystems
from quasar.cost import Backend
from quasar.partitioner import Partitioner


def test_mixed_backend_subsystems_partitioning() -> None:
    """GHZ, QAOA and dense blocks should map to distinct backends."""

    ghz_width = 4
    qaoa_width = 4
    dense_width = 4
    circuit = mixed_backend_subsystems(
        ghz_width=ghz_width,
        qaoa_width=qaoa_width,
        qaoa_layers=2,
        random_width=dense_width,
        seed=11,
    )

    partitioner = Partitioner()
    ssd = partitioner.partition(circuit, debug=True)

    assert len(ssd.partitions) >= 3

    ghz_qubits = set(range(ghz_width))
    qaoa_offset = ghz_width
    dense_offset = qaoa_offset + qaoa_width
    dense_qubits = set(range(dense_offset, dense_offset + dense_width))

    tableau_partition = next(
        (
            part
            for part in ssd.partitions
            if part.backend == Backend.TABLEAU
            and ghz_qubits.issubset(set(part.qubits))
        ),
        None,
    )
    assert tableau_partition is not None
    assert set(tableau_partition.history).issubset(CLIFFORD_GATES)

    dense_partitions = [
        part
        for part in ssd.partitions
        if part.backend in {Backend.STATEVECTOR, Backend.DECISION_DIAGRAM}
        and dense_qubits.issubset(set(part.qubits))
    ]
    assert dense_partitions
    assert any(
        any(name in {"T", "TDG"} for name in part.history)
        for part in dense_partitions
    )

    applied_trace = [entry for entry in ssd.trace if entry.applied]
    assert any(entry.to_backend == Backend.MPS for entry in applied_trace)
    assert any(
        entry.from_backend == Backend.MPS and entry.to_backend == Backend.STATEVECTOR
        for entry in applied_trace
    )

    assert any(
        conv.source == Backend.TABLEAU and conv.target in {Backend.DECISION_DIAGRAM, Backend.MPS}
        for conv in ssd.conversions
    )
    assert any(
        conv.source == Backend.DECISION_DIAGRAM and conv.target == Backend.MPS
        for conv in ssd.conversions
    )
