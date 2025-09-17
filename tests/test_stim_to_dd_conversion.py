"""Tests for the stabiliser-to-decision-diagram conversion benchmark."""

from __future__ import annotations

from benchmarks.partition_circuits import stim_to_dd_circuit
from quasar.cost import Backend
from quasar.partitioner import Partitioner


def test_stim_to_dd_partition_and_conversion() -> None:
    """GHZ subsystems should trigger a tableau→DD backend switch."""

    num_groups = 3
    group_size = 3
    circuit = stim_to_dd_circuit(
        num_groups=num_groups,
        group_size=group_size,
        entangling_layer=False,
    )

    partitioner = Partitioner()
    ssd = partitioner.partition(circuit, debug=True)

    assert len(ssd.partitions) >= 2
    first_partition = ssd.partitions[0]
    assert first_partition.backend == Backend.TABLEAU

    # The tableau fragment should expose one subsystem per GHZ group.
    ghz_groups = {tuple(group) for group in first_partition.subsystems}
    assert len(ghz_groups) == num_groups
    assert all(len(group) == group_size for group in ghz_groups)

    dd_partitions = [
        part for part in ssd.partitions[1:] if part.backend == Backend.DECISION_DIAGRAM
    ]
    assert dd_partitions

    # The DD partition should inherit the independent group structure so the
    # conversion happens per subsystem.
    dd_groups = {tuple(group) for group in dd_partitions[0].subsystems}
    assert dd_groups == ghz_groups

    # Check that the partitioner recorded the backend transition and
    # conversion layer diagnostics.
    assert any(
        conv.source == Backend.TABLEAU and conv.target == Backend.DECISION_DIAGRAM
        for conv in ssd.conversions
    )
    applied_switches = [
        entry
        for entry in ssd.trace
        if entry.applied
        and entry.from_backend == Backend.TABLEAU
        and entry.to_backend == Backend.DECISION_DIAGRAM
    ]
    assert applied_switches
