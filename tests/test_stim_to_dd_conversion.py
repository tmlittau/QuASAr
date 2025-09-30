"""Tests for the stabiliser-to-decision-diagram conversion benchmark."""

from __future__ import annotations

from benchmarks.partition_circuits import stim_to_dd_circuit
from quasar.cost import Backend
from quasar.partitioner import Partitioner


def test_stim_to_dd_partition_and_conversion() -> None:
    """GHZ subsystems remain on tableau until a deferred switch is worthwhile."""

    num_groups = 3
    group_size = 3
    circuit = stim_to_dd_circuit(
        num_groups=num_groups,
        group_size=group_size,
        entangling_layer=False,
    )

    partitioner = Partitioner()
    ssd = partitioner.partition(circuit, debug=True)

    assert len(ssd.partitions) == 1
    first_partition = ssd.partitions[0]
    assert first_partition.backend == Backend.TABLEAU

    # The tableau fragment should expose one subsystem per GHZ group.
    ghz_groups = {tuple(group) for group in first_partition.subsystems}
    assert len(ghz_groups) == num_groups
    assert all(len(group) == group_size for group in ghz_groups)

    # The partitioner should evaluate tableau→DD conversions but defer them
    # because the accumulated tableau cost stays cheaper.
    assert not ssd.conversions
    deferred_trace = [
        entry
        for entry in ssd.trace
        if not entry.applied
        and entry.from_backend == Backend.TABLEAU
        and entry.to_backend == Backend.DECISION_DIAGRAM
    ]
    assert deferred_trace
