"""Circuit families highlighting parallel subsystems for benchmarking."""
from __future__ import annotations

from typing import List

from quasar.circuit import Circuit, Gate


def many_ghz_subsystems(num_groups: int, group_size: int) -> Circuit:
    """Return ``num_groups`` disjoint GHZ chains of ``group_size`` qubits each.

    The generated circuit entangles qubits only within their respective
    subsystem, making the groups independent.  This allows the planner and
    scheduler to execute the resulting partitions concurrently.

    Args:
        num_groups: Number of independent GHZ subsystems to generate.
        group_size: Number of qubits per subsystem.  Must be positive to
            create entanglement; values less than one result in an empty
            circuit.

    Returns:
        A :class:`~quasar.circuit.Circuit` containing the requested GHZ
        subsystems laid out on contiguous qubit blocks.
    """

    if num_groups <= 0 or group_size <= 0:
        return Circuit([])

    gates: List[Gate] = []
    for group in range(num_groups):
        base = group * group_size
        # Prepare the GHZ state on the group's first qubit.
        gates.append(Gate("H", [base]))
        # Chain CX gates within the group to distribute entanglement.
        for offset in range(1, group_size):
            control = base + offset - 1
            target = base + offset
            gates.append(Gate("CX", [control, target]))
    return Circuit(gates)


__all__ = ["many_ghz_subsystems"]
