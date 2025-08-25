from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List

from .cost import Backend, Cost


@dataclass(frozen=True)
class SSDPartition:
    """Represents a set of identical subsystems along with metadata.

    Each entry in :attr:`subsystems` holds the qubits belonging to one
    independent subsystem that is in the same state as the others.  The
    partition also records the execution history, the chosen simulation
    backend and an estimated cost for simulating one representative
    subsystem.
    """

    subsystems: Tuple[Tuple[int, ...], ...]
    history: Tuple[str, ...] = ()
    backend: Backend = Backend.STATEVECTOR
    cost: Cost = field(default_factory=lambda: Cost(time=0.0, memory=0.0))

    @property
    def multiplicity(self) -> int:
        """Number of identical subsystems represented by this partition."""
        return len(self.subsystems)

    @property
    def qubits(self) -> Tuple[int, ...]:
        """Flattened tuple of all qubits represented in this partition."""
        return tuple(q for group in self.subsystems for q in group)


@dataclass
class SSD:
    """Simplified storage for circuit partitions."""

    partitions: List[SSDPartition]

    def total_qubits(self) -> int:
        return sum(len(p.qubits) for p in self.partitions)
