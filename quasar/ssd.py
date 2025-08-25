from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Dict

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
    """Simplified storage for circuit partitions and conversions."""

    partitions: List[SSDPartition]
    conversions: List["ConversionLayer"] = field(default_factory=list)

    def total_qubits(self) -> int:
        return sum(len(p.qubits) for p in self.partitions)

    def by_backend(self) -> Dict[Backend, List[SSDPartition]]:
        """Group partitions by their assigned simulation backend."""
        groups: Dict[Backend, List[SSDPartition]] = {}
        for part in self.partitions:
            groups.setdefault(part.backend, []).append(part)
        return groups


@dataclass(frozen=True)
class ConversionLayer:
    """Represents a conversion between two partition backends.

    Parameters
    ----------
    boundary:
        Qubits that lie on the boundary between the two partitions.
    source, target:
        Backends used before and after the conversion.
    rank:
        Estimated Schmidt rank across the cut.
    frontier:
        Decision diagram frontier size used for estimating conversion
        costs.
    primitive:
        Conversion primitive selected by the estimator (``B2B``, ``LW``,
        ``ST`` or ``Full``).
    cost:
        Estimated cost of performing the conversion.
    """

    boundary: Tuple[int, ...]
    source: Backend
    target: Backend
    rank: int
    frontier: int
    primitive: str
    cost: Cost


__all__ = ["SSDPartition", "SSD", "ConversionLayer"]

