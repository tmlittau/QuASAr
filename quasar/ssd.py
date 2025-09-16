from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Hashable, Callable

from .cost import Backend, Cost


@dataclass
class PartitionTraceEntry:
    """Diagnostics describing partition boundary decisions.

    Attributes
    ----------
    gate_index, gate_name:
        Identify the gate that triggered the decision.
    from_backend, to_backend:
        Backends involved in the potential switch.
    boundary, boundary_size:
        Sorted qubit indices along the cut and their count.
    rank, frontier:
        Estimated Schmidt rank and decision diagram frontier across the cut.
    primitive, cost:
        Conversion primitive and estimated cost when a conversion is required.
    applied:
        ``True`` when the backend change was accepted.
    reason:
        Short textual explanation for the decision.
    """

    gate_index: int
    gate_name: str
    from_backend: Backend | None
    to_backend: Backend
    boundary: Tuple[int, ...] = ()
    boundary_size: int = 0
    rank: int | None = None
    frontier: int | None = None
    primitive: str | None = None
    cost: Cost | None = None
    applied: bool = False
    reason: str = ""


@dataclass
class SSDPartition:
    """Represents a set of identical subsystems along with metadata.

    Each entry in :attr:`subsystems` holds the qubits belonging to one
    independent subsystem that is in the same state as the others.  The
    partition also records the execution history, the chosen simulation
    backend, an estimated cost for simulating one representative
    subsystem, and optionally the backend specific ``state`` object
    describing the terminal state of the subsystem.  Additional metadata
    tracks dependency edges, entanglement between partitions, compatible
    simulation methods and generic resource estimates.
    """

    subsystems: Tuple[Tuple[int, ...], ...]
    history: Tuple[str, ...] = ()
    backend: Backend = Backend.STATEVECTOR
    cost: Cost = field(default_factory=lambda: Cost(time=0.0, memory=0.0))
    state: object | None = field(default=None, repr=False, compare=False, hash=False)
    dependencies: Tuple[int, ...] = ()
    entangled_with: Tuple[int, ...] = ()
    compatible_methods: Tuple[Backend, ...] = ()
    resources: Dict[str, float] = field(default_factory=dict)
    boundary_qubits: Tuple[int, ...] = ()
    rank: int | None = None
    frontier: int | None = None
    fingerprint: Hashable | None = None

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        if self.fingerprint is None:
            self.fingerprint = (
                self.subsystems,
                self.history,
                self.backend.value,
                self.boundary_qubits,
                self.rank,
                self.frontier,
            )

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
    boundary_qubits: Tuple[int, ...] = ()
    rank: int | None = None
    frontier: int | None = None
    fingerprint: Hashable | None = None
    trace: List[PartitionTraceEntry] = field(
        default_factory=list, repr=False, compare=False
    )

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        if self.fingerprint is None:
            self.fingerprint = tuple(p.fingerprint for p in self.partitions)

    def total_qubits(self) -> int:
        return sum(len(p.qubits) for p in self.partitions)

    def by_backend(self) -> Dict[Backend, List[SSDPartition]]:
        """Group partitions by their assigned simulation backend."""
        groups: Dict[Backend, List[SSDPartition]] = {}
        for part in self.partitions:
            groups.setdefault(part.backend, []).append(part)
        return groups

    # ------------------------------------------------------------------
    def extract_state(self, partition: SSDPartition | int) -> object | None:
        """Return the backend specific state for ``partition``.

        Parameters
        ----------
        partition:
            Either a partition instance from :attr:`partitions` or an
            integer index into that list.

        Returns
        -------
        object or ``None``
            Backend dependent state representation.  For example a dense
            statevector, a list of MPS tensors, a ``stim.Tableau`` or a
            decision diagram node.  ``None`` is returned if no state was
            recorded for the given partition.
        """

        if isinstance(partition, int):
            partition = self.partitions[partition]
        return partition.state

    # ------------------------------------------------------------------
    def build_metadata(self) -> None:
        """Populate dependency, entanglement and resource metadata.

        This routine infers dependencies and entanglement annotations from
        :attr:`conversions`.  Each partition's compatible methods default to
        the assigned backend if not explicitly provided and resource
        estimates fall back to the recorded :class:`~quasar.cost.Cost`
        values.
        """

        qubit_sets: List[set[int]] = [set(p.qubits) for p in self.partitions]

        dep_sets: List[set[int]] = [set(p.dependencies) for p in self.partitions]
        ent_sets: List[set[int]] = [set(p.entangled_with) for p in self.partitions]

        for idx, part in enumerate(self.partitions):
            if not part.compatible_methods:
                part.compatible_methods = (part.backend,)
            if not part.resources:
                part.resources = {"time": part.cost.time, "memory": part.cost.memory}

        for conv in self.conversions:
            boundary = set(conv.boundary)
            involved = [i for i, qs in enumerate(qubit_sets) if qs & boundary]
            if len(involved) < 2:
                continue
            src = min(involved)
            for tgt in involved:
                if tgt == src:
                    continue
                dep_sets[tgt].add(src)
                ent_sets[src].add(tgt)
                ent_sets[tgt].add(src)

        for idx, part in enumerate(self.partitions):
            part.dependencies = tuple(sorted(dep_sets[idx]))
            part.entangled_with = tuple(sorted(ent_sets[idx]))



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


@dataclass
class SSDCache:
    """Cache for conversion and bridge results keyed by SSD fingerprints."""

    bridge_tensors: Dict[tuple, object] = field(default_factory=dict)
    conversions: Dict[tuple, object] = field(default_factory=dict)
    hits: int = 0

    def _fingerprint(self, ssd: object) -> Hashable:
        fp = getattr(ssd, "fingerprint", None)
        if fp is None:
            b = tuple(getattr(ssd, "boundary_qubits", []) or [])
            r = getattr(ssd, "rank", None)
            f = getattr(ssd, "frontier", None)
            fp = (b, r, f)
        return fp

    def bridge_tensor(self, left: object, right: object, builder: Callable[[], object]) -> object:
        key = (self._fingerprint(left), self._fingerprint(right))
        if key in self.bridge_tensors:
            self.hits += 1
            return self.bridge_tensors[key]
        self.bridge_tensors[key] = builder()
        return self.bridge_tensors[key]

    def convert(self, ssd: object, target: str, converter: Callable[[], object]) -> object:
        key = (self._fingerprint(ssd), target)
        if key in self.conversions:
            self.hits += 1
            return self.conversions[key]
        self.conversions[key] = converter()
        return self.conversions[key]


__all__ = ["SSDPartition", "SSD", "ConversionLayer", "SSDCache"]

