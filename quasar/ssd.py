from __future__ import annotations

import importlib
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from .cost import Backend, Cost, CostEstimator


if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as _nx
    from .circuit import Circuit, Gate
    from .method_selector import MethodSelector


def _require_networkx() -> "_nx":
    """Return the :mod:`networkx` module if available."""

    try:
        return importlib.import_module("networkx")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "networkx is required for SSD.to_networkx(); install it to "
            "visualise subsystem descriptors."
        ) from exc


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
class GateNode:
    """Single gate vertex in the hierarchical SSD gate graph."""

    id: int
    name: str
    params: Tuple[Tuple[str, Any], ...]
    predecessors: set[int] = field(default_factory=set)
    successors: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        """Return a human readable label for visualisation."""

        if not self.params:
            return self.name
        param_repr = ", ".join(f"{k}={v}" for k, v in self.params)
        return f"{self.name}({param_repr})"


class GateGraph:
    """Directed multigraph connecting unique gate operations."""

    def __init__(self) -> None:
        self._counter = itertools.count()
        self.nodes: Dict[int, GateNode] = {}
        self._key_index: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], GateNode] = {}
        init_node = self.add_gate("INIT", {})
        self.init_id = init_node.id

    def add_gate(self, name: str, params: Dict[str, Any]) -> GateNode:
        """Return an existing :class:`GateNode` or create a new one."""

        key = (name.upper(), tuple(sorted((k, v) for k, v in params.items())))
        node = self._key_index.get(key)
        if node is None:
            node = GateNode(id=next(self._counter), name=key[0], params=key[1])
            self._key_index[key] = node
            self.nodes[node.id] = node
        return node

    def add_transition(self, src_id: int, dst_id: int) -> None:
        """Register a transition between two gate nodes."""

        if src_id == dst_id:
            return
        src = self.nodes[src_id]
        dst = self.nodes[dst_id]
        src.successors.add(dst_id)
        dst.predecessors.add(src_id)


@dataclass
class GatePathNode:
    """Node describing a unique gate path executed by a subsystem."""

    id: int
    num_qubits: int
    operations: Tuple[Tuple[int, Tuple[int, ...]], ...]
    predecessors: set[int] = field(default_factory=set)
    successors: set[int] = field(default_factory=set)
    backend: Backend | None = None
    cost: Cost | None = None
    history: Tuple[str, ...] = ()
    last_gate_id: int = 0

    @property
    def is_root(self) -> bool:
        return not self.operations


class GatePathGraph:
    """Graph describing dependencies between gate paths."""

    def __init__(self, init_gate_id: int) -> None:
        self._counter = itertools.count()
        self._key_index: Dict[Tuple[int, Tuple[Tuple[int, Tuple[int, ...]], ...]], GatePathNode] = {}
        self.nodes: Dict[int, GatePathNode] = {}
        self.root = self._create_node(0, (), predecessors=(), last_gate_id=init_gate_id)

    def _create_node(
        self,
        num_qubits: int,
        operations: Tuple[Tuple[int, Tuple[int, ...]], ...],
        *,
        predecessors: Iterable[int],
        last_gate_id: int,
    ) -> GatePathNode:
        node = GatePathNode(
            id=next(self._counter),
            num_qubits=num_qubits,
            operations=operations,
            predecessors=set(predecessors),
            last_gate_id=last_gate_id,
        )
        self.nodes[node.id] = node
        key = (num_qubits, operations)
        self._key_index[key] = node
        for pred in predecessors:
            self.nodes[pred].successors.add(node.id)
        return node

    def get_or_create(
        self,
        num_qubits: int,
        operations: Tuple[Tuple[int, Tuple[int, ...]], ...],
        *,
        predecessors: Iterable[int],
        last_gate_id: int,
    ) -> GatePathNode:
        key = (num_qubits, operations)
        node = self._key_index.get(key)
        if node is None:
            node = self._create_node(num_qubits, operations, predecessors=predecessors, last_gate_id=last_gate_id)
        else:
            node.predecessors.update(predecessors)
            for pred in predecessors:
                self.nodes[pred].successors.add(node.id)
        return node

    def extend(
        self,
        prev: GatePathNode,
        gate_id: int,
        positions: Tuple[int, ...],
        *,
        num_qubits: int,
    ) -> GatePathNode:
        operations = prev.operations + ((gate_id, positions),)
        return self.get_or_create(
            num_qubits,
            operations,
            predecessors=(prev.id,),
            last_gate_id=gate_id,
        )

    def merge(
        self,
        predecessors: Sequence[GatePathNode],
        gate_id: int,
        positions: Tuple[int, ...],
        *,
        num_qubits: int,
    ) -> GatePathNode:
        pred_ids = tuple(node.id for node in predecessors)
        return self.get_or_create(
            num_qubits,
            ((gate_id, positions),),
            predecessors=pred_ids,
            last_gate_id=gate_id,
        )

    def to_networkx(self, gate_graph: GateGraph) -> "_nx.DiGraph":
        nx = _require_networkx()
        graph = nx.DiGraph()
        for node in self.nodes.values():
            label = tuple(gate_graph.nodes[gid].name for gid, _ in node.operations)
            backend_name = node.backend.name if node.backend is not None else None
            cost_time = node.cost.time if node.cost is not None else None
            graph.add_node(
                ("path", node.id),
                history=label,
                backend=backend_name,
                cost_time=cost_time,
                num_qubits=node.num_qubits,
            )
        for node in self.nodes.values():
            for succ in node.successors:
                graph.add_edge(("path", node.id), ("path", succ))
        return graph


@dataclass
class SubsystemNode:
    """Vertex in the qubit map layer representing a subsystem of qubits."""

    id: int
    qubits: Tuple[int, ...]
    path: int
    predecessors: set[int] = field(default_factory=set)
    successors: set[int] = field(default_factory=set)


class SubsystemGraph:
    """Graph tracking how independent qubit sets merge due to entanglement."""

    def __init__(self) -> None:
        self.nodes: List[SubsystemNode] = []

    def add_node(self, qubits: Iterable[int], path: int, predecessors: Iterable[int] = ()) -> SubsystemNode:
        node_id = len(self.nodes)
        node = SubsystemNode(id=node_id, qubits=tuple(sorted(qubits)), path=path, predecessors=set(predecessors))
        self.nodes.append(node)
        for pred in predecessors:
            self.nodes[pred].successors.add(node_id)
        return node

    def update_path(self, node_id: int, path: int) -> None:
        self.nodes[node_id].path = path


@dataclass
class SSDHierarchy:
    """Container bundling the three-layer hierarchical SSD representation."""

    gate_graph: GateGraph
    path_graph: GatePathGraph
    subsystem_graph: SubsystemGraph

    def path_networkx(self) -> "_nx.DiGraph":
        """Return a :mod:`networkx` representation of the gate path layer."""

        return self.path_graph.to_networkx(self.gate_graph)


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
    hierarchy: SSDHierarchy | None = None
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
    def to_path_networkx(self) -> "_nx.DiGraph":
        """Return the middle gate-path layer as a :mod:`networkx` graph."""

        if self.hierarchy is None:
            raise RuntimeError("SSD hierarchy is not populated; build the descriptor first.")
        return self.hierarchy.path_networkx()

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

    # ------------------------------------------------------------------
    def to_networkx(
        self,
        *,
        include_dependencies: bool = True,
        include_entanglement: bool = True,
        include_conversions: bool = True,
        include_backends: bool = True,
    ) -> "_nx.MultiDiGraph":
        """Return a :class:`networkx.MultiDiGraph` representing the SSD.

        Parameters
        ----------
        include_dependencies:
            When ``True`` include directed edges that encode execution
            dependencies between partitions.
        include_entanglement:
            When ``True`` include undirected edges (recorded as a single
            directed edge with the ``bidirectional`` flag) between partitions
            that share entanglement.
        include_conversions:
            When ``True`` add conversion layer nodes and connect them to the
            partitions and backends they touch.
        include_backends:
            When ``True`` add a node for every backend referenced by the SSD
            and connect partitions and conversions to their respective
            backends.

        Returns
        -------
        networkx.MultiDiGraph
            Graph describing partitions, conversions and (optionally) backend
            assignments.  Nodes carry metadata mirroring the SSD entries while
            edges annotate dependencies, entanglement and conversion
            boundaries.

        Raises
        ------
        RuntimeError
            If :mod:`networkx` is not installed in the current environment.
        """

        nx = _require_networkx()

        # Ensure dependency and entanglement metadata is populated before
        # constructing the graph.
        self.build_metadata()

        graph: "_nx.MultiDiGraph" = nx.MultiDiGraph()
        graph.graph.update(
            type="ssd",
            boundary_qubits=self.boundary_qubits,
            rank=self.rank,
            frontier=self.frontier,
            fingerprint=self.fingerprint,
            num_partitions=len(self.partitions),
            total_qubits=self.total_qubits(),
        )
        if self.trace:
            graph.graph["trace"] = list(self.trace)

        backend_nodes: Dict[str, tuple[str, str]] = {}

        def ensure_backend_node(backend: Backend) -> tuple[str, str]:
            if not include_backends:
                return ("backend", backend.name)
            key = backend.name
            node = backend_nodes.get(key)
            if node is None:
                node = ("backend", key)
                backend_nodes[key] = node
                graph.add_node(
                    node,
                    kind="backend",
                    backend=backend.name,
                    label=backend.value,
                )
            return node

        qubit_to_partition: Dict[int, int] = {}

        for idx, part in enumerate(self.partitions):
            node = ("partition", idx)
            compatible = tuple(
                method.name if isinstance(method, Backend) else str(method)
                for method in part.compatible_methods
            )
            graph.add_node(
                node,
                kind="partition",
                index=idx,
                backend=part.backend.name,
                subsystems=part.subsystems,
                multiplicity=part.multiplicity,
                qubits=part.qubits,
                history=part.history,
                boundary_qubits=part.boundary_qubits,
                rank=part.rank,
                frontier=part.frontier,
                compatible_methods=compatible,
                resources=dict(part.resources),
                cost_time=part.cost.time,
                cost_memory=part.cost.memory,
            )
            if include_backends:
                backend_node = ensure_backend_node(part.backend)
                graph.add_edge(node, backend_node, kind="backend_assignment")
            for qubit in part.qubits:
                qubit_to_partition[qubit] = idx

        if include_dependencies:
            for idx, part in enumerate(self.partitions):
                for dep in part.dependencies:
                    graph.add_edge(
                        ("partition", dep),
                        ("partition", idx),
                        kind="dependency",
                    )

        if include_entanglement:
            for idx, part in enumerate(self.partitions):
                for partner in part.entangled_with:
                    if idx < partner:
                        graph.add_edge(
                            ("partition", idx),
                            ("partition", partner),
                            kind="entanglement",
                            bidirectional=True,
                        )

        if include_conversions:
            for conv_idx, conv in enumerate(self.conversions):
                node = ("conversion", conv_idx)
                graph.add_node(
                    node,
                    kind="conversion",
                    index=conv_idx,
                    boundary=conv.boundary,
                    source=conv.source.name,
                    target=conv.target.name,
                    rank=conv.rank,
                    frontier=conv.frontier,
                    primitive=conv.primitive,
                    cost_time=conv.cost.time,
                    cost_memory=conv.cost.memory,
                )
                if include_backends:
                    src_backend = ensure_backend_node(conv.source)
                    tgt_backend = ensure_backend_node(conv.target)
                    graph.add_edge(
                        src_backend,
                        node,
                        kind="conversion_source",
                    )
                    graph.add_edge(
                        node,
                        tgt_backend,
                        kind="conversion_target",
                    )
                involved = sorted(
                    {
                        qubit_to_partition[qubit]
                        for qubit in conv.boundary
                        if qubit in qubit_to_partition
                    }
                )
                for part_idx in involved:
                    graph.add_edge(
                        ("partition", part_idx),
                        node,
                        kind="conversion_boundary",
                    )

        return graph


@dataclass
class _SubsystemState:
    """Internal helper tracking the active subsystem during construction."""

    node_id: int
    qubits: Tuple[int, ...]
    path_node: GatePathNode
    local_index: Dict[int, int]


class _HierarchyBuilder:
    """Construct the hierarchical SSD representation for a circuit."""

    def __init__(
        self,
        circuit: "Circuit",
        selector: "MethodSelector",
        *,
        max_memory: float | None = None,
        max_time: float | None = None,
        target_accuracy: float | None = None,
    ) -> None:
        self.circuit = circuit
        self.selector = selector
        self.max_memory = max_memory
        self.max_time = max_time
        self.target_accuracy = target_accuracy
        self.gate_graph = GateGraph()
        self.path_graph = GatePathGraph(self.gate_graph.init_id)
        self.subsystem_graph = SubsystemGraph()
        self._active: Dict[int, _SubsystemState] = {}

    # ------------------------------------------------------------------
    def build(self) -> SSD:
        if not self.circuit.gates:
            hierarchy = SSDHierarchy(self.gate_graph, self.path_graph, self.subsystem_graph)
            return SSD([], hierarchy=hierarchy)

        init_path = self.path_graph.root
        all_qubits = sorted({q for gate in self.circuit.gates for q in gate.qubits})
        for qubit in all_qubits:
            node = self.subsystem_graph.add_node((qubit,), init_path.id)
            state = _SubsystemState(
                node_id=node.id,
                qubits=(qubit,),
                path_node=init_path,
                local_index={qubit: 0},
            )
            self._active[qubit] = state

        for gate in self.circuit.gates:
            self._apply_gate(gate)

        self._assign_methods()
        return self._to_ssd()

    # ------------------------------------------------------------------
    def _apply_gate(self, gate: "Gate") -> None:
        gate_node = self.gate_graph.add_gate(gate.gate, gate.params)

        involved_states: List[_SubsystemState] = []
        seen: set[int] = set()
        for qubit in gate.qubits:
            state = self._active[qubit]
            if state.node_id not in seen:
                seen.add(state.node_id)
                involved_states.append(state)
            last_gate = state.path_node.last_gate_id if state.path_node.operations else self.gate_graph.init_id
            self.gate_graph.add_transition(last_gate, gate_node.id)

        if not involved_states:
            return

        if len(involved_states) == 1:
            state = involved_states[0]
            positions = tuple(state.local_index[q] for q in gate.qubits)
            new_path = self.path_graph.extend(
                state.path_node,
                gate_node.id,
                positions,
                num_qubits=len(state.qubits),
            )
            state.path_node = new_path
            self.subsystem_graph.update_path(state.node_id, new_path.id)
            return

        merged_qubits = sorted({q for state in involved_states for q in state.qubits})
        local_map = {q: i for i, q in enumerate(merged_qubits)}
        positions = tuple(local_map[q] for q in gate.qubits)
        new_path = self.path_graph.merge(
            [state.path_node for state in involved_states],
            gate_node.id,
            positions,
            num_qubits=len(merged_qubits),
        )
        new_node = self.subsystem_graph.add_node(
            merged_qubits,
            new_path.id,
            predecessors=[state.node_id for state in involved_states],
        )
        new_state = _SubsystemState(
            node_id=new_node.id,
            qubits=tuple(merged_qubits),
            path_node=new_path,
            local_index=local_map,
        )
        for qubit in merged_qubits:
            self._active[qubit] = new_state

    # ------------------------------------------------------------------
    def _assign_methods(self) -> None:
        if not self.path_graph.nodes:
            return

        from .circuit import Gate

        for node in self.path_graph.nodes.values():
            if node.is_root:
                continue
            gates: List[Gate] = []
            for gate_id, positions in node.operations:
                gate_node = self.gate_graph.nodes[gate_id]
                gates.append(
                    Gate(
                        gate_node.name,
                        list(positions),
                        dict(gate_node.params),
                    )
                )
            backend, cost = self.selector.select(
                gates,
                node.num_qubits or max((max(p) for _, p in node.operations), default=-1) + 1,
                max_memory=self.max_memory,
                max_time=self.max_time,
                target_accuracy=self.target_accuracy,
            )
            node.backend = backend
            node.cost = cost
            node.history = tuple(g.gate for g in gates)

    # ------------------------------------------------------------------
    def _to_ssd(self) -> SSD:
        partitions: List[SSDPartition] = []
        path_nodes = [node for node in self.path_graph.nodes.values() if not node.is_root]
        path_nodes.sort(key=lambda n: n.id)
        id_to_index = {node.id: idx for idx, node in enumerate(path_nodes)}

        path_to_qubits: Dict[int, List[Tuple[int, ...]]] = {}
        for subsystem in self.subsystem_graph.nodes:
            path_id = subsystem.path
            if path_id == self.path_graph.root.id:
                continue
            path_to_qubits.setdefault(path_id, []).append(subsystem.qubits)

        for node in path_nodes:
            qubit_groups = tuple(path_to_qubits.get(node.id, []))
            if not qubit_groups:
                qubit_groups = (tuple(range(node.num_qubits)),) if node.num_qubits else ((),)
            dependencies = tuple(
                sorted(id_to_index[p] for p in node.predecessors if p in id_to_index)
            )
            backend = node.backend or Backend.STATEVECTOR
            cost = node.cost or Cost(time=0.0, memory=0.0)
            partition = SSDPartition(
                subsystems=qubit_groups,
                history=node.history,
                backend=backend,
                cost=cost,
                dependencies=dependencies,
            )
            entangled = {dep for dep in dependencies}
            partition.entangled_with = tuple(sorted(entangled))
            partitions.append(partition)

        hierarchy = SSDHierarchy(self.gate_graph, self.path_graph, self.subsystem_graph)
        ssd = SSD(partitions=partitions, conversions=[], hierarchy=hierarchy)
        ssd.build_metadata()
        return ssd


def build_hierarchical_ssd(
    circuit: "Circuit",
    *,
    estimator: CostEstimator | None = None,
    selector: "MethodSelector" | None = None,
    max_memory: float | None = None,
    max_time: float | None = None,
    target_accuracy: float | None = None,
) -> SSD:
    """Construct the hierarchical SSD for ``circuit`` and assign methods."""

    if estimator is None:
        estimator = CostEstimator()
    if selector is None:
        from .method_selector import MethodSelector

        selector = MethodSelector(estimator)
    builder = _HierarchyBuilder(
        circuit,
        selector,
        max_memory=max_memory,
        max_time=max_time,
        target_accuracy=target_accuracy,
    )
    return builder.build()


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


__all__ = [
    "PartitionTraceEntry",
    "SSDPartition",
    "SSD",
    "ConversionLayer",
    "SSDCache",
    "build_hierarchical_ssd",
]

