from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING, Set

from .ssd import SSD, SSDPartition, ConversionLayer, PartitionTraceEntry
from .cost import Backend, CostEstimator, Cost
from .method_selector import MethodSelector, NoFeasibleBackendError
from .metrics import FragmentMetrics
from . import config

if TYPE_CHECKING:  # pragma: no cover
    from .circuit import Circuit, Gate


CLIFFORD_GATES = {
    "I",
    "ID",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "SDG",
    "CX",
    "CY",
    "CZ",
    "SWAP",
}

CLIFFORD_PLUS_T_GATES = CLIFFORD_GATES | {"T", "TDG"}


class Partitioner:
    """Partition circuits and assign simulation methods."""

    def __init__(
        self,
        estimator: CostEstimator | None = None,
        selector: MethodSelector | None = None,
        *,
        max_memory: float | None = None,
        max_time: float | None = None,
        target_accuracy: float | None = None,
        staging_chi_cap: int | None = config.DEFAULT.st_chi_cap,
        graph_cut_candidate_limit: int | None = config.DEFAULT.graph_cut_candidate_limit,
        graph_cut_neighbor_radius: int = config.DEFAULT.graph_cut_neighbor_radius,
        graph_cut_boundary_weight: float = config.DEFAULT.graph_cut_boundary_weight,
        graph_cut_rank_weight: float = config.DEFAULT.graph_cut_rank_weight,
        graph_cut_cost_weight: float = config.DEFAULT.graph_cut_cost_weight,
    ):
        self.estimator = estimator or CostEstimator()
        self.selector = selector or MethodSelector(self.estimator)
        self.max_memory = max_memory
        self.max_time = max_time
        self.target_accuracy = target_accuracy
        self.staging_chi_cap = None
        if staging_chi_cap is not None:
            cap = max(1, int(staging_chi_cap))
            self.staging_chi_cap = cap
            self.estimator.coeff["st_chi_cap"] = float(cap)
        self.graph_cut_candidate_limit = (
            None
            if graph_cut_candidate_limit is None
            else max(1, int(graph_cut_candidate_limit))
        )
        self.graph_cut_neighbor_radius = max(0, int(graph_cut_neighbor_radius))
        self.graph_cut_boundary_weight = float(graph_cut_boundary_weight)
        self.graph_cut_rank_weight = float(graph_cut_rank_weight)
        self.graph_cut_cost_weight = float(graph_cut_cost_weight)

    def partition(
        self,
        circuit: 'Circuit',
        *,
        graph_cut: bool = False,
        debug: bool = False,
        trace: Callable[[PartitionTraceEntry], None] | None = None,
    ) -> SSD:
        """Partition ``circuit`` into simulation segments.

        Parameters
        ----------
        circuit:
            Circuit to partition.
        graph_cut:
            When ``True`` evaluate multiple partition candidates using a
            graph-based heuristic that balances load and minimises conversion
            boundaries.  The default ``False`` uses the original sequential
            heuristic.
        debug:
            When ``True`` return an :class:`~quasar.ssd.SSD` populated with a
            ``trace`` describing every evaluated backend switch.
        trace:
            Optional callback invoked with a
            :class:`~quasar.ssd.PartitionTraceEntry` for every potential
            partition cut.  The returned :class:`~quasar.ssd.SSD` exposes the
            collected entries via :attr:`~quasar.ssd.SSD.trace` when either
            :paramref:`debug` is ``True`` or :paramref:`trace` is provided.

        Raises
        ------
        NoFeasibleBackendError
            If no simulation backend satisfies the resource constraints for a
            fragment.
        """

        trace_log: List[PartitionTraceEntry] | None = [] if (debug or trace is not None) else None

        if not circuit.gates:
            return SSD([], trace=trace_log if trace_log is not None else [])

        gates = circuit.gates

        # Pre-compute for each gate index the set of qubits that appear in
        # the remainder of the circuit. This lets us derive boundary sizes for
        # conversion layers without repeatedly scanning the gate list.
        future_qubits: List[Set[int]] = [set() for _ in range(len(gates) + 1)]
        running: Set[int] = set()
        for idx in range(len(gates) - 1, -1, -1):
            running |= set(gates[idx].qubits)
            future_qubits[idx] = running.copy()

        partitions: List[SSDPartition] = []
        conversions: List[ConversionLayer] = []

        current_gates: List['Gate'] = []
        current_qubits: Set[int] = set()
        current_backend: Backend | None = None
        current_cost: Cost | None = None
        current_metrics: FragmentMetrics | None = None

        @dataclass
        class _PendingSwitch:
            start_index: int
            gate_index: int
            source_backend: Backend
            source_cost: Cost
            boundary: Set[int]
            target_backend: Backend
            source_metrics: FragmentMetrics | None = None
            boundary_tuple: Tuple[int, ...] = ()
            rank: int | None = None
            frontier: int | None = None
            primitive: str | None = None
            conversion_cost: Cost | None = None
            target_cost: Cost | None = None
            window: int | None = None

        pending_switch: _PendingSwitch | None = None

        gate_indices = {id(gate): idx for idx, gate in enumerate(gates)}
        entanglement_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[int, int]] = {}

        class _StubGate:
            __slots__ = ("qubits",)

            def __init__(self, qubits: Tuple[int, ...]):
                self.qubits = qubits

        def _fragment_key(metrics: FragmentMetrics) -> Tuple[int, ...]:
            if not metrics.gates:
                return ()
            return tuple(
                gate_indices[id(g)]
                for g in metrics.gates
                if id(g) in gate_indices
            )

        def _entanglement_bounds(
            metrics: FragmentMetrics | None,
            boundary_tuple: Tuple[int, ...],
        ) -> Tuple[int, int]:
            if metrics is None or not boundary_tuple:
                return 1, 0
            boundary_size = len(boundary_tuple)
            if not metrics.gates:
                return 1, 0
            key = (boundary_tuple, _fragment_key(metrics))
            cached = entanglement_cache.get(key)
            if cached is not None:
                return cached

            boundary_set = set(boundary_tuple)
            fragment_qubits: List[int] = []
            for gate in metrics.gates:
                for qubit in gate.qubits:
                    if qubit not in fragment_qubits:
                        fragment_qubits.append(qubit)

            ordering: List[int] = list(boundary_tuple)
            ordering.extend(q for q in fragment_qubits if q not in boundary_set)

            if len(ordering) <= boundary_size:
                entanglement_cache[key] = (1, 0)
                return entanglement_cache[key]

            remap = {qubit: idx for idx, qubit in enumerate(ordering)}
            remapped_gates: List[_StubGate] = []
            for gate in metrics.gates:
                mapped = tuple(remap[q] for q in gate.qubits if q in remap)
                if len(mapped) >= 2:
                    remapped_gates.append(_StubGate(mapped))

            if not remapped_gates:
                entanglement_cache[key] = (1, 0)
                return entanglement_cache[key]

            bonds = self.estimator.bond_dimensions(len(ordering), remapped_gates)
            idx = boundary_size - 1
            if idx < 0 or idx >= len(bonds):
                rank_est = 1
            else:
                rank_est = bonds[idx]
            rank_est = max(1, min(rank_est, 2 ** boundary_size))
            if rank_est <= 1:
                frontier_est = 0
            else:
                frontier_est = min(
                    boundary_size,
                    max(1, int(math.ceil(math.log2(rank_est)))),
                )
            entanglement_cache[key] = (rank_est, frontier_est)
            return entanglement_cache[key]

        def _conversion_diagnostics(
            source: Backend | None,
            target: Backend,
            boundary: Set[int],
            *,
            metrics: FragmentMetrics | None = None,
        ) -> Tuple[
            Tuple[int, ...],
            int,
            int | None,
            int | None,
            str | None,
            Cost | None,
            int | None,
        ]:
            boundary_tuple = tuple(sorted(boundary))
            size = len(boundary_tuple)
            if size == 0 or source is None or source == target:
                rank = 1 if size == 0 else 2 ** size
                frontier = 0 if rank == 1 else size
                return boundary_tuple, size, rank, frontier, None, None, None

            dense_terms = 1 << size if size else 1
            rank = dense_terms
            frontier = size

            if metrics is not None and size > 0:
                bound_rank, bound_frontier = _entanglement_bounds(metrics, boundary_tuple)
                rank = max(1, min(rank, bound_rank))
                frontier = min(frontier, bound_frontier if bound_frontier is not None else frontier)
            elif rank == 1:
                frontier = 0

            compressed_terms = max(1, min(dense_terms, rank))
            bond_dimension = rank if size > 0 else None
            window = self.estimator.derive_conversion_window(
                size,
                rank=rank,
                compressed_terms=compressed_terms,
                bond_dimension=bond_dimension,
            )
            conv_est = self.estimator.conversion(
                source,
                target,
                num_qubits=size,
                rank=rank,
                frontier=frontier,
                window=window,
                compressed_terms=compressed_terms,
                bond_dimension=bond_dimension,
                chi_cap=self.staging_chi_cap,
            )
            return (
                boundary_tuple,
                size,
                rank,
                frontier,
                conv_est.primitive,
                conv_est.cost,
                conv_est.window,
            )

        def _gate_statistics(
            gate_seq: List['Gate'],
        ) -> Tuple[Set[int], int, int, int, int, int]:
            qubits = {q for g in gate_seq for q in g.qubits}
            num_gates = len(gate_seq)
            num_meas = sum(
                1 for g in gate_seq if g.gate.upper() in {"MEASURE", "RESET"}
            )
            num_1q = sum(
                1
                for g in gate_seq
                if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
            )
            num_2q = sum(1 for g in gate_seq if len(g.qubits) > 1)
            num_t = sum(1 for g in gate_seq if g.gate.upper() in {"T", "TDG"})
            return qubits, num_gates, num_meas, num_1q, num_2q, num_t

        def _estimate_cost(
            backend: Backend,
            gate_seq: List['Gate'],
        ) -> Cost:
            if not gate_seq:
                return Cost(0.0, 0.0)
            (
                qubits,
                num_gates,
                num_meas,
                num_1q,
                num_2q,
                num_t,
            ) = _gate_statistics(gate_seq)
            num_qubits = len(qubits)
            if backend == Backend.TABLEAU:
                return self.estimator.tableau(num_qubits, num_gates, num_meas=num_meas)
            if backend == Backend.EXTENDED_STABILIZER:
                num_clifford = max(0, num_gates - num_t - num_meas)
                return self.estimator.extended_stabilizer(
                    num_qubits,
                    num_clifford,
                    num_t,
                    num_meas=num_meas,
                    depth=num_gates,
                )
            if backend == Backend.MPS:
                return self.estimator.mps(
                    num_qubits,
                    num_1q + num_meas,
                    num_2q,
                    chi=4,
                    svd=True,
                )
            if backend == Backend.DECISION_DIAGRAM:
                return self.estimator.decision_diagram(
                    num_gates=num_gates,
                    frontier=num_qubits,
                )
            return self.estimator.statevector(
                num_qubits,
                num_1q,
                num_2q,
                num_meas,
            )

        def _combine_costs(*costs: Cost) -> Cost:
            valid = [c for c in costs if c is not None]
            if not valid:
                return Cost(0.0, 0.0)
            time = sum(c.time for c in valid)
            memory = max(c.memory for c in valid)
            log_depth = max(c.log_depth for c in valid)
            conversion = sum(c.conversion for c in valid)
            replay = sum(c.replay for c in valid)
            return Cost(time=time, memory=memory, log_depth=log_depth, conversion=conversion, replay=replay)

        def _cost_key(cost: Cost) -> Tuple[float, float]:
            return cost.memory, cost.time

        def _maybe_finalize_pending_switch() -> None:
            nonlocal pending_switch, current_gates, current_qubits, current_backend, current_cost, current_metrics
            if pending_switch is None:
                return
            suffix = current_gates[pending_switch.start_index :]
            if not suffix:
                pending_switch = None
                return
            suffix_cost = _estimate_cost(pending_switch.target_backend, suffix)
            source_metrics = pending_switch.source_metrics
            if source_metrics is None and pending_switch.start_index:
                source_metrics = FragmentMetrics.from_gates(
                    current_gates[: pending_switch.start_index]
                )
                pending_switch.source_metrics = source_metrics
            if pending_switch.boundary_tuple == () and pending_switch.boundary:
                (
                    boundary_tuple,
                    boundary_size,
                    rank,
                    frontier,
                    primitive,
                    conv_cost,
                    window,
                ) = _conversion_diagnostics(
                    pending_switch.source_backend,
                    pending_switch.target_backend,
                    pending_switch.boundary,
                    metrics=source_metrics,
                )
                pending_switch.boundary_tuple = boundary_tuple
                pending_switch.rank = rank
                pending_switch.frontier = frontier
                pending_switch.primitive = primitive
                pending_switch.conversion_cost = conv_cost
                pending_switch.window = window
                if boundary_size == 0:
                    pending_switch.boundary_tuple = ()
            if pending_switch.boundary and (
                pending_switch.conversion_cost is None
                or pending_switch.primitive is None
            ):
                pending_switch = None
                return
            total_current = _estimate_cost(pending_switch.source_backend, current_gates)
            current_cost = total_current
            if pending_switch.boundary and pending_switch.conversion_cost is None:
                return
            prefix_cost = pending_switch.source_cost
            conversion_cost = pending_switch.conversion_cost
            combined = _combine_costs(
                prefix_cost,
                conversion_cost if conversion_cost is not None else Cost(0.0, 0.0),
                suffix_cost,
            )
            if _cost_key(combined) < _cost_key(total_current):
                prefix = current_gates[: pending_switch.start_index]
                if prefix:
                    partitions.extend(
                        self._build_partitions(
                            prefix,
                            pending_switch.source_backend,
                            prefix_cost,
                        )
                    )
                if pending_switch.boundary_tuple and conversion_cost is not None and pending_switch.primitive:
                    conversions.append(
                        ConversionLayer(
                            boundary=pending_switch.boundary_tuple,
                            source=pending_switch.source_backend,
                            target=pending_switch.target_backend,
                            rank=pending_switch.rank,
                            frontier=pending_switch.frontier,
                            primitive=pending_switch.primitive,
                            cost=conversion_cost,
                            window=pending_switch.window,
                        )
                    )
                _emit_trace(
                    gate_index=pending_switch.gate_index,
                    gate_name=current_gates[pending_switch.start_index].gate,
                    source=pending_switch.source_backend,
                    target=pending_switch.target_backend,
                    boundary=pending_switch.boundary,
                    applied=True,
                    reason="deferred_backend_switch",
                    metrics_override=source_metrics,
                )
                current_gates = suffix.copy()
                current_metrics = FragmentMetrics.from_gates(current_gates)
                current_qubits = {q for g in current_gates for q in g.qubits}
                current_backend = pending_switch.target_backend
                current_cost = suffix_cost
                pending_switch = None
            else:
                pending_switch.target_cost = suffix_cost

        def _emit_trace(
            *,
            gate_index: int,
            gate_name: str,
            source: Backend | None,
            target: Backend,
            boundary: Set[int],
            applied: bool,
            reason: str,
            metrics_override: FragmentMetrics | None = None,
        ) -> None:
            if trace_log is None and trace is None:
                return
            (
                boundary_tuple,
                boundary_size,
                rank,
                frontier,
                primitive,
                conv_cost,
                window,
            ) = _conversion_diagnostics(
                source,
                target,
                boundary,
                metrics=metrics_override if metrics_override is not None else current_metrics,
            )
            entry = PartitionTraceEntry(
                gate_index=gate_index,
                gate_name=gate_name,
                from_backend=source,
                to_backend=target,
                boundary=boundary_tuple,
                boundary_size=boundary_size,
                rank=rank,
                frontier=frontier,
                window=window,
                primitive=primitive,
                cost=conv_cost,
                applied=applied,
                reason=reason,
            )
            if trace is not None:
                trace(entry)
            if trace_log is not None:
                trace_log.append(entry)

        for idx, gate in enumerate(gates):
            trial_gates = current_gates + [gate]
            trial_qubits = current_qubits | set(gate.qubits)
            if current_metrics is None:
                trial_metrics = FragmentMetrics()
            else:
                trial_metrics = current_metrics.copy()
            trial_metrics.update(gate)
            s_est = trial_metrics.sparsity
            pr_div = trial_metrics.phase_rotation_diversity
            ar_div = trial_metrics.amplitude_rotation_diversity
            backend_trial, _ = self.selector.select(
                trial_gates,
                len(trial_qubits),
                sparsity=s_est,
                phase_rotation_diversity=pr_div,
                amplitude_rotation_diversity=ar_div,
                max_memory=self.max_memory,
                max_time=self.max_time,
                target_accuracy=self.target_accuracy,
            )

            # If we've already committed to a statevector simulation, keep it
            # for the remainder of the fragment to avoid flip-flopping to less
            # expressive backends based on early gates.
            if current_backend == Backend.STATEVECTOR:
                if backend_trial != current_backend:
                    boundary = current_qubits & future_qubits[idx]
                    _emit_trace(
                        gate_index=idx,
                        gate_name=gate.gate,
                        source=current_backend,
                        target=backend_trial,
                        boundary=boundary,
                        applied=False,
                        reason="statevector_lock",
                    )
                current_gates = trial_gates
                current_qubits = trial_qubits
                current_metrics = trial_metrics
                current_cost = _estimate_cost(current_backend, current_gates)
                pending_switch = None
                _maybe_finalize_pending_switch()
                continue

            if current_backend is None:
                current_gates = trial_gates
                current_qubits = trial_qubits
                current_backend = backend_trial
                current_metrics = trial_metrics
                current_cost = _estimate_cost(current_backend, current_gates)
                continue

            if backend_trial != current_backend:
                if graph_cut and current_gates:
                    cut_idx, boundary = self._select_cut_point(
                        current_gates, gate, future_qubits[idx]
                    )
                    prefix = current_gates[:cut_idx]
                    suffix = current_gates[cut_idx:]

                    if prefix:
                        p_qubits = {
                            q for g in prefix for q in g.qubits
                        }
                        p_metrics = FragmentMetrics.from_gates(prefix)
                        ps = p_metrics.sparsity
                        pr = p_metrics.phase_rotation_diversity
                        ar = p_metrics.amplitude_rotation_diversity
                        p_backend, p_cost = self.selector.select(
                            prefix,
                            len(p_qubits),
                            sparsity=ps,
                            phase_rotation_diversity=pr,
                            amplitude_rotation_diversity=ar,
                            max_memory=self.max_memory,
                            max_time=self.max_time,
                            target_accuracy=self.target_accuracy,
                        )
                        partitions.extend(
                            self._build_partitions(prefix, p_backend, p_cost)
                        )
                    else:
                        p_backend = current_backend

                    s_gates = suffix + [gate]
                    s_qubits = {
                        q for g in s_gates for q in g.qubits
                    }
                    s_metrics = FragmentMetrics.from_gates(s_gates)
                    ss = s_metrics.sparsity
                    sr = s_metrics.phase_rotation_diversity
                    ar = s_metrics.amplitude_rotation_diversity
                    s_backend, s_cost = self.selector.select(
                        s_gates,
                        len(s_qubits),
                        sparsity=ss,
                        phase_rotation_diversity=sr,
                        amplitude_rotation_diversity=ar,
                        max_memory=self.max_memory,
                        max_time=self.max_time,
                        target_accuracy=self.target_accuracy,
                    )
                    (
                        boundary_tuple,
                        boundary_size,
                        rank,
                        frontier,
                        primitive,
                        conv_cost,
                        window,
                    ) = _conversion_diagnostics(
                        p_backend, s_backend, boundary, metrics=p_metrics
                    )
                    if (
                        boundary_size
                        and primitive is not None
                        and conv_cost is not None
                    ):
                        conversions.append(
                            ConversionLayer(
                                boundary=boundary_tuple,
                                source=p_backend,
                                target=s_backend,
                                rank=rank,
                                frontier=frontier,
                                primitive=primitive,
                                cost=conv_cost,
                                window=window,
                            )
                        )
                    _emit_trace(
                        gate_index=idx,
                        gate_name=gate.gate,
                        source=p_backend,
                        target=s_backend,
                        boundary=boundary,
                        applied=True,
                        reason="graph_cut",
                        metrics_override=p_metrics,
                    )

                    current_gates = s_gates
                    current_qubits = s_qubits
                    current_backend = s_backend
                    current_metrics = s_metrics
                    current_cost = s_cost
                    pending_switch = None
                    continue

                # If no multi-qubit gate has been processed yet, simply switch
                # the backend without creating a conversion cut. This avoids
                # spurious partitions for early single-qubit preamble.
                if not any(len(g.qubits) > 1 for g in current_gates):
                    boundary = current_qubits & future_qubits[idx]
                    _emit_trace(
                        gate_index=idx,
                        gate_name=gate.gate,
                        source=current_backend,
                        target=backend_trial,
                        boundary=boundary,
                        applied=True,
                        reason="single_qubit_preamble",
                    )
                    current_gates = trial_gates
                    current_qubits = trial_qubits
                    current_backend = backend_trial
                    current_metrics = trial_metrics
                    current_cost = _estimate_cost(current_backend, current_gates)
                    pending_switch = None
                    continue

                boundary = current_qubits & future_qubits[idx]
                if pending_switch is None:
                    pending_switch = _PendingSwitch(
                        start_index=len(current_gates),
                        gate_index=idx,
                        source_backend=current_backend,
                        source_cost=_estimate_cost(current_backend, current_gates),
                        boundary=set(boundary),
                        target_backend=backend_trial,
                        source_metrics=current_metrics.copy() if current_metrics is not None else None,
                    )
                else:
                    pending_switch.gate_index = idx
                    pending_switch.target_backend = backend_trial
                    pending_switch.boundary |= set(boundary)
                    pending_switch.boundary_tuple = ()
                    pending_switch.rank = None
                    pending_switch.frontier = None
                    pending_switch.primitive = None
                    pending_switch.conversion_cost = None
                    pending_switch.window = None
                _emit_trace(
                    gate_index=idx,
                    gate_name=gate.gate,
                    source=current_backend,
                    target=backend_trial,
                    boundary=boundary,
                    applied=False,
                    reason="deferred_switch_candidate",
                    metrics_override=pending_switch.source_metrics,
                )
                current_gates = trial_gates
                current_qubits = trial_qubits
                current_metrics = trial_metrics
                current_cost = _estimate_cost(current_backend, current_gates)
                _maybe_finalize_pending_switch()
                continue
            else:
                current_gates = trial_gates
                current_qubits = trial_qubits
                current_metrics = trial_metrics
                current_cost = _estimate_cost(current_backend, current_gates)
                _maybe_finalize_pending_switch()

        if current_gates:
            partitions.extend(
                self._build_partitions(current_gates, current_backend, current_cost)
            )

        trace_data: List[PartitionTraceEntry]
        if trace_log is not None:
            trace_data = trace_log
        else:
            trace_data = []

        ssd = SSD(partitions=partitions, conversions=conversions, trace=trace_data)
        ssd.build_metadata()
        return ssd

    # ------------------------------------------------------------------
    def parallel_groups(self, gates: List['Gate']) -> List[Tuple[Tuple[int, ...], List['Gate']]]:
        """Analyse entanglement structure within ``gates``.

        Parameters
        ----------
        gates:
            Contiguous list of gates operating under the same simulation
            backend.

        Returns
        -------
        list
            A list of ``(qubits, gate_list)`` tuples, one for each
            independent subcircuit that can be simulated in parallel.
        """

        if not gates:
            return []

        all_qubits = sorted({q for g in gates for q in g.qubits})
        q_to_idx = {q: i for i, q in enumerate(all_qubits)}
        idx_to_q = {i: q for q, i in q_to_idx.items()}
        n = len(all_qubits)

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Build connectivity graph ignoring gate order
        for gate in gates:
            qubits = [q_to_idx[q] for q in gate.qubits]
            if len(qubits) > 1:
                base = qubits[0]
                for other in qubits[1:]:
                    union(base, other)

        groups: Dict[int, List['Gate']] = {find(i): [] for i in range(n)}
        for gate in gates:
            root = find(q_to_idx[gate.qubits[0]])
            groups[root].append(gate)

        result: List[Tuple[Tuple[int, ...], List['Gate']]] = []
        for root, gate_list in groups.items():
            qubits = tuple(idx_to_q[i] for i in range(n) if find(i) == root)
            result.append((tuple(sorted(qubits)), gate_list))
        return result

    def _estimate_boundary_rank(
        self, gates: List['Gate'], boundary: Set[int]
    ) -> int:
        """Heuristically bound the Schmidt rank across ``boundary``.

        The method remaps the fragment's qubits into a dense ordering that
        places the boundary first, allowing reuse of the estimator's
        bond-dimension heuristic without mutating the original gates.
        """

        if not gates or not boundary:
            return 1

        boundary_tuple = tuple(sorted(boundary))
        boundary_size = len(boundary_tuple)
        fragment_qubits: List[int] = []
        for gate in gates:
            for qubit in gate.qubits:
                if qubit not in boundary and qubit not in fragment_qubits:
                    fragment_qubits.append(qubit)

        ordering: List[int] = list(boundary_tuple)
        ordering.extend(fragment_qubits)
        if len(ordering) <= boundary_size:
            return 1

        remap = {qubit: idx for idx, qubit in enumerate(ordering)}

        class _StubGate:
            __slots__ = ("qubits",)

            def __init__(self, qubits: Tuple[int, ...]):
                self.qubits = qubits

        remapped_gates: List[_StubGate] = []
        for gate in gates:
            mapped = tuple(remap[q] for q in gate.qubits if q in remap)
            if len(mapped) >= 2:
                remapped_gates.append(_StubGate(mapped))

        if not remapped_gates:
            return 1

        bonds = self.estimator.bond_dimensions(len(ordering), remapped_gates)
        if not bonds:
            return 1

        idx = boundary_size - 1
        if idx < 0 or idx >= len(bonds):
            return 1

        rank_est = max(1, min(2 ** boundary_size, bonds[idx]))
        return rank_est

    def _select_cut_point(
        self, gates: List['Gate'], gate: 'Gate', future: Set[int]
    ) -> Tuple[int, Set[int]]:
        """Return a cut index and boundary for ``gates``.

        Candidate cuts are ranked using three signals:

        - boundary size (number of shared qubits between fragments),
        - an estimated Schmidt rank for the boundary derived from the cached
          entanglement estimator, and
        - the projected downstream execution cost of the suffix once the
          backend switch is applied.

        Only a shortlist of cut positions is evaluated in detail; it is built
        from the smallest conversion boundaries and optionally expanded with
        neighbouring indices to approximate a lightweight graph/min-cut pass.
        """

        if not gates:
            return 0, set()

        num_gates = len(gates)

        # Prefix and suffix qubit sets -------------------------------
        prefix_qubits: List[Set[int]] = []
        running: Set[int] = set()
        for g in gates:
            running |= set(g.qubits)
            prefix_qubits.append(running.copy())

        suffix_qubits: List[Set[int]] = [set() for _ in range(num_gates + 1)]
        running = set(gate.qubits) | set(future)
        suffix_qubits[num_gates] = running.copy()
        for i in range(num_gates - 1, -1, -1):
            running |= set(gates[i].qubits)
            suffix_qubits[i] = running.copy()

        boundary_map: Dict[int, Set[int]] = {}
        for idx in range(1, num_gates + 1):
            left = prefix_qubits[idx - 1]
            right = suffix_qubits[idx]
            boundary_map[idx] = left & right

        limit = self.graph_cut_candidate_limit
        if limit is None or limit > num_gates:
            limit = num_gates

        # Start with the smallest conversion boundaries.
        sorted_indices = sorted(
            boundary_map.keys(), key=lambda i: (len(boundary_map[i]), i)
        )
        base_candidates = sorted_indices[:limit]
        candidate_set = set(base_candidates)
        candidate_set.add(num_gates)

        if self.graph_cut_neighbor_radius > 0 and base_candidates:
            for idx in base_candidates:
                for offset in range(1, self.graph_cut_neighbor_radius + 1):
                    for neighbour in (idx - offset, idx + offset):
                        if 1 <= neighbour <= num_gates:
                            candidate_set.add(neighbour)

        candidate_indices = sorted(candidate_set)

        best_score: float | None = None
        best_idx = num_gates
        best_boundary: Set[int] = boundary_map.get(num_gates, set())

        for idx in candidate_indices:
            boundary = boundary_map.get(idx, set())
            prefix_gates = gates[:idx]
            suffix_gates = gates[idx:] + [gate]

            # Estimate entanglement using previously analysed metrics.
            rank_est = (
                self._estimate_boundary_rank(prefix_gates, boundary)
                if prefix_gates
                else 1
            )

            # Evaluate the projected downstream cost using the estimator.
            suffix_metrics = FragmentMetrics.from_gates(suffix_gates)
            s_qubits = suffix_metrics.qubits
            sparsity = suffix_metrics.sparsity
            phase_rot = suffix_metrics.phase_rotation_diversity
            amp_rot = suffix_metrics.amplitude_rotation_diversity
            _, cost = self.selector.select(
                suffix_gates,
                len(s_qubits),
                sparsity=sparsity,
                phase_rotation_diversity=phase_rot,
                amplitude_rotation_diversity=amp_rot,
                max_memory=self.max_memory,
                max_time=self.max_time,
                target_accuracy=self.target_accuracy,
            )

            cost_score = math.log1p(cost.memory) + cost.time
            rank_score = math.log2(rank_est) if rank_est > 1 else 0.0
            boundary_score = float(len(boundary))

            score = (
                self.graph_cut_boundary_weight * boundary_score
                + self.graph_cut_rank_weight * rank_score
                + self.graph_cut_cost_weight * cost_score
            )

            # Slight bias towards keeping larger fragments together on ties.
            load_balance = abs(idx - (num_gates - idx))
            score += 0.01 * load_balance

            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
                best_boundary = boundary

        return best_idx, best_boundary

    def _build_partitions(
        self, gates: List['Gate'], backend: Backend, cost: Cost
    ) -> List[SSDPartition]:
        """Compress a contiguous gate list into SSD partitions.

        The routine mirrors the union-find based history compression of the
        original partitioner but operates on a gate subsequence and assumes a
        fixed backend and cost for all resulting partitions.
        """

        if not gates:
            return []

        all_qubits = sorted({q for g in gates for q in g.qubits})
        q_to_idx = {q: i for i, q in enumerate(all_qubits)}
        idx_to_q = {i: q for q, i in q_to_idx.items()}
        n = len(all_qubits)

        parent = list(range(n))
        history: Dict[int, List['Gate']] = {i: [] for i in range(n)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            parent[rb] = ra
            history[ra].extend(history.pop(rb))

        for gate in gates:
            qubits = [q_to_idx[q] for q in gate.qubits]
            if len(qubits) > 1:
                base = qubits[0]
                for other in qubits[1:]:
                    union(base, other)
                root = find(base)
                history[root].append(gate)
            else:
                root = find(qubits[0])
                history[root].append(gate)

        subsystems: Dict[int, List[int]] = {find(i): [] for i in range(n)}
        for idx in range(n):
            subsystems[find(idx)].append(idx_to_q[idx])

        root_info = [(tuple(sorted(qs)), history[r]) for r, qs in subsystems.items()]

        hist_map: Dict[Tuple[str, ...], List[Tuple[Tuple[int, ...], List['Gate']]]] = {}
        for qs, gate_list in root_info:
            names = tuple(g.gate for g in gate_list)
            hist_map.setdefault(names, []).append((qs, gate_list))

        parts: List[SSDPartition] = []
        for hist, group_list in hist_map.items():
            qubit_groups = [qs for qs, _ in group_list]
            parts.append(
                SSDPartition(
                    subsystems=tuple(qubit_groups),
                    history=hist,
                    backend=backend,
                    cost=cost,
                )
            )
        return parts
