from __future__ import annotations

from typing import Callable, Dict, List, Tuple, TYPE_CHECKING, Set

from .ssd import SSD, SSDPartition, ConversionLayer, PartitionTraceEntry
from .cost import Backend, CostEstimator, Cost
from .method_selector import MethodSelector, NoFeasibleBackendError

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
    ):
        self.estimator = estimator or CostEstimator()
        self.selector = selector or MethodSelector(self.estimator)
        self.max_memory = max_memory
        self.max_time = max_time
        self.target_accuracy = target_accuracy

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

        from .circuit import Circuit
        from .sparsity import sparsity_estimate
        from .symmetry import (
            phase_rotation_diversity as rot_phase,
            amplitude_rotation_diversity as rot_amp,
        )

        def _metrics(gate_seq: List['Gate']) -> Tuple[float, int, int]:
            """Return sparsity and rotation metrics for ``gate_seq``."""
            frag = Circuit(gate_seq, use_classical_simplification=False)
            return (
                sparsity_estimate(frag),
                rot_phase(frag),
                rot_amp(frag),
            )

        def _conversion_diagnostics(
            source: Backend | None,
            target: Backend,
            boundary: Set[int],
        ) -> Tuple[Tuple[int, ...], int, int | None, int | None, str | None, Cost | None]:
            boundary_tuple = tuple(sorted(boundary))
            size = len(boundary_tuple)
            if size == 0 or source is None or source == target:
                rank = 1 if size == 0 else 2 ** size
                frontier = size
                return boundary_tuple, size, rank, frontier, None, None
            rank = 2 ** size
            frontier = size
            conv_est = self.estimator.conversion(
                source,
                target,
                num_qubits=size,
                rank=rank,
                frontier=frontier,
            )
            return boundary_tuple, size, rank, frontier, conv_est.primitive, conv_est.cost

        def _emit_trace(
            *,
            gate_index: int,
            gate_name: str,
            source: Backend | None,
            target: Backend,
            boundary: Set[int],
            applied: bool,
            reason: str,
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
            ) = _conversion_diagnostics(source, target, boundary)
            entry = PartitionTraceEntry(
                gate_index=gate_index,
                gate_name=gate_name,
                from_backend=source,
                to_backend=target,
                boundary=boundary_tuple,
                boundary_size=boundary_size,
                rank=rank,
                frontier=frontier,
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
            s_est, pr_div, ar_div = _metrics(trial_gates)
            backend_trial, cost_trial = self.selector.select(
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
                current_cost = cost_trial
                continue

            if current_backend is None:
                current_gates = trial_gates
                current_qubits = trial_qubits
                current_backend = backend_trial
                current_cost = cost_trial
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
                        ps, pr, ar = _metrics(prefix)
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
                    ss, sr, ar = _metrics(s_gates)
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
                    ) = _conversion_diagnostics(p_backend, s_backend, boundary)
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
                    )

                    current_gates = s_gates
                    current_qubits = s_qubits
                    current_backend = s_backend
                    current_cost = s_cost
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
                    current_cost = cost_trial
                    continue

                # Finalise current partition before switching backends
                partitions.extend(
                    self._build_partitions(
                        current_gates, current_backend, current_cost
                    )
                )

                boundary = current_qubits & future_qubits[idx]
                (
                    boundary_tuple,
                    boundary_size,
                    rank,
                    frontier,
                    primitive,
                    conv_cost,
                ) = _conversion_diagnostics(current_backend, backend_trial, boundary)
                if (
                    boundary_size
                    and primitive is not None
                    and conv_cost is not None
                ):
                    conversions.append(
                        ConversionLayer(
                            boundary=boundary_tuple,
                            source=current_backend,
                            target=backend_trial,
                            rank=rank,
                            frontier=frontier,
                            primitive=primitive,
                            cost=conv_cost,
                        )
                    )
                _emit_trace(
                    gate_index=idx,
                    gate_name=gate.gate,
                    source=current_backend,
                    target=backend_trial,
                    boundary=boundary,
                    applied=True,
                    reason="backend_switch",
                )

                current_gates = [gate]
                current_qubits = set(gate.qubits)
                # Start the new fragment using the backend decided for the
                # extended gate sequence (``backend_trial``).
                if backend_trial == Backend.TABLEAU:
                    current_cost = self.estimator.tableau(len(current_qubits), 1)
                elif backend_trial == Backend.MPS:
                    is_meas = gate.gate.upper() in {"MEASURE", "RESET"}
                    num_1q = 1 if len(gate.qubits) == 1 or is_meas else 0
                    num_2q = 1 if len(gate.qubits) > 1 else 0
                    current_cost = self.estimator.mps(
                        len(current_qubits), num_1q, num_2q, chi=4, svd=True
                    )
                elif backend_trial == Backend.DECISION_DIAGRAM:
                    current_cost = self.estimator.decision_diagram(
                        num_gates=1, frontier=len(current_qubits)
                    )
                else:
                    is_meas = gate.gate.upper() in {"MEASURE", "RESET"}
                    num_1q = 1 if len(gate.qubits) == 1 and not is_meas else 0
                    num_2q = 1 if len(gate.qubits) > 1 else 0
                    num_meas = 1 if is_meas else 0
                    current_cost = self.estimator.statevector(
                        len(current_qubits), num_1q, num_2q, num_meas
                    )
                current_backend = backend_trial
            else:
                current_gates = trial_gates
                current_qubits = trial_qubits
                current_cost = cost_trial

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

    def _select_cut_point(
        self, gates: List['Gate'], gate: 'Gate', future: Set[int]
    ) -> Tuple[int, Set[int]]:
        """Return a cut index and boundary for ``gates``.

        A simple graph-based heuristic evaluates all possible cut positions
        within ``gates`` and chooses the one that minimises a cost function
        combining boundary size and load imbalance.  ``gate`` is the first
        operation of the new fragment and ``future`` are the qubits used by
        the remaining gates in the circuit.
        """

        if not gates:
            return 0, set()

        # Prefix and suffix qubit sets -------------------------------
        prefix: List[Set[int]] = []
        running: Set[int] = set()
        for g in gates:
            running |= set(g.qubits)
            prefix.append(running.copy())

        suffix: List[Set[int]] = []
        running = set(gate.qubits) | set(future)
        suffix.append(running.copy())
        for g in reversed(gates):
            running |= set(g.qubits)
            suffix.append(running.copy())
        suffix.reverse()  # index i corresponds to cut after gates[:i]

        best_cost: float | None = None
        best_idx = len(gates)
        best_boundary: Set[int] = set()

        for idx in range(1, len(gates) + 1):
            left = prefix[idx - 1]
            right = suffix[idx]
            boundary = left & right
            load_diff = abs(idx - (len(gates) - idx + 1))
            cost = len(boundary) * 10 + load_diff
            if best_cost is None or cost < best_cost:
                best_cost = cost
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
