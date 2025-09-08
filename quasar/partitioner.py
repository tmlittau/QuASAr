from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING, Set

from .ssd import SSD, SSDPartition, ConversionLayer
from .cost import Backend, CostEstimator, Cost
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
    "CSWAP",
}


class Partitioner:
    """Partition circuits and assign simulation methods."""

    def __init__(self, estimator: CostEstimator | None = None):
        self.estimator = estimator or CostEstimator()

    def partition(self, circuit: 'Circuit') -> SSD:
        if not circuit.gates:
            return SSD([])

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

        sparsity = getattr(circuit, "sparsity", None)
        phase_rot = getattr(circuit, "phase_rotation_diversity", None)
        amp_rot = getattr(circuit, "amplitude_rotation_diversity", None)
        from .sparsity import adaptive_dd_sparsity_threshold, sparsity_estimate
        from .symmetry import (
            phase_rotation_diversity as rot_phase,
            amplitude_rotation_diversity as rot_amp,
        )
        if sparsity is None:
            sparsity = sparsity_estimate(circuit)
        if phase_rot is None:
            phase_rot = rot_phase(circuit)
        if amp_rot is None:
            amp_rot = rot_amp(circuit)
        nnz_estimate = int((1 - sparsity) * (2 ** circuit.num_qubits))
        s_thresh = adaptive_dd_sparsity_threshold(circuit.num_qubits)
        amp_thresh = config.adaptive_dd_amplitude_rotation_threshold(
            circuit.num_qubits, sparsity
        )
        passes = (
            sparsity >= s_thresh
            and nnz_estimate <= config.DEFAULT.dd_nnz_threshold
            and phase_rot <= config.DEFAULT.dd_phase_rotation_diversity_threshold
            and amp_rot <= amp_thresh
        )
        dd_metric = False
        if passes:
            s_score = sparsity / s_thresh if s_thresh > 0 else 0.0
            nnz_score = 1 - nnz_estimate / config.DEFAULT.dd_nnz_threshold
            phase_score = 1 - phase_rot / config.DEFAULT.dd_phase_rotation_diversity_threshold
            amp_score = 1 - amp_rot / amp_thresh
            weight_sum = (
                config.DEFAULT.dd_sparsity_weight
                + config.DEFAULT.dd_nnz_weight
                + config.DEFAULT.dd_phase_rotation_weight
                + config.DEFAULT.dd_amplitude_rotation_weight
            )
            weighted = (
                config.DEFAULT.dd_sparsity_weight * s_score
                + config.DEFAULT.dd_nnz_weight * nnz_score
                + config.DEFAULT.dd_phase_rotation_weight * phase_score
                + config.DEFAULT.dd_amplitude_rotation_weight * amp_score
            )
            metric = weighted / weight_sum if weight_sum else 0.0
            dd_metric = metric >= config.DEFAULT.dd_metric_threshold

        for idx, gate in enumerate(gates):
            trial_gates = current_gates + [gate]
            trial_qubits = current_qubits | set(gate.qubits)
            backend_trial, cost_trial = self._choose_backend(
                trial_gates, len(trial_qubits), dd_metric=dd_metric
            )

            # If we've already committed to a statevector simulation, keep it
            # for the remainder of the fragment to avoid flip-flopping to less
            # expressive backends based on early gates.
            if current_backend == Backend.STATEVECTOR:
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
                # If no multi-qubit gate has been processed yet, simply switch
                # the backend without creating a conversion cut. This avoids
                # spurious partitions for early single-qubit preamble.
                if not any(len(g.qubits) > 1 for g in current_gates):
                    current_gates = trial_gates
                    current_qubits = trial_qubits
                    current_backend = backend_trial
                    current_cost = cost_trial
                    continue

                # Finalise current partition before switching backends
                partitions.extend(self._build_partitions(current_gates, current_backend, current_cost))

                boundary = sorted(current_qubits & future_qubits[idx])
                if boundary:
                    rank = 2 ** len(boundary)
                    frontier = len(boundary)
                    conv_est = self.estimator.conversion(
                        current_backend,
                        backend_trial,
                        num_qubits=len(boundary),
                        rank=rank,
                        frontier=frontier,
                    )
                    conversions.append(
                        ConversionLayer(
                            boundary=tuple(boundary),
                            source=current_backend,
                            target=backend_trial,
                            rank=rank,
                            frontier=frontier,
                            primitive=conv_est.primitive,
                            cost=conv_est.cost,
                        )
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
            partitions.extend(self._build_partitions(current_gates, current_backend, current_cost))

        return SSD(partitions=partitions, conversions=conversions)

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

    # ------------------------------------------------------------------
    def _choose_backend(
        self, gates: List['Gate'], num_qubits: int, *, dd_metric: bool = False
    ) -> Tuple[Backend, 'Cost']:
        """Select the best simulation backend for a partition."""

        names = [g.gate.upper() for g in gates]
        num_gates = len(gates)
        num_meas = sum(1 for g in gates if g.gate.upper() in {"MEASURE", "RESET"})
        num_1q = sum(
            1 for g in gates if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
        )
        num_2q = num_gates - num_1q - num_meas

        if all(name in CLIFFORD_GATES for name in names):
            backend = Backend.TABLEAU
            cost = self.estimator.tableau(num_qubits, num_gates)
            return backend, cost

        multi = [g for g in gates if len(g.qubits) > 1]
        local = multi and all(
            len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi
        )

        candidates: List[Tuple[Backend, Cost]] = []
        # Candidate costs computed using calibrated coefficients.
        if dd_metric:
            dd_cost = self.estimator.decision_diagram(
                num_gates=num_gates, frontier=num_qubits
            )
            candidates.append((Backend.DECISION_DIAGRAM, dd_cost))
        if local:
            mps_cost = self.estimator.mps(
                num_qubits,
                num_1q + num_meas,
                num_2q,
                chi=4,
                svd=True,
            )
            candidates.append((Backend.MPS, mps_cost))
        sv_cost = self.estimator.statevector(num_qubits, num_1q, num_2q, num_meas)
        candidates.append((Backend.STATEVECTOR, sv_cost))

        # Selection leverages calibrated coefficients for realism.
        backend, cost = min(candidates, key=lambda bc: (bc[1].memory, bc[1].time))
        return backend, cost

    # ------------------------------------------------------------------
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
