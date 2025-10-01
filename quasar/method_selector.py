from __future__ import annotations

"""Backend selection based on multi-criteria constraints."""

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple, TYPE_CHECKING

from .cost import Backend, Cost, CostEstimator
from . import config
from .metrics import FragmentMetrics

if TYPE_CHECKING:  # pragma: no cover
    from .circuit import Gate

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


def _sequential_cost(costs: Sequence[Cost]) -> Cost:
    """Combine sequential cost estimates."""

    if not costs:
        return Cost(0.0, 0.0)
    time = sum(cost.time for cost in costs)
    memory = max(cost.memory for cost in costs)
    log_depth = max(cost.log_depth for cost in costs)
    conversion = sum(cost.conversion for cost in costs)
    replay = sum(cost.replay for cost in costs)
    return Cost(time=time, memory=memory, log_depth=log_depth, conversion=conversion, replay=replay)


def _parallel_groups(gates: Sequence['Gate']) -> List[Tuple[Tuple[int, ...], List['Gate']]]:
    """Return connectivity groups for ``gates`` ignoring order."""

    gate_list = list(gates)
    if not gate_list:
        return []

    qubits = sorted({q for gate in gate_list for q in gate.qubits})
    if not qubits:
        return []

    index = {q: i for i, q in enumerate(qubits)}
    parent = list(range(len(qubits)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for gate in gate_list:
        mapped = [index[q] for q in gate.qubits]
        if len(mapped) > 1:
            base = mapped[0]
            for other in mapped[1:]:
                union(base, other)

    groups: Dict[int, List['Gate']] = {find(i): [] for i in range(len(qubits))}
    for gate in gate_list:
        if not gate.qubits:
            continue
        root = find(index[gate.qubits[0]])
        groups[root].append(gate)

    result: List[Tuple[Tuple[int, ...], List['Gate']]] = []
    for root, gate_seq in groups.items():
        member_qubits = tuple(q for q in qubits if find(index[q]) == root)
        if member_qubits:
            result.append((member_qubits, gate_seq))
    return result


def _statistics(qubits: Sequence[int], gates: Sequence['Gate']) -> Dict[str, Any]:
    """Return basic statistics for ``gates`` on ``qubits``."""

    num_gates = len(gates)
    num_meas = sum(1 for gate in gates if gate.gate.upper() in {"MEASURE", "RESET"})
    num_1q = sum(
        1
        for gate in gates
        if len(gate.qubits) == 1 and gate.gate.upper() not in {"MEASURE", "RESET"}
    )
    num_2q = num_gates - num_1q - num_meas
    num_t = sum(1 for gate in gates if gate.gate.upper() in {"T", "TDG"})
    return {
        "qubits": tuple(qubits),
        "num_qubits": len(qubits),
        "num_gates": num_gates,
        "num_meas": num_meas,
        "num_1q": num_1q,
        "num_2q": num_2q,
        "num_t": num_t,
    }


def _soft_penalty(value: float, threshold: float, softness: float) -> float:
    """Return a smooth penalty in ``[0, 1]`` for exceeding ``threshold``."""

    if threshold <= 0:
        return 0.0 if value > 0 else 1.0
    if value <= threshold:
        return 1.0
    overshoot = value / threshold - 1.0
    softness = max(1.0, softness)
    return 1.0 / (1.0 + overshoot**softness)


class NoFeasibleBackendError(RuntimeError):
    """Raised when no backend satisfies the provided constraints."""


class MethodSelector:
    """Select simulation backends for circuit fragments.

    The selector combines heuristic circuit analysis with calibrated
    performance models to pick a backend that satisfies resource limits and
    optional accuracy/time targets.
    """

    def __init__(self, estimator: CostEstimator | None = None) -> None:
        self.estimator = estimator or CostEstimator()

    def describe_fragment(self, gates: Sequence['Gate']) -> Dict[str, Any]:
        """Compute structural metrics for ``gates``.

        The returned mapping is JSON serialisable and mirrors the quantities the
        selector evaluates when ranking backends, including sparsity,
        interaction locality and an entanglement proxy.
        """

        fragment = FragmentMetrics.from_gates(gates)
        base = fragment.metrics_entry()
        qubit_list = list(base["qubits"])
        multi = [gate for gate in gates if len(gate.qubits) > 1]
        two_qubit = [gate for gate in multi if len(gate.qubits) == 2]
        higher_arity = len(multi) - len(two_qubit)
        non_local_two = [
            gate for gate in two_qubit if abs(gate.qubits[0] - gate.qubits[1]) > 1
        ]
        non_local_count = len(non_local_two) + max(higher_arity, 0)
        total_multi = len(multi)
        max_interaction_distance = 0
        for gate in multi:
            qubits = sorted(gate.qubits)
            if not qubits:
                continue
            span = qubits[-1] - qubits[0]
            if span > max_interaction_distance:
                max_interaction_distance = span
        if total_multi:
            long_range_fraction = non_local_count / total_multi
            extent_den = max(fragment.num_qubits - 1, 1)
            long_range_extent = max(0.0, (max_interaction_distance - 1) / extent_den)
        else:
            long_range_fraction = 0.0
            long_range_extent = 0.0

        qubit_tuple = tuple(qubit_list)
        if qubit_tuple:
            remap = {q: i for i, q in enumerate(qubit_tuple)}
            remapped = [
                SimpleNamespace(qubits=[remap[q] for q in gate.qubits])
                for gate in gates
            ]
            entanglement = self.estimator.entanglement_entropy(
                len(qubit_tuple), remapped
            )
        else:
            entanglement = 0.0

        rotation_total = (
            fragment.phase_rotation_diversity + fragment.amplitude_rotation_diversity
        )
        rotation_density = (
            rotation_total / max(fragment.num_gates, 1)
            if fragment.num_gates
            else 0.0
        )

        summary: Dict[str, Any] = {
            "qubits": qubit_list,
            "num_qubits": fragment.num_qubits,
            "num_gates": fragment.num_gates,
            "num_meas": fragment.num_meas,
            "num_1q": fragment.num_1q,
            "num_2q": fragment.num_2q,
            "num_t": fragment.num_t,
            "sparsity": fragment.sparsity,
            "phase_rotation_diversity": fragment.phase_rotation_diversity,
            "amplitude_rotation_diversity": fragment.amplitude_rotation_diversity,
            "rotation_density": min(max(rotation_density, 0.0), 1.0),
            "entanglement_entropy": entanglement,
            "long_range_fraction": long_range_fraction,
            "long_range_extent": long_range_extent,
            "max_interaction_distance": max_interaction_distance,
            "local": total_multi > 0 and non_local_count == 0,
            "multi_qubit_gates": total_multi,
            "non_local_multi_qubit_gates": non_local_count,
            "non_local_two_qubit_gates": len(non_local_two),
            "higher_arity_gates": max(higher_arity, 0),
        }
        return summary

    def select(
        self,
        gates: List['Gate'],
        num_qubits: int,
        *,
        sparsity: float | None = None,
        phase_rotation_diversity: int | None = None,
        amplitude_rotation_diversity: int | None = None,
        allow_tableau: bool = True,
        max_memory: float | None = None,
        max_time: float | None = None,
        target_accuracy: float | None = None,
        diagnostics: dict[str, Any] | None = None,
        _split_parallel: bool = True,
    ) -> Tuple[Backend, Cost]:
        """Return the preferred backend and its cost for ``gates``.

        Parameters
        ----------
        gates:
            Gate sequence representing a contiguous fragment.
        num_qubits:
            Number of qubits touched by the fragment.
        sparsity, phase_rotation_diversity, amplitude_rotation_diversity:
            Optional analysis metrics for the overall circuit.
        allow_tableau:
            Permit tableau simulation when the fragment is Clifford-only.
        max_memory:
            Upper bound on allowed memory consumption in bytes.
        max_time:
            Upper bound on allowed runtime in seconds.
        target_accuracy:
            Desired lower bound on simulation fidelity.
        diagnostics:
            Optional dictionary populated with the evaluated metrics and
            backend-specific feasibility checks.  The mapping contains a
            ``"metrics"`` entry describing the fragment followed by a
            ``"backends"`` mapping of :class:`Backend` values to diagnostic
            details.  Callers may inspect the populated structure to explain
            backend choices or rejection reasons.

        Raises
        ------
        NoFeasibleBackendError
            If no backend satisfies the resource constraints.
        """

        names = [g.gate.upper() for g in gates]
        num_gates = len(gates)
        num_meas = sum(1 for g in gates if g.gate.upper() in {"MEASURE", "RESET"})
        num_1q = sum(
            1
            for g in gates
            if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
        )
        num_2q = num_gates - num_1q - num_meas
        num_t_gates = sum(1 for n in names if n in {"T", "TDG"})

        groups = _parallel_groups(gates) if (num_qubits > 1 and _split_parallel) else []
        group_stats = [_statistics(qubits, seq) for qubits, seq in groups if qubits]
        if len(group_stats) <= 1 and (
            not group_stats or group_stats[0]["num_qubits"] == num_qubits
        ):
            group_stats = []

        if group_stats:
            metrics_seq = group_stats
        else:
            qubit_tuple = tuple(sorted({q for gate in gates for q in gate.qubits}))
            metrics_seq = [
                {
                    "qubits": qubit_tuple,
                    "num_qubits": num_qubits,
                    "num_gates": num_gates,
                    "num_meas": num_meas,
                    "num_1q": num_1q,
                    "num_2q": num_2q,
                    "num_t": num_t_gates,
                }
            ]

        largest_subsystem = max(m["num_qubits"] for m in metrics_seq) if metrics_seq else 0

        diag_backends: dict[Backend, dict[str, Any]] | None = None
        diag: dict[str, Any] | None = diagnostics

        if diag is not None:
            diag_backends = diag.setdefault("backends", {})
            metrics = diag.setdefault("metrics", {})
            metrics.update({
                "num_qubits": num_qubits,
                "num_gates": num_gates,
            })
            if len(metrics_seq) > 1:
                metrics.update(
                    {
                        "num_subsystems": len(metrics_seq),
                        "largest_subsystem": largest_subsystem,
                        "subsystem_qubits": [m["qubits"] for m in metrics_seq],
                    }
                )
        else:
            diag_backends = None

        if _split_parallel and len(metrics_seq) > 1:
            subsystem_info: List[Dict[str, Any]] = []
            subsystem_costs: List[Cost] = []
            subsystem_backend: Backend | None = None
            consistent_backend = True

            for qubits, seq in groups:
                if not qubits:
                    continue
                sub_diag: dict[str, Any] | None
                if diag is not None:
                    sub_diag = {}
                else:
                    sub_diag = None
                backend, cost = self.select(
                    seq,
                    len(qubits),
                    sparsity=None,
                    phase_rotation_diversity=None,
                    amplitude_rotation_diversity=None,
                    allow_tableau=allow_tableau,
                    max_memory=max_memory,
                    max_time=max_time,
                    target_accuracy=target_accuracy,
                    diagnostics=sub_diag,
                    _split_parallel=True,
                )
                subsystem_costs.append(cost)
                if subsystem_backend is None:
                    subsystem_backend = backend
                elif backend != subsystem_backend:
                    consistent_backend = False
                if diag is not None:
                    subsystem_info.append(
                        {
                            "qubits": tuple(qubits),
                            "backend": backend,
                            "cost": cost,
                            "diagnostics": sub_diag,
                        }
                    )

            if diag is not None:
                diag["parallel_subsystems"] = subsystem_info

            if subsystem_backend is not None and consistent_backend:
                combined_cost = _sequential_cost(subsystem_costs)
                reasons: list[str] = []
                feasible = True
                if max_memory is not None and combined_cost.memory > max_memory:
                    reasons.append("memory > threshold")
                    feasible = False
                if max_time is not None and combined_cost.time > max_time:
                    reasons.append("time > threshold")
                    feasible = False
                if feasible:
                    if diag is not None:
                        metrics = diag.setdefault("metrics", {})
                        overall_summary = self.describe_fragment(gates)
                        fragment_entries: List[Dict[str, Any]] = []
                        overall_entry = dict(overall_summary)
                        overall_entry["scope"] = "fragment"
                        fragment_entries.append(overall_entry)
                        if subsystem_info:
                            for idx, ((_, seq), info) in enumerate(zip(groups, subsystem_info)):
                                sub_diag = info.get("diagnostics") or {}
                                sub_metrics = sub_diag.get("metrics") or {}
                                sub_fragments = sub_metrics.get("fragments")
                                if sub_fragments:
                                    for frag in sub_fragments:
                                        frag_entry = dict(frag)
                                        frag_entry.setdefault("scope", "subsystem")
                                        frag_entry["subsystem_index"] = idx
                                        fragment_entries.append(frag_entry)
                                else:
                                    sub_entry = dict(self.describe_fragment(seq))
                                    sub_entry["scope"] = "subsystem"
                                    sub_entry["subsystem_index"] = idx
                                    fragment_entries.append(sub_entry)
                        overall_metrics_seq = [
                            {
                                "qubits": tuple(sorted({q for gate in gates for q in gate.qubits})),
                                "num_qubits": num_qubits,
                                "num_gates": num_gates,
                                "num_meas": num_meas,
                                "num_1q": num_1q,
                                "num_2q": num_2q,
                                "num_t": num_t_gates,
                            }
                        ]
                        temp_diag: dict[str, Any] = {}
                        temp_backends: dict[Backend, dict[str, Any]] = {}
                        try:
                            self._select_fragment(
                                gates=gates,
                                names=names,
                                num_qubits=num_qubits,
                                num_gates=num_gates,
                                num_meas=num_meas,
                                num_1q=num_1q,
                                num_2q=num_2q,
                                num_t_gates=num_t_gates,
                                metrics_seq=overall_metrics_seq,
                                allow_tableau=allow_tableau,
                                max_memory=max_memory,
                                max_time=max_time,
                                target_accuracy=target_accuracy,
                                sparsity=sparsity,
                                phase_rotation_diversity=phase_rotation_diversity,
                                amplitude_rotation_diversity=amplitude_rotation_diversity,
                                diagnostics=temp_diag,
                                diag_backends=temp_backends,
                            )
                        except NoFeasibleBackendError:
                            temp_diag = {}
                        metrics.update(temp_diag.get("metrics", {}))
                        metrics["fragments"] = fragment_entries
                    if diag is not None and diag_backends is not None:
                        entry = diag_backends.setdefault(
                            subsystem_backend,
                            {
                                "feasible": True,
                                "reasons": [],
                                "cost": combined_cost,
                            },
                        )
                        entry.update(
                            {
                                "feasible": True,
                                "reasons": [],
                                "cost": combined_cost,
                                "selected": True,
                                "parallel": True,
                            }
                        )
                        for other in Backend:
                            if other == subsystem_backend:
                                continue
                            diag_backends.setdefault(
                                other,
                                {
                                    "feasible": False,
                                    "reasons": [
                                        "skipped: parallel subsystem backend preferred"
                                    ],
                                    "selected": False,
                                },
                            )
                        diag["selected_backend"] = subsystem_backend
                        diag["selected_cost"] = combined_cost
                    return subsystem_backend, combined_cost
                else:
                    if diag_backends is not None and subsystem_backend is not None:
                        diag_backends[subsystem_backend] = {
                            "feasible": False,
                            "reasons": reasons,
                            "cost": combined_cost,
                            "parallel": True,
                        }

        return self._select_fragment(
            gates=gates,
            names=names,
            num_qubits=num_qubits,
            num_gates=num_gates,
            num_meas=num_meas,
            num_1q=num_1q,
            num_2q=num_2q,
            num_t_gates=num_t_gates,
            metrics_seq=metrics_seq,
            allow_tableau=allow_tableau,
            max_memory=max_memory,
            max_time=max_time,
            target_accuracy=target_accuracy,
            sparsity=sparsity,
            phase_rotation_diversity=phase_rotation_diversity,
            amplitude_rotation_diversity=amplitude_rotation_diversity,
            diagnostics=diag,
            diag_backends=diag_backends,
        )

    def _select_fragment(
        self,
        *,
        gates: Sequence['Gate'],
        names: Sequence[str],
        num_qubits: int,
        num_gates: int,
        num_meas: int,
        num_1q: int,
        num_2q: int,
        num_t_gates: int,
        metrics_seq: Sequence[Dict[str, Any]],
        allow_tableau: bool,
        max_memory: float | None,
        max_time: float | None,
        target_accuracy: float | None,
        sparsity: float | None,
        phase_rotation_diversity: int | None,
        amplitude_rotation_diversity: int | None,
        diagnostics: dict[str, Any] | None,
        diag_backends: dict[Backend, dict[str, Any]] | None,
    ) -> Tuple[Backend, Cost]:
        diag = diagnostics
        largest_subsystem = (
            max(stats["num_qubits"] for stats in metrics_seq)
            if metrics_seq
            else 0
        )
        fragment_summary = self.describe_fragment(gates)
        sparse_fragment = fragment_summary["sparsity"]
        phase_fragment = fragment_summary["phase_rotation_diversity"]
        amp_fragment = fragment_summary["amplitude_rotation_diversity"]
        entanglement = fragment_summary["entanglement_entropy"]
        long_range_fraction = fragment_summary["long_range_fraction"]
        long_range_extent = fragment_summary["long_range_extent"]
        max_interaction_distance = fragment_summary["max_interaction_distance"]
        local = fragment_summary["local"]
        total_multi = fragment_summary["multi_qubit_gates"]

        sparse = min(
            max(sparsity if sparsity is not None else sparse_fragment, 0.0), 1.0
        )
        phase_rot = float(
            phase_rotation_diversity
            if phase_rotation_diversity is not None
            else phase_fragment
        )
        amp_rot = float(
            amplitude_rotation_diversity
            if amplitude_rotation_diversity is not None
            else amp_fragment
        )
        rotation_total = phase_rot + amp_rot
        rotation_density = (
            rotation_total / max(num_gates, 1) if num_gates else fragment_summary["rotation_density"]
        )
        rotation_density = min(max(rotation_density, 0.0), 1.0)
        if diag is not None:
            metrics = diag.setdefault("metrics", {})
            metrics.update(
                {
                    "sparsity": sparse,
                    "phase_rotation_diversity": phase_rot,
                    "amplitude_rotation_diversity": amp_rot,
                    "rotation_density": rotation_density,
                    "entanglement_entropy": entanglement,
                }
            )
            fragments: List[Dict[str, Any]] = []
            fragment_entry = dict(fragment_summary)
            fragment_entry["scope"] = "fragment"
            fragments.append(fragment_entry)
            if len(metrics_seq) > 1:
                for idx, (qubits, seq) in enumerate(_parallel_groups(gates) if gates else []):
                    if not qubits or not seq:
                        continue
                    sub_summary = self.describe_fragment(seq)
                    sub_entry = dict(sub_summary)
                    sub_entry["scope"] = "subsystem"
                    sub_entry["subsystem_index"] = idx
                    fragments.append(sub_entry)
            metrics["fragments"] = fragments

        if allow_tableau and names and all(n in CLIFFORD_GATES for n in names):
            tableau_costs = [
                self.estimator.tableau(m["num_qubits"], m["num_gates"])
                for m in metrics_seq
            ]
            cost = _sequential_cost(tableau_costs)
            tableau_reasons: list[str] = []
            feasible = True
            if max_memory is not None and cost.memory > max_memory:
                tableau_reasons.append("memory > threshold")
                feasible = False
            if max_time is not None and cost.time > max_time:
                tableau_reasons.append("time > threshold")
                feasible = False
            if diag_backends is not None:
                diag_backends[Backend.TABLEAU] = {
                    "feasible": feasible,
                    "reasons": tableau_reasons,
                    "cost": cost,
                }
            if feasible:
                if diag is not None and diag_backends is not None:
                    diag["selected_backend"] = Backend.TABLEAU
                    diag["selected_cost"] = cost
                    diag_backends[Backend.TABLEAU]["selected"] = True
                    for other in (
                        Backend.DECISION_DIAGRAM,
                        Backend.MPS,
                        Backend.STATEVECTOR,
                    ):
                        diag_backends.setdefault(
                            other,
                            {
                                "feasible": False,
                                "reasons": [
                                    "skipped: tableau preferred for Clifford fragment"
                                ],
                                "selected": False,
                            },
                        )
                return Backend.TABLEAU, cost
        else:
            if diag_backends is not None:
                reason = "tableau disabled" if not allow_tableau else "non-clifford gates"
                diag_backends[Backend.TABLEAU] = {
                    "feasible": False,
                    "reasons": [reason] if names or not allow_tableau else ["no gates"],
                }

        clifford_t = names and all(n in CLIFFORD_PLUS_T_GATES for n in names)

        ext_cost = None
        total_t = sum(stats["num_t"] for stats in metrics_seq)
        if clifford_t and names:
            ext_costs = []
            for stats in metrics_seq:
                num_clifford = stats["num_gates"] - stats["num_t"] - stats["num_meas"]
                num_clifford = max(0, num_clifford)
                ext_costs.append(
                    self.estimator.extended_stabilizer(
                        stats["num_qubits"],
                        num_clifford,
                        stats["num_t"],
                        num_meas=stats["num_meas"],
                        depth=stats["num_gates"],
                    )
                )
            ext_cost = _sequential_cost(ext_costs)
            ext_reasons: list[str] = []
            ext_feasible = True
            if max_memory is not None and ext_cost.memory > max_memory:
                ext_reasons.append("memory > threshold")
                ext_feasible = False
            if max_time is not None and ext_cost.time > max_time:
                ext_reasons.append("time > threshold")
                ext_feasible = False
            if ext_feasible:
                candidates: dict[Backend, Cost] = {Backend.EXTENDED_STABILIZER: ext_cost}
            else:
                candidates = {}
            if diag_backends is not None:
                diag_backends[Backend.EXTENDED_STABILIZER] = {
                    "feasible": ext_feasible,
                    "reasons": ext_reasons,
                    "cost": ext_cost,
                    "num_t_gates": total_t,
                }
        else:
            candidates = {}
            if diag_backends is not None:
                reason = "no gates" if not names else "contains non-Clifford+T gates"
                diag_backends[Backend.EXTENDED_STABILIZER] = {
                    "feasible": False,
                    "reasons": [reason],
                    "num_t_gates": total_t,
                }

        # ------------------------------------------------------------------
        # Heuristics for decision diagram suitability
        # ------------------------------------------------------------------
        from .sparsity import adaptive_dd_sparsity_threshold

        nnz = int((1 - sparse) * (2**num_qubits))
        nnz = max(1, nnz)
        s_thresh = adaptive_dd_sparsity_threshold(num_qubits)
        amp_thresh = config.adaptive_dd_amplitude_rotation_threshold(num_qubits, sparse)
        size_override = num_qubits <= 10 and nnz <= config.DEFAULT.dd_nnz_threshold
        entangling_names = {g.gate.upper() for g in gates if len(g.qubits) > 1}
        structure_override = (
            size_override
            and sparse < s_thresh
            and amp_rot == 0
            and phase_rot <= 2
            and entangling_names <= {"CX"}
        )
        effective_sparse = sparse
        if structure_override:
            # Small fragments often exhibit structured superpositions (e.g. Grover)
            # that trigger dense heuristics despite being favourable for DDs.
            # Boost the sparsity score smoothly so the metric can consider DD.
            bonus = 1.0 + max(0, 10 - num_qubits) / 5.0
            effective_sparse = min(1.0, max(sparse, s_thresh * bonus))
        passes = (
            effective_sparse >= s_thresh and nnz <= config.DEFAULT.dd_nnz_threshold
        )
        softness = max(1.0, config.DEFAULT.dd_rotation_softness)
        hybrid_frontier = 0
        if num_qubits > 0 and nnz > 0:
            hybrid_frontier = min(num_qubits, int(math.ceil(math.log2(nnz))))
        if num_gates and hybrid_frontier == 0:
            hybrid_frontier = 1
        hybrid_penalty = max(0, num_qubits - hybrid_frontier)
        hybrid_replay = 0
        if num_qubits > 0 and num_gates > 0:
            hybrid_replay = int(
                round(num_gates * hybrid_penalty / max(num_qubits, 1))
            )
        dd_metric = False
        if diag is not None:
            metrics = diag.setdefault("metrics", {})
            metrics.update(
                {
                    "sparsity": sparse,
                    "phase_rotation_diversity": phase_rot,
                    "amplitude_rotation_diversity": amp_rot,
                    "nnz": nnz,
                    "local": False,
                    "dd_hybrid_frontier": hybrid_frontier,
                    "dd_hybrid_penalty": hybrid_penalty,
                }
            )
            metrics["dd_size_override"] = structure_override
            metrics["effective_dd_sparsity"] = effective_sparse

        if passes:
            s_score = effective_sparse / s_thresh if s_thresh > 0 else 0.0
            s_score = min(max(s_score, 0.0), 1.2)
            nnz_score = 1 - nnz / config.DEFAULT.dd_nnz_threshold
            nnz_score = min(max(nnz_score, -1.0), 1.0)
            phase_score = _soft_penalty(
                phase_rot,
                config.DEFAULT.dd_phase_rotation_diversity_threshold,
                softness,
            )
            amp_score = _soft_penalty(amp_rot, amp_thresh, softness)
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
            if diag is not None:
                metrics = diag.setdefault("metrics", {})
                metrics["decision_diagram_metric"] = metric
                metrics["dd_metric_threshold"] = config.DEFAULT.dd_metric_threshold
                metrics["dd_phase_score"] = phase_score
                metrics["dd_amplitude_score"] = amp_score
                metrics["dd_hybrid_replay"] = hybrid_replay
            if not dd_metric and diag_backends is not None:
                diag_backends[Backend.DECISION_DIAGRAM] = {
                    "feasible": False,
                    "reasons": ["metric below threshold"],
                    "metric": metric,
                    "hybrid_frontier": hybrid_frontier,
                }
        elif diag_backends is not None:
            reasons = []
            if sparse < s_thresh:
                reasons.append("sparsity below threshold")
            if nnz > config.DEFAULT.dd_nnz_threshold:
                reasons.append("nnz above threshold")
            phase_ratio = phase_rot / max(
                config.DEFAULT.dd_phase_rotation_diversity_threshold, 1
            )
            amp_ratio = amp_rot / max(amp_thresh, 1)
            if phase_ratio > 1:
                reasons.append("phase diversity incurs penalty")
            if amp_ratio > 1:
                reasons.append("amplitude diversity incurs penalty")
            diag_backends[Backend.DECISION_DIAGRAM] = {
                "feasible": False,
                "reasons": reasons,
                "hybrid_frontier": hybrid_frontier,
            }

        if diag is not None:
            metrics = diag.setdefault("metrics", {})
            metrics["local"] = local
            metrics["mps_long_range_fraction"] = long_range_fraction
            metrics["mps_long_range_extent"] = long_range_extent
            metrics["mps_max_interaction_distance"] = max_interaction_distance

        if dd_metric:
            dd_costs: list[Cost] = []
            for stats in metrics_seq:
                sub_qubits = stats["num_qubits"]
                sub_gates = stats["num_gates"]
                sub_nnz = max(1, int((1 - sparse) * (2**sub_qubits)))
                sub_frontier = 0
                if sub_qubits > 0 and sub_nnz > 0:
                    sub_frontier = min(
                        sub_qubits, int(math.ceil(math.log2(sub_nnz)))
                    )
                if sub_gates and sub_frontier == 0:
                    sub_frontier = 1
                sub_penalty = max(0, sub_qubits - sub_frontier)
                sub_replay = 0
                if sub_qubits > 0 and sub_gates > 0 and sub_penalty > 0:
                    sub_replay = int(
                        round(sub_gates * sub_penalty / max(sub_qubits, 1))
                    )
                dd_costs.append(
                    self.estimator.decision_diagram(
                        num_gates=sub_gates,
                        frontier=sub_qubits,
                        sparsity=sparse,
                        phase_rotation_diversity=phase_rot,
                        amplitude_rotation_diversity=amp_rot,
                        entanglement_entropy=entanglement,
                        converted_frontier=sub_frontier,
                        hybrid_replay_gates=sub_replay,
                    )
                )
            dd_cost = _sequential_cost(dd_costs)
            dd_reasons: list[str] = []
            feasible = True
            if max_memory is not None and dd_cost.memory > max_memory:
                dd_reasons.append("memory > threshold")
                feasible = False
            if max_time is not None and dd_cost.time > max_time:
                dd_reasons.append("time > threshold")
                feasible = False
            if feasible:
                candidates[Backend.DECISION_DIAGRAM] = dd_cost
            if diag_backends is not None:
                entry = {
                    "feasible": feasible,
                    "reasons": dd_reasons,
                    "cost": dd_cost,
                }
                if diag is not None and "metrics" in diag and "decision_diagram_metric" in diag["metrics"]:
                    entry["metric"] = diag["metrics"]["decision_diagram_metric"]
                entry["hybrid_frontier"] = hybrid_frontier
                entry["hybrid_penalty"] = hybrid_penalty
                diag_backends[Backend.DECISION_DIAGRAM] = entry

        if total_multi:
            chi = getattr(self.estimator, "chi_max", None) or 4
            chi_cap: int | None = None
            infeasible_chi = False
            if target_accuracy is not None or max_memory is not None:
                chi_cap = self.estimator.chi_for_constraints(
                    num_qubits,
                    gates,
                    target_accuracy
                    if target_accuracy is not None
                    else config.DEFAULT.mps_target_fidelity,
                    max_memory,
                )
                if chi_cap > 0:
                    chi = chi_cap
                else:
                    infeasible_chi = True
            over_fraction = (
                long_range_fraction
                > config.DEFAULT.mps_long_range_fraction_threshold
            )
            over_extent = (
                long_range_extent
                > config.DEFAULT.mps_long_range_extent_threshold
            )
            enforce_locality = (
                over_fraction
                and over_extent
                and (
                    largest_subsystem
                    >= config.DEFAULT.mps_locality_strict_qubits
                    or max_interaction_distance
                    >= config.DEFAULT.mps_locality_strict_distance
                )
            )
            locality_reasons: list[str] = []
            if over_fraction:
                locality_reasons.append(
                    "non-local interactions exceed fraction threshold"
                )
            if over_extent:
                locality_reasons.append("interaction span exceeds extent threshold")
            capture_mps_details = diag_backends is not None
            mps_detail: dict[str, float] | None = {} if capture_mps_details else None
            mps_costs: list[Cost] = []
            for idx, stats in enumerate(metrics_seq):
                detail_arg = mps_detail if capture_mps_details and idx == 0 else None
                mps_costs.append(
                    self.estimator.mps(
                        stats["num_qubits"],
                        stats["num_1q"] + stats["num_meas"],
                        stats["num_2q"],
                        chi=chi,
                        svd=True,
                        entanglement_entropy=entanglement,
                        sparsity=sparse,
                        rotation_diversity=rotation_density,
                        long_range_fraction=long_range_fraction,
                        long_range_extent=long_range_extent,
                        details=detail_arg,
                    )
                )
            mps_cost = _sequential_cost(mps_costs)
            mps_reasons: list[str] = []
            feasible = True
            if enforce_locality:
                mps_reasons.extend(locality_reasons)
                feasible = False
            if infeasible_chi:
                mps_reasons.append("bond dimension exceeds memory limit")
                feasible = False
            if max_memory is not None and mps_cost.memory > max_memory:
                mps_reasons.append("memory > threshold")
                feasible = False
            if max_time is not None and mps_cost.time > max_time:
                mps_reasons.append("time > threshold")
                feasible = False
            if diag_backends is not None:
                entry = {
                    "feasible": feasible,
                    "reasons": mps_reasons,
                    "cost": mps_cost,
                    "chi": chi,
                    "long_range_fraction": long_range_fraction,
                    "long_range_extent": long_range_extent,
                    "max_interaction_distance": max_interaction_distance,
                }
                if mps_detail:
                    entry["modifiers"] = dict(mps_detail)
                if locality_reasons:
                    entry["locality_warnings"] = locality_reasons
                if chi_cap is not None:
                    entry["chi_limit"] = chi_cap
                diag_backends[Backend.MPS] = entry
            if feasible:
                candidates[Backend.MPS] = mps_cost
        elif diag_backends is not None:
            reason = "no multi-qubit gates"
            diag_backends[Backend.MPS] = {
                "feasible": False,
                "reasons": [reason],
                "long_range_fraction": 0.0,
                "long_range_extent": 0.0,
                "max_interaction_distance": 0,
            }

        capture_sv_details = diag_backends is not None
        sv_detail: dict[str, float] | None = {} if capture_sv_details else None
        sv_costs: list[Cost] = []
        for idx, stats in enumerate(metrics_seq):
            detail_arg = sv_detail if capture_sv_details and idx == 0 else None
            sv_costs.append(
                self.estimator.statevector(
                    stats["num_qubits"],
                    stats["num_1q"],
                    stats["num_2q"],
                    stats["num_meas"],
                    sparsity=sparse,
                    rotation_diversity=rotation_density,
                    entanglement_entropy=entanglement,
                    details=detail_arg,
                )
            )
        sv_cost = _sequential_cost(sv_costs)
        sv_reasons: list[str] = []
        sv_feasible = True
        if max_memory is not None and sv_cost.memory > max_memory:
            sv_reasons.append("memory > threshold")
            sv_feasible = False
        if max_time is not None and sv_cost.time > max_time:
            sv_reasons.append("time > threshold")
            sv_feasible = False
        if sv_feasible:
            candidates[Backend.STATEVECTOR] = sv_cost
        if diag_backends is not None:
            entry = {
                "feasible": sv_feasible,
                "reasons": sv_reasons,
                "cost": sv_cost,
            }
            if sv_detail:
                entry["modifiers"] = dict(sv_detail)
            diag_backends[Backend.STATEVECTOR] = entry

        if not candidates:
            if diag is not None:
                diag["selected_backend"] = None
                diag["selected_cost"] = None
            raise NoFeasibleBackendError(
                "No simulation backend satisfies the given constraints"
            )

        backend = min(
            candidates, key=lambda b: (candidates[b].memory, candidates[b].time)
        )
        if diag is not None:
            diag["selected_backend"] = backend
            diag["selected_cost"] = candidates[backend]
            for candidate, entry in diag_backends.items():
                entry["selected"] = candidate == backend
        return backend, candidates[backend]
