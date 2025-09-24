from __future__ import annotations

"""Backend selection based on multi-criteria constraints."""

from typing import Any, List, Tuple, TYPE_CHECKING

from .cost import Backend, Cost, CostEstimator
from . import config

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

        diag_backends: dict[Backend, dict[str, Any]] | None = None
        diag: dict[str, Any] | None = diagnostics

        # Clifford fragments can run on the tableau simulator directly.
        if diag is not None:
            diag_backends = {}
            diag["backends"] = diag_backends
            metrics = diag.setdefault("metrics", {})
            metrics.update({
                "num_qubits": num_qubits,
                "num_gates": num_gates,
            })
        else:
            diag_backends = None

        if allow_tableau and names and all(n in CLIFFORD_GATES for n in names):
            cost = self.estimator.tableau(num_qubits, num_gates)
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

        # ------------------------------------------------------------------
        # Heuristics for decision diagram suitability
        # ------------------------------------------------------------------
        from .sparsity import adaptive_dd_sparsity_threshold

        sparse = sparsity if sparsity is not None else 0.0
        phase_rot = phase_rotation_diversity or 0
        amp_rot = amplitude_rotation_diversity or 0
        nnz = int((1 - sparse) * (2**num_qubits))
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
            effective_sparse >= s_thresh
            and nnz <= config.DEFAULT.dd_nnz_threshold
            and phase_rot <= config.DEFAULT.dd_phase_rotation_diversity_threshold
            and amp_rot <= amp_thresh
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
                }
            )
            metrics["dd_size_override"] = structure_override
            metrics["effective_dd_sparsity"] = effective_sparse

        if passes:
            s_score = effective_sparse / s_thresh if s_thresh > 0 else 0.0
            nnz_score = 1 - nnz / config.DEFAULT.dd_nnz_threshold
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
            if diag is not None:
                metrics = diag.setdefault("metrics", {})
                metrics["decision_diagram_metric"] = metric
                metrics["dd_metric_threshold"] = config.DEFAULT.dd_metric_threshold
            if not dd_metric and diag_backends is not None:
                diag_backends[Backend.DECISION_DIAGRAM] = {
                    "feasible": False,
                    "reasons": ["metric below threshold"],
                    "metric": metric,
                }
        elif diag_backends is not None:
            reasons = []
            if sparse < s_thresh:
                reasons.append("sparsity below threshold")
            if nnz > config.DEFAULT.dd_nnz_threshold:
                reasons.append("nnz above threshold")
            if phase_rot > config.DEFAULT.dd_phase_rotation_diversity_threshold:
                reasons.append("phase diversity above threshold")
            if amp_rot > amp_thresh:
                reasons.append("amplitude diversity above threshold")
            diag_backends[Backend.DECISION_DIAGRAM] = {
                "feasible": False,
                "reasons": reasons,
            }

        # Characterise locality for matrix product state simulation.
        multi = [g for g in gates if len(g.qubits) > 1]
        two_qubit = [g for g in multi if len(g.qubits) == 2]
        higher_arity = len(multi) - len(two_qubit)
        long_range_two = [
            g for g in two_qubit if abs(g.qubits[0] - g.qubits[1]) > 1
        ]
        non_local_count = len(long_range_two) + max(higher_arity, 0)
        total_multi = len(multi)
        long_range_fraction = (
            non_local_count / total_multi if total_multi else 0.0
        )
        max_interaction_distance = 0
        for gate in multi:
            qubits = sorted(gate.qubits)
            if qubits:
                span = qubits[-1] - qubits[0]
                if span > max_interaction_distance:
                    max_interaction_distance = span
        long_range_extent = (
            max(0.0, (max_interaction_distance - 1) / max(num_qubits - 1, 1))
            if total_multi
            else 0.0
        )
        local = total_multi > 0 and non_local_count == 0

        if diag is not None:
            metrics = diag.setdefault("metrics", {})
            metrics["local"] = local
            metrics["mps_long_range_fraction"] = long_range_fraction
            metrics["mps_long_range_extent"] = long_range_extent
            metrics["mps_max_interaction_distance"] = max_interaction_distance

        candidates: dict[Backend, Cost] = {}
        if dd_metric:
            dd_cost = self.estimator.decision_diagram(num_gates=num_gates, frontier=num_qubits)
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
            mps_cost = self.estimator.mps(
                num_qubits,
                num_1q + num_meas,
                num_2q,
                chi=chi,
                svd=True,
                long_range_fraction=long_range_fraction,
                long_range_extent=long_range_extent,
            )
            mps_reasons: list[str] = []
            feasible = True
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

        sv_cost = self.estimator.statevector(num_qubits, num_1q, num_2q, num_meas)
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
            diag_backends[Backend.STATEVECTOR] = {
                "feasible": sv_feasible,
                "reasons": sv_reasons,
                "cost": sv_cost,
            }

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
