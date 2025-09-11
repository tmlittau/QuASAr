from __future__ import annotations

"""Backend selection based on multi-criteria constraints."""

from typing import List, Tuple, TYPE_CHECKING

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
    "CSWAP",
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

        Raises
        ------
        NoFeasibleBackendError
            If no backend satisfies the resource constraints.
        """

        names = [g.gate.upper() for g in gates]
        num_gates = len(gates)
        num_meas = sum(1 for g in gates if g.gate.upper() in {"MEASURE", "RESET"})
        num_1q = sum(
            1 for g in gates if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
        )
        num_2q = num_gates - num_1q - num_meas

        # Clifford fragments can run on the tableau simulator directly.
        if allow_tableau and names and all(n in CLIFFORD_GATES for n in names):
            cost = self.estimator.tableau(num_qubits, num_gates)
            if (max_memory is None or cost.memory <= max_memory) and (
                max_time is None or cost.time <= max_time
            ):
                return Backend.TABLEAU, cost

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
        passes = (
            sparse >= s_thresh
            and nnz <= config.DEFAULT.dd_nnz_threshold
            and phase_rot <= config.DEFAULT.dd_phase_rotation_diversity_threshold
            and amp_rot <= amp_thresh
        )
        dd_metric = False
        if passes:
            s_score = sparse / s_thresh if s_thresh > 0 else 0.0
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

        # Determine whether the fragment is local enough for MPS
        multi = [g for g in gates if len(g.qubits) > 1]
        local = bool(multi) and all(
            len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi
        )

        candidates: dict[Backend, Cost] = {}
        if dd_metric:
            dd_cost = self.estimator.decision_diagram(num_gates=num_gates, frontier=num_qubits)
            if (max_memory is None or dd_cost.memory <= max_memory) and (
                max_time is None or dd_cost.time <= max_time
            ):
                candidates[Backend.DECISION_DIAGRAM] = dd_cost

        if local:
            chi = getattr(self.estimator, "chi_max", None) or 4
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
            mps_cost = self.estimator.mps(
                num_qubits,
                num_1q + num_meas,
                num_2q,
                chi=chi,
                svd=True,
            )
            if (max_memory is None or mps_cost.memory <= max_memory) and (
                max_time is None or mps_cost.time <= max_time
            ):
                candidates[Backend.MPS] = mps_cost

        sv_cost = self.estimator.statevector(num_qubits, num_1q, num_2q, num_meas)
        if (max_memory is None or sv_cost.memory <= max_memory) and (
            max_time is None or sv_cost.time <= max_time
        ):
            candidates[Backend.STATEVECTOR] = sv_cost

        if not candidates:
            raise NoFeasibleBackendError(
                "No simulation backend satisfies the given constraints"
            )

        backend = min(
            candidates, key=lambda b: (candidates[b].memory, candidates[b].time)
        )
        return backend, candidates[backend]
