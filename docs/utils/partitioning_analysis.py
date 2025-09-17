"""Helpers for scoring synthetic partitioning scenarios.

This module provides a light-weight façade over :class:`quasar.cost.CostEstimator`
so that documentation and interactive tutorials can explore the planner's
behaviour without constructing full :class:`~quasar.circuit.Gate` objects.  The
functions mirror :class:`quasar.method_selector.MethodSelector`'s feasibility
checks and expose aggregate plan cost calculations that include conversion
steps between heterogeneous backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence

from quasar import config
from quasar.cost import Backend, Cost, CostEstimator, ConversionEstimate


@dataclass(slots=True)
class FragmentStats:
    """Summary statistics describing a synthetic circuit fragment."""

    num_qubits: int
    num_1q_gates: int
    num_2q_gates: int
    num_measurements: int = 0
    is_clifford: bool = False
    is_local: bool = False
    frontier: int | None = None
    chi: int | Sequence[int] | None = None

    @property
    def total_gates(self) -> int:
        """Return the total number of operations in the fragment."""

        return self.num_1q_gates + self.num_2q_gates + self.num_measurements


@dataclass(slots=True)
class BoundarySpec:
    """Parameters describing a conversion boundary between fragments."""

    num_qubits: int
    rank: int
    frontier: int
    window: int | None = None
    window_1q_gates: int = 0
    window_2q_gates: int = 0
    s_max: int | None = None
    r_max: int | None = None
    q_max: int | None = None


def _update_peak_memory(current: float, candidates: Iterable[float]) -> float:
    """Return the maximum memory footprint seen across ``candidates``."""

    for value in candidates:
        if value > current:
            current = value
    return current


def evaluate_fragment_backends(
    stats: FragmentStats,
    *,
    sparsity: float | None = None,
    phase_rotation_diversity: int | None = None,
    amplitude_rotation_diversity: int | None = None,
    allow_tableau: bool = True,
    max_memory: float | None = None,
    max_time: float | None = None,
    target_accuracy: float | None = None,
    estimator: CostEstimator | None = None,
) -> tuple[Backend | None, MutableMapping[str, object]]:
    """Evaluate backend feasibility for a synthetic fragment.

    Parameters
    ----------
    stats:
        Summary describing the fragment under consideration.
    sparsity, phase_rotation_diversity, amplitude_rotation_diversity:
        Circuit-level metrics used by the decision diagram heuristic.
    allow_tableau:
        Permit stabiliser simulation when the fragment is Clifford only.
    max_memory, max_time:
        Optional resource limits applied to each backend estimate.
    target_accuracy:
        Desired lower bound on simulation fidelity.  When supplied together
        with ``stats.chi`` the value is surfaced in the diagnostics to mirror
        :class:`~quasar.method_selector.MethodSelector`.
    estimator:
        Optional estimator instance.  A new :class:`CostEstimator` is created
        when omitted.

    Returns
    -------
    tuple
        Selected backend (or ``None`` when all candidates are infeasible) and
        a diagnostics mapping mirroring the structure produced by
        :class:`~quasar.method_selector.MethodSelector`.
    """

    estimator = estimator or CostEstimator()

    diag: MutableMapping[str, object] = {
        "metrics": {
            "num_qubits": stats.num_qubits,
            "num_gates": stats.total_gates,
            "sparsity": sparsity if sparsity is not None else 0.0,
            "phase_rotation_diversity": phase_rotation_diversity or 0,
            "amplitude_rotation_diversity": amplitude_rotation_diversity or 0,
            "local": stats.is_local,
        },
        "backends": {},
    }

    candidates: dict[Backend, Cost] = {}
    num_qubits = stats.num_qubits
    num_gates = stats.total_gates
    frontier = stats.frontier or stats.num_qubits

    # ------------------------------------------------------------------
    # Tableau backend
    # ------------------------------------------------------------------
    if allow_tableau and stats.is_clifford and num_gates:
        table_cost = estimator.tableau(num_qubits, num_gates)
        feasible = True
        reasons: list[str] = []
        if max_memory is not None and table_cost.memory > max_memory:
            feasible = False
            reasons.append("memory > threshold")
        if max_time is not None and table_cost.time > max_time:
            feasible = False
            reasons.append("time > threshold")
        diag["backends"][Backend.TABLEAU] = {
            "feasible": feasible,
            "reasons": reasons,
            "cost": table_cost,
        }
        if feasible:
            candidates[Backend.TABLEAU] = table_cost
    else:
        reason = "tableau disabled"
        if not allow_tableau:
            reason = "tableau disabled"
        elif not stats.is_clifford:
            reason = "non-clifford fragment"
        elif not num_gates:
            reason = "no gates"
        diag["backends"][Backend.TABLEAU] = {
            "feasible": False,
            "reasons": [reason],
        }

    # ------------------------------------------------------------------
    # Decision diagram backend
    # ------------------------------------------------------------------
    sparse = sparsity if sparsity is not None else 0.0
    phase_rot = phase_rotation_diversity or 0
    amp_rot = amplitude_rotation_diversity or 0
    nnz = int((1 - sparse) * (2**num_qubits))
    s_thresh = config.DEFAULT.dd_sparsity_threshold
    amp_thresh = config.adaptive_dd_amplitude_rotation_threshold(num_qubits, sparsity)

    passes = (
        sparse >= s_thresh
        and nnz <= config.DEFAULT.dd_nnz_threshold
        and phase_rot <= config.DEFAULT.dd_phase_rotation_diversity_threshold
        and amp_rot <= amp_thresh
    )

    dd_metric = False
    metric_value: float | None = None
    if passes:
        s_score = sparse / s_thresh if s_thresh else 0.0
        nnz_score = 1 - nnz / config.DEFAULT.dd_nnz_threshold
        phase_score = 1 - (
            phase_rot / config.DEFAULT.dd_phase_rotation_diversity_threshold
            if config.DEFAULT.dd_phase_rotation_diversity_threshold
            else 0.0
        )
        amp_score = 1 - (amp_rot / amp_thresh if amp_thresh else 0.0)
        weight_sum = (
            config.DEFAULT.dd_sparsity_weight
            + config.DEFAULT.dd_nnz_weight
            + config.DEFAULT.dd_phase_rotation_weight
            + config.DEFAULT.dd_amplitude_rotation_weight
        )
        metric_value = (
            config.DEFAULT.dd_sparsity_weight * s_score
            + config.DEFAULT.dd_nnz_weight * nnz_score
            + config.DEFAULT.dd_phase_rotation_weight * phase_score
            + config.DEFAULT.dd_amplitude_rotation_weight * amp_score
        )
        metric_value = metric_value / weight_sum if weight_sum else 0.0
        dd_metric = metric_value >= config.DEFAULT.dd_metric_threshold
    else:
        reasons = []
        if sparse < s_thresh:
            reasons.append("sparsity below threshold")
        if nnz > config.DEFAULT.dd_nnz_threshold:
            reasons.append("nnz above threshold")
        if phase_rot > config.DEFAULT.dd_phase_rotation_diversity_threshold:
            reasons.append("phase diversity above threshold")
        if amp_rot > amp_thresh:
            reasons.append("amplitude diversity above threshold")
        diag["backends"][Backend.DECISION_DIAGRAM] = {
            "feasible": False,
            "reasons": reasons,
        }

    if dd_metric:
        dd_cost = estimator.decision_diagram(num_gates=num_gates, frontier=frontier)
        feasible = True
        reasons: list[str] = []
        if max_memory is not None and dd_cost.memory > max_memory:
            feasible = False
            reasons.append("memory > threshold")
        if max_time is not None and dd_cost.time > max_time:
            feasible = False
            reasons.append("time > threshold")
        entry: MutableMapping[str, object] = {
            "feasible": feasible,
            "reasons": reasons,
            "cost": dd_cost,
            "metric": metric_value,
            "dd_metric_threshold": config.DEFAULT.dd_metric_threshold,
        }
        diag["backends"][Backend.DECISION_DIAGRAM] = entry
        if feasible:
            candidates[Backend.DECISION_DIAGRAM] = dd_cost

    # ------------------------------------------------------------------
    # Matrix product state backend
    # ------------------------------------------------------------------
    if stats.is_local and num_qubits:
        chosen_chi = stats.chi
        if chosen_chi is None:
            chosen_chi = estimator.chi_max or 4
        chi_limit: int | None = None
        infeasible_chi = False
        if max_memory is not None:
            chi_limit = estimator.chi_from_memory(num_qubits, max_memory)
            if chi_limit <= 0:
                infeasible_chi = True
            else:
                max_chi = (
                    max(chosen_chi)
                    if isinstance(chosen_chi, Sequence) and not isinstance(chosen_chi, (str, bytes))
                    else int(chosen_chi)
                )
                if max_chi > chi_limit:
                    infeasible_chi = True
        mps_cost = estimator.mps(
            num_qubits,
            stats.num_1q_gates + stats.num_measurements,
            stats.num_2q_gates,
            chi=chosen_chi,
            svd=True,
        )
        feasible = not infeasible_chi
        reasons: list[str] = []
        if infeasible_chi:
            reasons.append("bond dimension exceeds memory limit")
        if max_memory is not None and mps_cost.memory > max_memory:
            feasible = False
            reasons.append("memory > threshold")
        if max_time is not None and mps_cost.time > max_time:
            feasible = False
            reasons.append("time > threshold")
        entry = {
            "feasible": feasible,
            "reasons": reasons,
            "cost": mps_cost,
            "chi": chosen_chi,
        }
        if chi_limit is not None:
            entry["chi_limit"] = chi_limit
        if target_accuracy is not None:
            entry["target_accuracy"] = target_accuracy
        diag["backends"][Backend.MPS] = entry
        if feasible:
            candidates[Backend.MPS] = mps_cost
    else:
        reason = "non-local gates" if stats.num_2q_gates else "no multi-qubit gates"
        diag["backends"][Backend.MPS] = {
            "feasible": False,
            "reasons": [reason],
        }

    # ------------------------------------------------------------------
    # Statevector backend
    # ------------------------------------------------------------------
    sv_cost = estimator.statevector(
        num_qubits,
        stats.num_1q_gates,
        stats.num_2q_gates,
        stats.num_measurements,
    )
    sv_feasible = True
    reasons: list[str] = []
    if max_memory is not None and sv_cost.memory > max_memory:
        sv_feasible = False
        reasons.append("memory > threshold")
    if max_time is not None and sv_cost.time > max_time:
        sv_feasible = False
        reasons.append("time > threshold")
    diag["backends"][Backend.STATEVECTOR] = {
        "feasible": sv_feasible,
        "reasons": reasons,
        "cost": sv_cost,
    }
    if sv_feasible:
        candidates[Backend.STATEVECTOR] = sv_cost

    if not candidates:
        diag["selected_backend"] = None
        diag["selected_cost"] = None
        return None, diag

    selected = min(candidates, key=lambda b: (candidates[b].memory, candidates[b].time))
    diag["selected_backend"] = selected
    diag["selected_cost"] = candidates[selected]
    for backend, entry in diag["backends"].items():
        if isinstance(entry, Mapping):
            entry["selected"] = backend == selected
    return selected, diag


def estimate_conversion(
    source: Backend,
    target: Backend,
    boundary: BoundarySpec,
    *,
    estimator: CostEstimator | None = None,
) -> ConversionEstimate:
    """Return the cheapest conversion primitive for the provided boundary."""

    estimator = estimator or CostEstimator()
    return estimator.conversion(
        source,
        target,
        boundary.num_qubits,
        boundary.rank,
        boundary.frontier,
        boundary.window,
        window_1q_gates=boundary.window_1q_gates,
        window_2q_gates=boundary.window_2q_gates,
        s_max=boundary.s_max,
        r_max=boundary.r_max,
        q_max=boundary.q_max,
    )


def aggregate_single_backend_plan(
    fragments: Sequence[tuple[Backend, Cost]]
) -> Cost:
    """Aggregate costs for a plan that uses a single backend."""

    total_time = sum(cost.time for _, cost in fragments)
    peak_memory = _update_peak_memory(0.0, (cost.memory for _, cost in fragments))
    log_depth = max((cost.log_depth for _, cost in fragments), default=0.0)
    conversion_time = sum(cost.conversion for _, cost in fragments)
    return Cost(
        time=total_time,
        memory=peak_memory,
        log_depth=log_depth,
        conversion=conversion_time,
    )


def aggregate_partitioned_plan(
    fragments: Sequence[tuple[Backend, Cost]],
    boundaries: Sequence[BoundarySpec],
    *,
    estimator: CostEstimator | None = None,
) -> MutableMapping[str, object]:
    """Aggregate costs for a heterogeneous plan with conversions.

    ``boundaries`` must contain ``len(fragments) - 1`` entries describing the
    interfaces between consecutive fragments.  Conversions are skipped when the
    neighbouring fragments already use the same backend.
    """

    if boundaries and len(boundaries) != max(len(fragments) - 1, 0):
        raise ValueError("number of boundaries must match fragment transitions")

    estimator = estimator or CostEstimator()
    total_time = 0.0
    peak_memory = 0.0
    log_depth = 0.0
    conversion_time = 0.0
    conversions: list[dict[str, object]] = []

    for backend, cost in fragments:
        total_time += cost.time
        peak_memory = _update_peak_memory(peak_memory, [cost.memory])
        log_depth = max(log_depth, cost.log_depth)
        conversion_time += cost.conversion

    for idx in range(len(fragments) - 1):
        src_backend, _ = fragments[idx]
        dst_backend, _ = fragments[idx + 1]
        if src_backend == dst_backend:
            continue
        spec = boundaries[idx]
        estimate = estimate_conversion(src_backend, dst_backend, spec, estimator=estimator)
        conversions.append(
            {
                "index": idx,
                "source": src_backend,
                "target": dst_backend,
                "primitive": estimate.primitive,
                "cost": estimate.cost,
            }
        )
        total_time += estimate.cost.time
        conversion_time += estimate.cost.time
        peak_memory = _update_peak_memory(peak_memory, [estimate.cost.memory])
        log_depth = max(log_depth, estimate.cost.log_depth)

    total_cost = Cost(
        time=total_time,
        memory=peak_memory,
        log_depth=log_depth,
        conversion=conversion_time,
    )
    return {
        "total_cost": total_cost,
        "fragments": list(fragments),
        "conversions": conversions,
    }


__all__ = [
    "BoundarySpec",
    "FragmentStats",
    "aggregate_partitioned_plan",
    "aggregate_single_backend_plan",
    "evaluate_fragment_backends",
    "estimate_conversion",
]
