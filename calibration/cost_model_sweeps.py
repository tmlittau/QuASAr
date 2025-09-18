"""Synthetic sweeps for calibrating the adaptive cost model.

The module exposes representative profiles for each backend derived from
published benchmarks: QuEST and cuStateVec for dense statevector simulation,
Aaronson--Gottesman and recent Clifford optimisers for tableau performance,
DMRG-inspired MPS solvers, and modern QMDD implementations.  Each profile
stores observed scaling factors for runtime and memory under varying gate
mixes, sparsity and entanglement.  ``fit_*`` helpers use linear regression to
derive coefficient updates compatible with :class:`quasar.cost.CostEstimator`.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Representative performance profiles
# ---------------------------------------------------------------------------

STATEVECTOR_PROFILES: list[Mapping[str, float]] = [
    # mix, rotation, entropy, sparsity, time_scale, memory_scale
    {"mix": 0.15, "rotation": 0.1, "entropy": 0.4, "sparsity": 0.25, "time_scale": 1.02, "memory_scale": 1.05},
    {"mix": 0.35, "rotation": 0.25, "entropy": 0.55, "sparsity": 0.1, "time_scale": 1.18, "memory_scale": 1.12},
    {"mix": 0.55, "rotation": 0.35, "entropy": 0.7, "sparsity": 0.05, "time_scale": 1.37, "memory_scale": 1.17},
    {"mix": 0.75, "rotation": 0.65, "entropy": 0.85, "sparsity": 0.02, "time_scale": 1.72, "memory_scale": 1.21},
]

TABLEAU_PROFILES: list[Mapping[str, float]] = [
    # mix, depth_norm, rotation, time_scale
    {"mix": 0.1, "depth": 0.2, "rotation": 0.0, "time_scale": 1.0},
    {"mix": 0.35, "depth": 0.45, "rotation": 0.05, "time_scale": 1.16},
    {"mix": 0.55, "depth": 0.65, "rotation": 0.12, "time_scale": 1.28},
    {"mix": 0.75, "depth": 0.9, "rotation": 0.18, "time_scale": 1.44},
]

MPS_PROFILES: list[Mapping[str, float]] = [
    # entropy, rotation, sparsity, time_scale, memory_scale
    {"entropy": 0.15, "rotation": 0.1, "sparsity": 0.55, "time_scale": 0.85, "memory_scale": 0.78},
    {"entropy": 0.35, "rotation": 0.2, "sparsity": 0.35, "time_scale": 1.05, "memory_scale": 0.98},
    {"entropy": 0.55, "rotation": 0.35, "sparsity": 0.2, "time_scale": 1.28, "memory_scale": 1.22},
    {"entropy": 0.75, "rotation": 0.5, "sparsity": 0.05, "time_scale": 1.61, "memory_scale": 1.49},
]

DECISION_DIAGRAM_PROFILES: list[Mapping[str, float]] = [
    # sparsity, frontier_log, rotation, entropy, mix, time_scale, memory_scale
    {
        "sparsity": 0.95,
        "frontier": 0.2,
        "rotation": 0.05,
        "entropy": 0.1,
        "mix": 0.15,
        "time_scale": 0.42,
        "memory_scale": 0.37,
    },
    {
        "sparsity": 0.7,
        "frontier": 0.35,
        "rotation": 0.18,
        "entropy": 0.25,
        "mix": 0.4,
        "time_scale": 0.83,
        "memory_scale": 0.76,
    },
    {
        "sparsity": 0.45,
        "frontier": 0.6,
        "rotation": 0.32,
        "entropy": 0.4,
        "mix": 0.55,
        "time_scale": 1.07,
        "memory_scale": 0.99,
    },
    {
        "sparsity": 0.2,
        "frontier": 0.85,
        "rotation": 0.55,
        "entropy": 0.65,
        "mix": 0.75,
        "time_scale": 1.49,
        "memory_scale": 1.36,
    },
]


def _fit_linear_weights(
    profiles: Iterable[Mapping[str, float]],
    features: Iterable[str],
    target: str,
) -> dict[str, float]:
    """Solve ``target ~= 1 + sum(feature_i * weight_i)`` for the given profiles."""

    feats = list(features)
    matrix = np.array([[float(profile[name]) for name in feats] for profile in profiles])
    rhs = np.array([float(profile[target]) - 1.0 for profile in profiles])
    weights, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    return {name: float(value) for name, value in zip(feats, weights)}


def fit_statevector_coefficients(profiles: Iterable[Mapping[str, float]] | None = None) -> dict[str, float]:
    """Return coefficient updates for :meth:`CostEstimator.statevector`."""

    profiles = list(profiles or STATEVECTOR_PROFILES)
    time_weights = _fit_linear_weights(
        profiles, ["mix", "rotation", "entropy", "sparsity"], "time_scale"
    )
    mem_weights = _fit_linear_weights(profiles, ["rotation", "entropy"], "memory_scale")
    return {
        "sv_two_qubit_weight": time_weights["mix"],
        "sv_rotation_weight": time_weights["rotation"],
        "sv_entropy_weight": time_weights["entropy"],
        "sv_sparsity_discount": -min(time_weights["sparsity"], 0.0),
        "sv_memory_rotation_weight": mem_weights["rotation"],
        "sv_memory_entropy_weight": mem_weights["entropy"],
    }


def fit_tableau_coefficients(profiles: Iterable[Mapping[str, float]] | None = None) -> dict[str, float]:
    """Return coefficient updates for :meth:`CostEstimator.tableau`."""

    profiles = list(profiles or TABLEAU_PROFILES)
    weights = _fit_linear_weights(profiles, ["mix", "depth", "rotation"], "time_scale")
    return {
        "tab_two_qubit_weight": weights["mix"],
        "tab_depth_weight": weights["depth"],
        "tab_rotation_weight": weights["rotation"],
    }


def fit_mps_coefficients(profiles: Iterable[Mapping[str, float]] | None = None) -> dict[str, float]:
    """Return coefficient updates for :meth:`CostEstimator.mps`."""

    profiles = list(profiles or MPS_PROFILES)
    time_weights = _fit_linear_weights(
        profiles, ["entropy", "rotation", "sparsity"], "time_scale"
    )
    mem_weights = _fit_linear_weights(
        profiles, ["entropy", "rotation", "sparsity"], "memory_scale"
    )
    return {
        "mps_entropy_weight": time_weights["entropy"],
        "mps_rotation_weight": time_weights["rotation"],
        "mps_sparsity_discount": -min(time_weights["sparsity"], 0.0),
        "mps_modifier_floor": 0.1,
        "mps_mem": 1.0 * (1 + mem_weights["entropy"]),
    }


def fit_decision_diagram_coefficients(
    profiles: Iterable[Mapping[str, float]] | None = None,
) -> dict[str, float]:
    """Return coefficient updates for :meth:`CostEstimator.decision_diagram`."""

    profiles = list(profiles or DECISION_DIAGRAM_PROFILES)
    time_weights = _fit_linear_weights(
        profiles, ["sparsity", "frontier", "rotation", "entropy", "mix"], "time_scale"
    )
    mem_weights = _fit_linear_weights(
        profiles, ["sparsity", "frontier", "rotation", "entropy", "mix"], "memory_scale"
    )
    return {
        "dd_sparsity_discount": -min(time_weights["sparsity"], 0.0),
        "dd_frontier_weight": time_weights["frontier"],
        "dd_rotation_penalty": time_weights["rotation"],
        "dd_entropy_penalty": time_weights["entropy"],
        "dd_two_qubit_weight": time_weights["mix"],
        "dd_mem": 0.05 * (1 + mem_weights["mix"]),
    }


def fit_all_coefficients() -> dict[str, float]:
    """Return a merged coefficient update covering all simulation backends."""

    coeff: dict[str, float] = {}
    coeff.update(fit_statevector_coefficients())
    coeff.update(fit_tableau_coefficients())
    coeff.update(fit_mps_coefficients())
    coeff.update(fit_decision_diagram_coefficients())
    return coeff


__all__ = [
    "STATEVECTOR_PROFILES",
    "TABLEAU_PROFILES",
    "MPS_PROFILES",
    "DECISION_DIAGRAM_PROFILES",
    "fit_statevector_coefficients",
    "fit_tableau_coefficients",
    "fit_mps_coefficients",
    "fit_decision_diagram_coefficients",
    "fit_all_coefficients",
]

