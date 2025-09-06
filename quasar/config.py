import os
from dataclasses import dataclass, field
from typing import List

from .cost import Backend


def _int_from_env(name: str, default: int | None) -> int | None:
    val = os.getenv(name)
    if val is None:
        return default
    if val.lower() == "none":
        return None
    try:
        return int(val)
    except ValueError:
        return default


def _float_from_env(name: str, default: float) -> float:
    """Return a floating-point value parsed from the environment."""

    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _order_from_env(name: str, default: List[Backend]) -> List[Backend]:
    val = os.getenv(name)
    if val is None or not val.strip():
        return list(default)
    order: List[Backend] = []
    for item in val.split(","):
        item = item.strip().upper()
        if not item:
            continue
        try:
            order.append(Backend[item])
        except KeyError:
            continue
    return order or list(default)


def _backends_from_env(name: str, default: List[Backend]) -> List[Backend]:
    """Return a list of backends parsed from the comma-separated env var."""
    return _order_from_env(name, default)


@dataclass
class Config:
    """Runtime configuration defaults for QuASAr.

    Values may be overridden via environment variables or by supplying
    explicit arguments to :class:`Planner` and :class:`Scheduler`.
    """

    quick_max_qubits: int | None = _int_from_env("QUASAR_QUICK_MAX_QUBITS", None)
    quick_max_gates: int | None = _int_from_env("QUASAR_QUICK_MAX_GATES", None)
    quick_max_depth: int | None = _int_from_env("QUASAR_QUICK_MAX_DEPTH", None)
    preferred_backend_order: List[Backend] = field(
        default_factory=lambda: _order_from_env(
            "QUASAR_BACKEND_ORDER",
            [Backend.MPS, Backend.DECISION_DIAGRAM, Backend.STATEVECTOR, Backend.TABLEAU],
        )
    )
    parallel_backends: List[Backend] = field(
        default_factory=lambda: _backends_from_env(
            "QUASAR_PARALLEL_BACKENDS",
            [Backend.STATEVECTOR, Backend.MPS],
        )
    )
    mps_target_fidelity: float = _float_from_env(
        "QUASAR_MPS_TARGET_FIDELITY", 1.0
    )
    dd_sparsity_threshold: float = _float_from_env(
        "QUASAR_DD_SPARSITY_THRESHOLD", 0.8
    )
    dd_nnz_threshold: int = _int_from_env(
        "QUASAR_DD_NNZ_THRESHOLD", 1_000_000
    )
    dd_sparsity_weight: float = _float_from_env(
        "QUASAR_DD_SPARSITY_WEIGHT", 1.0
    )
    dd_nnz_weight: float = _float_from_env(
        "QUASAR_DD_NNZ_WEIGHT", 1.0
    )
    dd_rotation_weight: float = _float_from_env(
        "QUASAR_DD_ROTATION_WEIGHT", 1.0
    )
    dd_metric_threshold: float = _float_from_env(
        "QUASAR_DD_METRIC_THRESHOLD", 0.8
    )
    dd_rotation_diversity_threshold: int = _int_from_env(
        "QUASAR_DD_ROTATION_DIVERSITY_THRESHOLD", 16
    )


# Global configuration instance used when modules import ``quasar.config``.
DEFAULT = Config()
