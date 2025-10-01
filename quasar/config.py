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


def _bool_from_env(name: str, default: bool) -> bool:
    """Return a boolean value parsed from the environment."""

    val = os.getenv(name)
    if val is None or not val.strip():
        return default
    val = val.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
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


def _default_parallel_backends() -> List[Backend]:
    base: List[Backend] = [Backend.STATEVECTOR]
    # Merge support for the MPS backend is covered by dedicated tests.
    base.append(Backend.MPS)
    return _backends_from_env("QUASAR_PARALLEL_BACKENDS", base)


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
            [
                Backend.MPS,
                Backend.DECISION_DIAGRAM,
                Backend.EXTENDED_STABILIZER,
                Backend.STATEVECTOR,
                Backend.TABLEAU,
            ],
        )
    )
    parallel_backends: List[Backend] = field(default_factory=_default_parallel_backends)
    mps_target_fidelity: float = _float_from_env(
        "QUASAR_MPS_TARGET_FIDELITY", 1.0
    )
    mps_long_range_fraction_threshold: float = _float_from_env(
        "QUASAR_MPS_LONG_RANGE_FRACTION_THRESHOLD", 0.35
    )
    mps_long_range_extent_threshold: float = _float_from_env(
        "QUASAR_MPS_LONG_RANGE_EXTENT_THRESHOLD", 0.25
    )
    mps_locality_strict_qubits: int = _int_from_env(
        "QUASAR_MPS_LOCALITY_STRICT_QUBITS", 6
    )
    mps_locality_strict_distance: int = _int_from_env(
        "QUASAR_MPS_LOCALITY_STRICT_DISTANCE", 4
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
    dd_phase_rotation_weight: float = _float_from_env(
        "QUASAR_DD_PHASE_ROTATION_WEIGHT", 1.0
    )
    dd_amplitude_rotation_weight: float = _float_from_env(
        "QUASAR_DD_AMPLITUDE_ROTATION_WEIGHT", 1.0
    )
    dd_metric_threshold: float = _float_from_env(
        "QUASAR_DD_METRIC_THRESHOLD", 0.8
    )
    dd_phase_rotation_diversity_threshold: int = _int_from_env(
        "QUASAR_DD_PHASE_ROTATION_DIVERSITY_THRESHOLD", 16
    )
    dd_amplitude_rotation_diversity_threshold: int = _int_from_env(
        "QUASAR_DD_AMPLITUDE_ROTATION_DIVERSITY_THRESHOLD", 16
    )
    backend_selection_log: str | None = os.getenv("QUASAR_BACKEND_SELECTION_LOG")
    verbose_selection: bool = _bool_from_env("QUASAR_VERBOSE_SELECTION", False)
    coeff_ema_decay: float = _float_from_env("QUASAR_COEFF_EMA_DECAY", 0.5)
    st_chi_cap: int | None = _int_from_env("QUASAR_ST_CHI_CAP", 16)


# Global configuration instance used when modules import ``quasar.config``.
DEFAULT = Config()


def adaptive_dd_amplitude_rotation_threshold(
    n_qubits: int, sparsity: float | None = None
) -> int:
    """Return the amplitude rotation diversity limit for a circuit.

    The base threshold ``DEFAULT.dd_amplitude_rotation_diversity_threshold``
    is scaled with circuit width and sparsity.  Extremely sparse circuits such
    as W states therefore retain the decision diagram backend even when they
    contain many distinct rotation angles.

    Args:
        n_qubits: Number of qubits in the circuit segment.
        sparsity: Estimated sparsity of the segment.  When provided the
            threshold grows with the estimated number of non-zero amplitudes.

    Returns:
        An adjusted diversity threshold.
    """

    base = DEFAULT.dd_amplitude_rotation_diversity_threshold
    if sparsity is None:
        return max(base, n_qubits)
    nnz = int((1 - sparsity) * (2**n_qubits))
    return max(base, nnz)
