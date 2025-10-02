"""Utilities for modelling dense statevector baselines.

This module provides lightweight helpers that mirror the behaviour requested by
QuASAr's stitched benchmark suite.  The helpers intentionally avoid any direct
backend dependencies so they can be used for both offline estimation and at
runtime when a dense simulator cannot be executed due to memory constraints.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

# ---- Gate counting ---------------------------------------------------------
ONE_Q = {
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "u",
    "u1",
    "u2",
    "u3",
    "p",
}
TWO_Q = {"cx", "cz", "swap", "crz", "cry", "crx", "cp", "cswap"}
DIAG_2Q = {"cz", "crz", "cp"}
THREE_Q = {"ccx", "ccz"}


def _norm(name: str) -> str:
    """Normalise a gate name for bucket classification."""

    return name.lower().replace("-", "").replace("_", "")


def count_gates(gates: Iterable[Any]) -> Dict[str, int]:
    """Count gates grouped into coarse buckets for SV cost modelling."""

    counts = {"1q": 0, "2q": 0, "diag2q": 0, "3q": 0, "other": 0, "total": 0}
    for gate in gates:
        name = _norm(getattr(gate, "name", getattr(gate, "gate", "other")))
        targets = getattr(gate, "qubits", getattr(gate, "targets", None))
        nq = len(targets or ())
        counts["total"] += 1
        if name in ONE_Q or nq == 1:
            counts["1q"] += 1
        elif name in THREE_Q or nq == 3:
            counts["3q"] += 1
        elif name in TWO_Q or nq == 2:
            counts["2q"] += 1
            if name in DIAG_2Q:
                counts["diag2q"] += 1
        else:
            counts["other"] += 1
    return counts


# ---- Statevector memory model ----------------------------------------------
def predict_sv_peak_bytes(
    n: int,
    *,
    dtype_bytes: int = 16,
    scratch_factor: float = 1.5,
) -> int:
    """Return the peak bytes required for an ``n`` qubit dense statevector."""

    amplitudes = 1 << n
    peak = amplitudes * dtype_bytes * scratch_factor
    return int(peak)


# ---- Statevector runtime model ---------------------------------------------
def predict_sv_runtime_au(
    n: int,
    counts: Dict[str, int],
    *,
    c_1q: float = 1.0,
    c_2q: float = 2.5,
    c_diag2q: float = 0.8,
    c_3q: float = 5.0,
    c_other: float = 2.0,
) -> float:
    """Return the theoretical runtime in arbitrary units for dense SV."""

    scale = float(1 << n)
    non_diag_2q = counts.get("2q", 0) - counts.get("diag2q", 0)
    value = (
        c_1q * counts.get("1q", 0)
        + c_2q * max(non_diag_2q, 0)
        + c_diag2q * counts.get("diag2q", 0)
        + c_3q * counts.get("3q", 0)
        + c_other * counts.get("other", 0)
    )
    return scale * value


# ---- Budget/OOM helpers ----------------------------------------------------
def will_sv_oom(
    n: int,
    mem_budget_bytes: int | None,
    *,
    dtype_bytes: int = 16,
    scratch_factor: float = 1.5,
) -> bool:
    """Return ``True`` when an ``n`` qubit SV would exceed ``mem_budget_bytes``."""

    if mem_budget_bytes is None or mem_budget_bytes <= 0:
        return False
    required = predict_sv_peak_bytes(
        n, dtype_bytes=dtype_bytes, scratch_factor=scratch_factor
    )
    return required > mem_budget_bytes


__all__ = [
    "count_gates",
    "predict_sv_peak_bytes",
    "predict_sv_runtime_au",
    "will_sv_oom",
]
