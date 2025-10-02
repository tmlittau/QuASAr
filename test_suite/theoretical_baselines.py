from __future__ import annotations
from typing import Dict


def predict_sv_peak_bytes(n: int, dtype_bytes: int = 16, scratch_factor: float = 1.5) -> int:
    amps = 1 << n  # 2**n
    return int(amps * dtype_bytes * scratch_factor)


def predict_sv_runtime_au(
    n: int,
    gate_counts: Dict[str, int],
    c_1q: float = 1.0,
    c_2q: float = 2.5,
    c_diag2q: float = 0.8,
    c_3q: float = 5.0,
    c_other: float = 2.0,
) -> float:
    scale = float(1 << n)
    return scale * (
        c_1q * gate_counts.get("1q", 0)
        + c_2q * (gate_counts.get("2q", 0) - gate_counts.get("diag2q", 0))
        + c_diag2q * gate_counts.get("diag2q", 0)
        + c_3q * gate_counts.get("3q", 0)
        + c_other * gate_counts.get("other", 0)
    )


def will_sv_oom(
    n: int,
    mem_budget_bytes: int,
    dtype_bytes: int = 16,
    scratch_factor: float = 1.5,
) -> bool:
    return predict_sv_peak_bytes(n, dtype_bytes, scratch_factor) > mem_budget_bytes


# Conversion cost: depends only on n (your assumption). Calibrate (a,b,p).
def predict_conversion_time_au(
    n: int,
    a: float = 2e-6,
    b: float = 2e-4,
    p: float = 2.0,
) -> float:
    return a * (n ** p) + b * n
