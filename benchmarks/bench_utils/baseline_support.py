"""Shared helpers for estimating baseline costs and feasibility checks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Tuple

from quasar.cost import Backend, Cost
from quasar.metrics import FragmentMetrics
from quasar.planner import _simulation_cost

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

if __package__ in {None, ""}:  # pragma: no cover - script execution
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import circuits as circuit_lib  # type: ignore[no-redef]
    from memory_utils import max_qubits_statevector  # type: ignore[no-redef]
else:  # pragma: no cover - package import
    from . import circuits as circuit_lib
    from .memory_utils import max_qubits_statevector


def format_bytes(value: float) -> str:
    """Return ``value`` formatted as a human readable byte string."""

    value = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    index = 0
    while value >= 1024.0 and index < len(units) - 1:
        value /= 1024.0
        index += 1
    if index == 0:
        return f"{int(value)} {units[index]}"
    return f"{value:.2f} {units[index]}"


def format_seconds(value: float) -> str:
    """Return ``value`` formatted with a seconds suffix."""

    return f"{float(value):.2f} s"


def estimate_backend_cost(
    engine,
    circuit: object,
    backend: Backend,
) -> Tuple[Cost, FragmentMetrics]:
    """Return theoretical cost and fragment metrics for ``backend``."""

    planner = getattr(engine, "planner", None)
    estimator = getattr(planner, "estimator", None)
    gates = list(getattr(circuit, "gates", ()))
    metrics = FragmentMetrics.from_gates(gates)
    if estimator is None:
        return Cost(0.0, 0.0), metrics
    depth = getattr(circuit, "depth", None)
    cost = _simulation_cost(
        estimator,
        backend,
        metrics.num_qubits,
        metrics.num_1q,
        metrics.num_2q,
        metrics.num_meas,
        num_t_gates=metrics.num_t,
        depth=depth,
    )
    return cost, metrics


def baseline_support_status(
    backend: Backend,
    *,
    width: int,
    circuit: object | None,
    memory_bytes: int | None,
) -> Tuple[bool, str | None]:
    """Return whether ``backend`` can execute ``circuit`` and a skip reason."""

    if backend == Backend.STATEVECTOR:
        limit = max_qubits_statevector(memory_bytes)
        if width > limit:
            return (
                False,
                f"circuit width {width} exceeds statevector limit of {limit} qubits",
            )

    if backend == Backend.TABLEAU and circuit is not None:
        gates = getattr(circuit, "gates", ())
        forbidden = {"CCX", "CCZ", "MCX", "CSWAP"}
        for gate in gates:
            name = getattr(gate, "gate", "").upper()
            if name in forbidden:
                return False, f"{name} gate is unsupported by the tableau backend"
        try:
            is_clifford = circuit_lib.is_clifford(circuit)
        except Exception:  # pragma: no cover - defensive
            is_clifford = False
        if not is_clifford:
            return False, "non-Clifford gates are unsupported by the tableau backend"

    return True, None


__all__ = [
    "baseline_support_status",
    "estimate_backend_cost",
    "format_bytes",
    "format_seconds",
]

