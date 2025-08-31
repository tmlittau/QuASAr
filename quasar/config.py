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


@dataclass
class Config:
    """Runtime configuration defaults for QuASAr.

    Values may be overridden via environment variables or by supplying
    explicit arguments to :class:`Planner` and :class:`Scheduler`.
    """

    quick_max_qubits: int | None = _int_from_env("QUASAR_QUICK_MAX_QUBITS", 25)
    quick_max_gates: int | None = _int_from_env("QUASAR_QUICK_MAX_GATES", 200)
    quick_max_depth: int | None = _int_from_env("QUASAR_QUICK_MAX_DEPTH", 50)
    preferred_backend_order: List[Backend] = field(
        default_factory=lambda: _order_from_env(
            "QUASAR_BACKEND_ORDER",
            [Backend.MPS, Backend.DECISION_DIAGRAM, Backend.STATEVECTOR, Backend.TABLEAU],
        )
    )


# Global configuration instance used when modules import ``quasar.config``.
DEFAULT = Config()
