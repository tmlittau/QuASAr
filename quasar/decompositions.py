"""Gate decomposition utilities for QuASAr.

This module provides helpers for expanding multi-controlled gates into
sequences of supported elementary operations. Currently only the Toffoli
(``CCX``) gate is implemented.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .circuit import Gate


def decompose_ccx(control1: int, control2: int, target: int) -> List["Gate"]:
    """Return a decomposition of a Toffoli (``CCX``) gate.

    The returned sequence consists solely of single- and two-qubit gates
    (``H``, ``T``, ``CX`` and ``TDG``) as commonly used for a fault-tolerant
    implementation of the Toffoli gate.

    Parameters
    ----------
    control1, control2, target:
        Indices of the first control, second control and target qubits.
    """

    from .circuit import Gate

    return [
        Gate("H", [target]),
        Gate("CX", [control2, target]),
        Gate("TDG", [target]),
        Gate("CX", [control1, target]),
        Gate("T", [target]),
        Gate("CX", [control2, target]),
        Gate("TDG", [target]),
        Gate("CX", [control1, target]),
        Gate("T", [control2]),
        Gate("T", [target]),
        Gate("H", [target]),
        Gate("CX", [control1, control2]),
        Gate("T", [control1]),
        Gate("TDG", [control2]),
        Gate("CX", [control1, control2]),
    ]


__all__ = ["decompose_ccx"]
