"""Gate decomposition utilities for QuASAr.

This module provides helpers for expanding multi-controlled gates into
sequences of supported elementary operations. Currently the Toffoli
(``CCX``) and controlled-controlled-Z (``CCZ``) gates are implemented.
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


def decompose_ccz(control1: int, control2: int, target: int) -> List["Gate"]:
    r"""Return a decomposition of a controlled-controlled-Z (``CCZ``) gate.

    The implementation follows the standard relation

    .. math:: CCZ = H_t \cdot CCX \cdot H_t,

    where ``CCX`` is decomposed into a sequence of single- and two-qubit
    gates.  The resulting gate list therefore contains only operations that
    are natively supported by the various backends (``H``, ``T``, ``TDG`` and
    ``CX``).

    Parameters
    ----------
    control1, control2, target:
        Indices of the first control, second control and target qubits.
    """

    from .circuit import Gate

    # ``CCZ`` can be implemented as ``H`` on the target, followed by a
    # ``CCX`` and another ``H``.  We purposely keep the full Toffoli
    # decomposition including its leading Hadamard to preserve the correct
    # phase relationships, even though this results in two consecutive ``H``
    # gates at the start of the sequence.
    ccx_sequence = decompose_ccx(control1, control2, target)
    return [Gate("H", [target]), *ccx_sequence, Gate("H", [target])]


__all__ = ["decompose_ccx", "decompose_ccz"]
