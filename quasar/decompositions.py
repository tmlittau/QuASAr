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


def decompose_mcx(controls: List[int], target: int) -> List["Gate"]:
    """Return a decomposition of an n-controlled X gate.

    The decomposition reduces an arbitrary multi-controlled X operation into a
    sequence of Toffoli gates which are further expanded into single- and
    two-qubit gates.  ``n``−2 ancillary qubits are introduced and uncomputed at
    the end of the sequence.  Ancillas are allocated using increasing qubit
    indices beyond the controls and target.

    Parameters
    ----------
    controls:
        List of control qubit indices.  The last qubit in the list is not the
        target but an additional control.
    target:
        Index of the target qubit.
    """

    from .circuit import Gate

    num_controls = len(controls)
    if num_controls == 0:
        return [Gate("X", [target])]
    if num_controls == 1:
        return [Gate("CX", [controls[0], target])]
    if num_controls == 2:
        return decompose_ccx(controls[0], controls[1], target)

    max_index = max(controls + [target])
    ancillas = [max_index + 1 + i for i in range(num_controls - 2)]
    gates: List[Gate] = []

    c1, c2 = controls[0], controls[1]
    gates.extend(decompose_ccx(c1, c2, ancillas[0]))
    for i in range(2, num_controls - 1):
        gates.extend(decompose_ccx(ancillas[i - 2], controls[i], ancillas[i - 1]))
    gates.extend(decompose_ccx(ancillas[-1], controls[-1], target))
    for i in reversed(range(2, num_controls - 1)):
        gates.extend(decompose_ccx(ancillas[i - 2], controls[i], ancillas[i - 1]))
    gates.extend(decompose_ccx(c1, c2, ancillas[0]))
    return gates


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


__all__ = ["decompose_ccx", "decompose_mcx", "decompose_ccz"]
