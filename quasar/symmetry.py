"""Symmetry heuristics for quantum circuits."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    from .circuit import Circuit, Gate


def _gate_key(gate: "Gate") -> Tuple[str, Tuple[Tuple[str, object], ...]]:
    """Return a hashable description of a gate's type and parameters."""
    return gate.gate, tuple(sorted(gate.params.items()))


def symmetry_score(circuit: "Circuit") -> float:
    """Compute a simple symmetry score for ``circuit``.

    The score measures the fraction of gates that share their type and
    parameter values with at least one other gate in the circuit. Higher
    scores indicate more structural repetition across qubits and layers.

    Parameters
    ----------
    circuit:
        Circuit to analyse.

    Returns
    -------
    float
        Symmetry score in the range ``[0, 1]``.
    """

    gates = getattr(circuit, "gates", None)
    if not gates:
        return 1.0

    counts = Counter(_gate_key(g) for g in gates)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(gates)
