"""Heuristic symmetry metric for circuits."""

from __future__ import annotations

from collections import Counter

from .circuit import Circuit


def symmetry_score(circuit: Circuit) -> float:
    """Estimate the structural symmetry of ``circuit``.

    The heuristic scans the circuit's gate sequence and counts how many
    distinct gate types (including their parameters) are repeated.  The
    number of repeated gate specifications is then normalised by the
    total number of qubit layers (the circuit depth) to yield a score in
    ``[0, 1]``.

    Parameters
    ----------
    circuit:
        Circuit to analyse.

    Returns
    -------
    float
        Symmetry score where ``0`` indicates no repeated gate patterns and
        ``1`` represents high repetition relative to depth.
    """

    if circuit.depth == 0:
        return 0.0

    signatures = [
        (gate.gate, tuple(sorted(gate.params.items()))) for gate in circuit.gates
    ]
    counts = Counter(signatures)
    repeats = sum(1 for c in counts.values() if c > 1)
    return min(repeats / circuit.depth, 1.0)
