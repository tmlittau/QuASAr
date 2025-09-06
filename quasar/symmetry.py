"""Heuristic symmetry metric for circuits."""

from __future__ import annotations

from collections import Counter


PHASE_ROTATION_GATES = {"RZ", "P", "PHASE", "CP", "CRZ"}
AMPLITUDE_ROTATION_GATES = {"RY", "CRY", "RX", "CRX"}

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



def _rotation_diversity(circuit: Circuit, gate_set: set[str]) -> int:
    """Count distinct rotation parameters for the given ``gate_set``."""

    values = set()
    for gate in circuit.gates:
        name = gate.gate.upper()
        if name not in gate_set:
            continue
        val = None
        for param in gate.params.values():
            if isinstance(param, (int, float)):
                val = float(param)
                break
        if val is not None:
            values.add(round(val, 12))
    return len(values)


def phase_rotation_diversity(circuit: Circuit) -> int:
    """Count distinct phase/Z rotation parameters."""

    return _rotation_diversity(circuit, PHASE_ROTATION_GATES)


def amplitude_rotation_diversity(circuit: Circuit) -> int:
    """Count distinct X/Y rotation parameters."""

    return _rotation_diversity(circuit, AMPLITUDE_ROTATION_GATES)


# Backward compatibility: old name refers to phase rotations only
rotation_diversity = phase_rotation_diversity
