"""Heuristics for estimating circuit sparsity."""

from __future__ import annotations

from .circuit import Circuit, Gate

BRANCHING_GATES = {"H", "RY", "RX", "U", "U2", "U3"}


def is_controlled(gate: Gate) -> bool:
    """Return ``True`` if ``gate`` is a controlled operation.

    The helper simply checks whether the gate name is prefixed with one or more
    ``"C"`` characters, which is how controlled gates are represented within
    QuASAr's gate descriptions (e.g. ``CX`` for a controlled‑X and ``CRY`` for a
    controlled‑RY rotation).
    """

    return gate.gate.startswith("C")


def sparsity_estimate(circuit: Circuit) -> float:
    """Estimate the expected sparsity of a circuit's state vector.

    This heuristic tracks an approximation ``nnz`` of the number of non‑zero
    amplitudes generated when the circuit acts on ``|0…0>``.  Uncontrolled
    *branching* gates—Hadamard and generic single‑qubit rotations—are assumed to
    double ``nnz``.  Controlled versions of those gates add a single extra
    amplitude because the branch only occurs when the control qubit is non‑zero.

    The count is clamped to the dimension of the state space ``2**n`` (where ``n``
    is ``circuit.num_qubits``) and the returned value is ``1 - nnz / 2**n``: the
    estimated fraction of zero amplitudes.

    The estimate ignores interference effects between branches and assumes
    independence between successive branching operations, so it should be treated
    purely as an inexpensive heuristic.
    """

    nnz = 1
    full_dim = 2 ** circuit.num_qubits
    for gate in circuit.gates:
        base_gate = gate.gate.lstrip("C")
        controlled = is_controlled(gate)
        if base_gate in BRANCHING_GATES:
            if controlled:
                nnz += 1
            else:
                nnz *= 2
        if nnz > full_dim:
            nnz = full_dim
    return 1 - nnz / full_dim
