from __future__ import annotations

"""Hybrid large-scale benchmark circuits for QuASAr.

This module provides generator functions that assemble wide circuits from
smaller algorithmic components.  They are intended to showcase QuASAr's
partitioning capabilities on circuits that mix structured subroutines with
substantial entanglement across distant qubit groups.
"""

from typing import List

from quasar.circuit import Circuit, Gate
from .circuits import ghz_circuit, qaoa_circuit, adder_circuit, grover_circuit
from .large_scale_circuits import surface_corrected_qaoa


def _shift_gates(gates: List[Gate], offset: int) -> List[Gate]:
    """Return ``gates`` with all qubit indices increased by ``offset``."""

    return [Gate(g.gate, [q + offset for q in g.qubits], dict(g.params)) for g in gates]


def surface_code_qaoa_circuit(
    bit_width: int, distance: int = 3, rounds: int = 1
) -> Circuit:
    """QAOA ring interleaved with surface-code cycles for hybrid partitioning.

    The returned circuit is constructed by
    :func:`benchmarks.large_scale_circuits.surface_corrected_qaoa` and mixes two
    distinct subroutines that QuASAr partitions onto different simulators.  Each
    QAOA layer contains low-degree ``RZZ``/``RX`` rotations on a cyclic register,
    encouraging an MPS backend, while the inserted surface-code cycles consist
    solely of Clifford operations on additional ancilla qubits, ideal for the
    tableau simulator.  Analysing the circuit therefore produces alternating
    partitions whose state descriptor repeatedly transitions from the MPS
    representation into the stabiliser tableau and back again.

    Parameters
    ----------
    bit_width:
        Number of problem qubits arranged on a cycle graph.
    distance:
        Code distance of the surface-code cycles.  The lattice must contain at
        least ``bit_width`` data qubits.
    rounds:
        Number of QAOA layers, each followed by one surface-code round.

    Returns
    -------
    Circuit
        Combined circuit interleaving QAOA dynamics with stabiliser correction
        layers.
    """

    return surface_corrected_qaoa(bit_width, distance, rounds)


def adder_ghz_qaoa_circuit(
    bit_width: int, qaoa_layers: int = 1, adder_kind: str = "vbe"
) -> Circuit:
    """Combine a ripple-carry adder, GHZ state and global QAOA layers.

    The circuit constructs a ``bit_width``-bit ripple-carry adder acting on two
    registers.  An independent GHZ state of the same size is prepared on a
    disjoint set of qubits.  All qubits then undergo ``qaoa_layers`` rounds of a
    ring-graph QAOA circuit.  The total qubit count is ``3 * bit_width + 2``.

    Parameters
    ----------
    bit_width:
        Number of bits in the adder operands and in the GHZ register.
    qaoa_layers:
        Number of QAOA problem/mixing layer pairs to apply across the full
        system.
    adder_kind:
        Variant of the ripple-carry adder.  Passed directly to
        :func:`adder_circuit`.

    Returns
    -------
    Circuit
        Complete circuit combining arithmetic, entangling and QAOA layers.
    """

    if bit_width <= 0 or qaoa_layers <= 0:
        return Circuit([])

    adder = adder_circuit(bit_width, kind=adder_kind)
    adder_qubits = 2 * bit_width + 2
    ghz = ghz_circuit(bit_width)

    gates = list(adder.gates)
    gates.extend(_shift_gates(ghz.gates, adder_qubits))
    total_qubits = adder_qubits + bit_width
    qaoa = qaoa_circuit(total_qubits, repetitions=qaoa_layers)
    gates.extend(qaoa.gates)
    return Circuit(gates)


def ghz_grover_fusion_circuit(
    ghz_qubits: int, grover_qubits: int, iterations: int = 1
) -> Circuit:
    """Prepare independent GHZ and Grover prefixes before a fusion entangler.

    The first register is initialised in a GHZ state, consisting entirely of
    Clifford operations that QuASAr assigns to the tableau backend.  In
    parallel, a Grover search routine runs on a second register shifted by
    ``ghz_qubits`` positions; its non-Clifford multi-controlled oracles remain
    on the statevector backend.  Because the prefixes touch disjoint qubit
    sets, the scheduler can execute them concurrently and only synchronises
    when the final cross-register ``CX`` fuses the two partitions into a single
    state descriptor.

    Parameters
    ----------
    ghz_qubits:
        Number of qubits in the GHZ register.
    grover_qubits:
        Number of qubits processed by the Grover search.
    iterations:
        Grover iterations applied to the second register.

    Returns
    -------
    Circuit
        Combined circuit that prepares both registers and entangles them once
        their prefixes finish executing on separate backends.
    """

    gates: List[Gate] = []

    if ghz_qubits > 0:
        ghz = ghz_circuit(ghz_qubits)
        gates.extend(ghz.gates)

    if grover_qubits > 0 and iterations > 0:
        grover = grover_circuit(grover_qubits, n_iterations=iterations)
        gates.extend(_shift_gates(grover.gates, ghz_qubits))

    if ghz_qubits > 0 and grover_qubits > 0 and iterations > 0:
        gates.append(Gate("CX", [ghz_qubits - 1, ghz_qubits]))

    return Circuit(gates)


def qaoa_toffoli_gadget_circuit(
    width: int, rounds_before: int = 1, rounds_after: int = 1
) -> Circuit:
    """Insert a central Toffoli gadget between QAOA layers to induce switching.

    The routine first applies ``rounds_before`` layers of the ring-graph QAOA
    ansatz on ``width`` qubits using :func:`qaoa_circuit`.  A single ``CCX`` gate
    then couples the three middle qubits, forcing QuASAr to migrate away from an
    MPS backend, before ``rounds_after`` additional QAOA layers resume the
    low-degree ``RZZ``/``RX`` pattern that remains MPS-suitable.

    Parameters
    ----------
    width:
        Number of qubits in the ring.  Must be at least three so that the
        central ``CCX`` operates on distinct qubits.
    rounds_before:
        Number of QAOA layers to apply before inserting the Toffoli gadget.
    rounds_after:
        Number of QAOA layers appended after the Toffoli gadget.

    Returns
    -------
    Circuit
        Full circuit combining QAOA evolution with the central Toffoli gadget.
    """

    if width <= 0:
        return Circuit([])
    if width < 3:
        raise ValueError("width must be at least three to place a CCX gadget")
    if rounds_before < 0 or rounds_after < 0:
        raise ValueError("QAOA round counts must be non-negative")

    gates: List[Gate] = []

    # Prefix QAOA layers, including the initial Hadamards even if zero rounds
    # are requested so that the gadget always follows a uniform superposition.
    prefix = qaoa_circuit(width, repetitions=rounds_before)
    gates.extend(prefix.gates)

    middle = width // 2
    ccx_qubits = [middle - 1, middle, middle + 1]
    gates.append(Gate("CCX", ccx_qubits))

    if rounds_after > 0:
        suffix = list(qaoa_circuit(width, repetitions=rounds_after).gates)
        # Remove the initial Hadamard layer from the appended QAOA circuit to
        # avoid duplicating it around the Toffoli gadget.
        suffix = suffix[width:]
        gates.extend(suffix)

    return Circuit(gates)


__all__ = [
    "surface_code_qaoa_circuit",
    "adder_ghz_qaoa_circuit",
    "ghz_grover_fusion_circuit",
    "qaoa_toffoli_gadget_circuit",
]
