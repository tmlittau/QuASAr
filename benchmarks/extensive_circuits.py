from __future__ import annotations

"""Hybrid large-scale benchmark circuits for QuASAr.

This module provides generator functions that assemble wide circuits from
smaller algorithmic components.  They are intended to showcase QuASAr's
partitioning capabilities on circuits that mix structured subroutines with
substantial entanglement across distant qubit groups.
"""

from typing import List

from quasar.circuit import Circuit, Gate
from .circuits import ghz_circuit, qaoa_circuit, adder_circuit
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


__all__ = ["surface_code_qaoa_circuit", "adder_ghz_qaoa_circuit"]
