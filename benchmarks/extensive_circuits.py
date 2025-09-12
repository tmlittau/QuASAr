from __future__ import annotations

"""Hybrid large-scale benchmark circuits for QuASAr.

This module provides generator functions that assemble wide circuits from
smaller algorithmic components.  They are intended to showcase QuASAr's
partitioning capabilities on circuits that mix structured subroutines with
substantial entanglement across distant qubit groups.
"""

from typing import List

from quasar.circuit import Circuit, Gate
from .circuits import ghz_circuit, qft_circuit, qaoa_circuit, adder_circuit


def _shift_gates(gates: List[Gate], offset: int) -> List[Gate]:
    """Return ``gates`` with all qubit indices increased by ``offset``."""

    return [Gate(g.gate, [q + offset for q in g.qubits], dict(g.params)) for g in gates]


def dual_ghz_qft_circuit(width: int) -> Circuit:
    """Prepare two GHZ states and entangle them with a global QFT.

    Parameters
    ----------
    width:
        Number of qubits in each GHZ register.  The circuit operates on
        ``2 * width`` qubits in total.

    Returns
    -------
    Circuit
        Combined circuit generating two disjoint GHZ states followed by a
        quantum Fourier transform over all qubits.
    """

    if width <= 0:
        return Circuit([])

    ghz_a = ghz_circuit(width, use_classical_simplification=False)
    ghz_b = ghz_circuit(width, use_classical_simplification=False)
    gates = list(ghz_a.gates)
    gates.extend(_shift_gates(ghz_b.gates, width))
    qft = qft_circuit(2 * width, use_classical_simplification=False)
    gates.extend(qft.gates)
    return Circuit(gates, use_classical_simplification=False)


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
    ghz = ghz_circuit(bit_width, use_classical_simplification=False)

    gates = list(adder.gates)
    gates.extend(_shift_gates(ghz.gates, adder_qubits))
    total_qubits = adder_qubits + bit_width
    qaoa = qaoa_circuit(total_qubits, repetitions=qaoa_layers)
    gates.extend(qaoa.gates)
    return Circuit(gates, use_classical_simplification=False)


__all__ = ["dual_ghz_qft_circuit", "adder_ghz_qaoa_circuit"]
