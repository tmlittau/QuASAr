from __future__ import annotations

"""Composite benchmark circuits targeting multiple simulation backends."""

from random import Random
from typing import Iterable, List

from quasar.circuit import Circuit, Gate

from .circuits import ghz_circuit, random_circuit


def _shift_gates(gates: Iterable[Gate], offset: int) -> List[Gate]:
    """Return a list of gates with qubit indices increased by ``offset``."""

    return [Gate(g.gate, [q + offset for q in g.qubits], dict(g.params)) for g in gates]


def mixed_backend_subsystems(
    *,
    ghz_width: int = 4,
    qaoa_width: int = 4,
    qaoa_layers: int = 2,
    random_width: int = 4,
    seed: int = 7,
) -> Circuit:
    """Combine GHZ, QAOA and dense random blocks to stress backend switching.

    The circuit prepares three contiguous subsystems that naturally favour
    different simulators:

    ``ghz_width`` Clifford-only gates initialise a GHZ state suitable for the
    tableau backend.  ``qaoa_width`` qubits then execute a low-entanglement linear-chain
    QAOA routine which the method selector maps to the MPS backend.  Finally, a
    dense non-local block on ``random_width`` qubits mixes random rotations with
    a Toffoli gadget, forcing the selector towards the statevector (or
    decision-diagram) backend.  Cross-partition entangling gates connect the
    three regions to trigger explicit conversion layers during planning.

    Parameters
    ----------
    ghz_width:
        Number of qubits in the Clifford GHZ prefix.  Must be at least three so
        that multiple entangling connectors can be placed.
    qaoa_width:
        Number of qubits evolved by the QAOA block.  Requires a minimum of
        three qubits to form a non-trivial linear topology.
    qaoa_layers:
        Number of QAOA layers applied to the MPS-friendly block.  A positive
        integer is required.
    random_width:
        Number of qubits used for the dense random suffix.  At least four
        qubits are required to host the Toffoli gadget and non-local ZZ
        rotations that promote dense backends.
    seed:
        Seed controlling the random parameters of the QAOA and dense random
        blocks to keep the benchmark deterministic.

    Returns
    -------
    Circuit
        Combined circuit spanning all three subsystems with explicit conversion
        boundaries.
    """

    if ghz_width < 3:
        raise ValueError("ghz_width must be at least three to expose entanglement boundaries")
    if qaoa_width < 3:
        raise ValueError("qaoa_width must be at least three for a ring QAOA block")
    if qaoa_layers <= 0:
        raise ValueError("qaoa_layers must be positive")
    if random_width < 4:
        raise ValueError("random_width must be at least four to host dense gadgets")

    rng = Random(seed)

    gates: List[Gate] = []

    ghz = ghz_circuit(ghz_width)
    gates.extend(ghz.gates)

    qaoa_offset = ghz_width
    qaoa_first = qaoa_offset
    qaoa_last = qaoa_offset + qaoa_width - 1

    # Entangle the GHZ register with the upcoming QAOA block to force a
    # conversion boundary once non-Clifford gates appear on the QAOA qubits.
    gates.append(Gate("CX", [ghz_width - 1, qaoa_first]))

    # Construct a linear-chain QAOA block so the selector favours the MPS backend.
    for qubit in range(qaoa_width):
        gates.append(Gate("H", [qaoa_offset + qubit]))
    for _ in range(qaoa_layers):
        for qubit in range(qaoa_width - 1):
            zz_angle = rng.uniform(0.2, 2.8)
            gates.append(
                Gate("RZZ", [qaoa_offset + qubit, qaoa_offset + qubit + 1], {"theta": zz_angle})
            )
        bridge_theta = rng.uniform(0.25, 1.35)
        gates.append(Gate("RZZ", [ghz_width - 1, qaoa_first], {"theta": bridge_theta}))
        for qubit in range(qaoa_width):
            rx_angle = rng.uniform(0.1, 2.6)
            gates.append(Gate("RX", [qaoa_offset + qubit], {"theta": rx_angle}))

    # Local single-qubit rotations increase amplitude diversity and keep the block firmly in the MPS regime.
    for qubit in range(qaoa_width):
        ry_angle = rng.uniform(0.15, 1.35)
        gates.append(Gate("RY", [qaoa_offset + qubit], {"theta": ry_angle}))
        rz_angle = rng.uniform(0.1, 1.25)
        gates.append(Gate("RZ", [qaoa_offset + qubit], {"phi": rz_angle}))

    final_theta = rng.uniform(0.2, 1.2)
    gates.append(Gate("RZZ", [qaoa_last - 1, qaoa_last], {"theta": final_theta}))

    random_offset = qaoa_offset + qaoa_width
    random_first = random_offset
    random_last = random_offset + random_width - 1

    # Connect the QAOA block with the dense suffix using Clifford entanglers so
    # the partitioner must convert the shared boundary when switching backends.
    gates.append(Gate("CX", [qaoa_last, random_first]))
    zz_bridge = rng.uniform(0.3, 1.4)
    gates.append(Gate("RZZ", [qaoa_last, random_first], {"theta": zz_bridge}))

    dense = random_circuit(random_width, seed=seed + 1)
    gates.extend(_shift_gates(dense.gates, random_offset))

    # Reinforce the dense nature of the suffix with explicit non-local
    # interactions and a Toffoli gadget which decomposes into T gates and
    # long-range CNOTs.  The angles are deterministic but non-Clifford.
    zz_angle = rng.uniform(0.35, 1.45)
    gates.append(Gate("RZZ", [random_first, random_last], {"theta": zz_angle}))
    ccx_control_b = random_offset + random_width // 2
    gates.append(Gate("CCX", [random_first, ccx_control_b, random_last]))
    crz_angle = rng.uniform(0.2, 1.1)
    gates.append(Gate("CRZ", [random_first + 1, random_last - 1], {"phi": crz_angle}))

    return Circuit(gates, use_classical_simplification=False)


__all__ = ["mixed_backend_subsystems"]
