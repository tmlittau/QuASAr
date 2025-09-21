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


def hybrid_dense_to_mps_circuit(
    *,
    ghz_width: int = 4,
    random_width: int = 5,
    qaoa_width: int = 5,
    qaoa_layers: int = 3,
    seed: int = 5,
) -> Circuit:
    """Dense random prefix followed by an MPS-friendly suffix.

    The circuit prepares a Clifford GHZ register, entangles it with a dense
    random block and finally executes a linear-chain QAOA routine that favours
    the matrix product state backend.  Cross-register entangling gates ensure
    that switching to the QAOA suffix requires an explicit conversion when
    ``conversion_cost_multiplier`` is small.  Larger multipliers make the
    planner keep the dense backend for the entire circuit, providing a compact
    example where the final method choice changes with ``alpha``.

    Parameters
    ----------
    ghz_width:
        Number of qubits in the initial Clifford GHZ prefix.  Must be at least
        three to expose an entanglement boundary.
    random_width:
        Width of the dense random block in the middle section.  Requires at
        least four qubits to generate a non-trivial dense region.
    qaoa_width:
        Number of qubits evolved by the linear-chain QAOA suffix.  A minimum of
        three qubits is required to form a non-trivial chain.
    qaoa_layers:
        Number of QAOA layers applied to the suffix.  Must be positive.
    seed:
        Seed controlling the random parameters of the dense and QAOA blocks to
        keep the benchmark deterministic.

    Returns
    -------
    Circuit
        Combined circuit whose final backend toggles between MPS and
        statevector depending on the conversion penalty.
    """

    if ghz_width < 3:
        raise ValueError("ghz_width must be at least three to expose entanglement boundaries")
    if random_width < 4:
        raise ValueError("random_width must be at least four to create a dense region")
    if qaoa_width < 3:
        raise ValueError("qaoa_width must be at least three for a linear chain")
    if qaoa_layers <= 0:
        raise ValueError("qaoa_layers must be positive")

    rng = Random(seed)

    gates: List[Gate] = []

    ghz = ghz_circuit(ghz_width)
    gates.extend(ghz.gates)

    dense_offset = ghz_width
    dense_first = dense_offset
    dense_last = dense_offset + random_width - 1

    # Couple the Clifford prefix to the dense random block to enforce a
    # conversion once non-Clifford gates appear.
    gates.append(Gate("CX", [ghz_width - 1, dense_first]))

    dense = random_circuit(random_width, seed=seed + 1)
    gates.extend(_shift_gates(dense.gates, dense_offset))

    qaoa_offset = dense_offset + random_width
    qaoa_first = qaoa_offset

    # Connect the dense block with the QAOA suffix using Clifford entanglers so
    # switching backends requires a conversion step.
    gates.append(Gate("CZ", [dense_last, qaoa_first]))

    for qubit in range(qaoa_width):
        gates.append(Gate("H", [qaoa_offset + qubit]))
    for _ in range(qaoa_layers):
        for qubit in range(qaoa_width - 1):
            theta = rng.uniform(0.2, 1.5)
            gates.append(
                Gate("RZZ", [qaoa_offset + qubit, qaoa_offset + qubit + 1], {"theta": theta})
            )
        bridge = rng.uniform(0.2, 1.5)
        gates.append(Gate("RZZ", [dense_last, qaoa_first], {"theta": bridge}))
        for qubit in range(qaoa_width):
            rx_angle = rng.uniform(0.1, 2.5)
            gates.append(Gate("RX", [qaoa_offset + qubit], {"theta": rx_angle}))

    for qubit in range(qaoa_width):
        ry_angle = rng.uniform(0.1, 1.2)
        gates.append(Gate("RY", [qaoa_offset + qubit], {"theta": ry_angle}))
        rz_angle = rng.uniform(0.1, 1.2)
        gates.append(Gate("RZ", [qaoa_offset + qubit], {"phi": rz_angle}))

    return Circuit(gates, use_classical_simplification=False)


def stim_to_dd_circuit(
    *,
    num_groups: int = 3,
    group_size: int = 4,
    entangling_layer: bool = False,
) -> Circuit:
    """Create disjoint GHZ subsystems that later require DD simulation.

    The circuit builds ``num_groups`` independent GHZ states using only
    Clifford gates so that the partitioner initially favours the tableau
    backend.  A global layer of ``T`` rotations then breaks the Clifford
    structure for every subsystem, forcing a conversion to the
    decision-diagram backend while the groups remain independent.  When
    ``entangling_layer`` is ``True`` the routine appends a final chain of
    CNOT gates that couples neighbouring subsystems *after* the ``T``
    layer, ensuring the conversion to decision diagrams happens on each
    group before any cross-entanglement is introduced.

    Parameters
    ----------
    num_groups:
        Number of identical stabiliser subsystems to prepare.  Must be at
        least one.
    group_size:
        Number of qubits per subsystem.  Requires at least three qubits to
        form a non-trivial GHZ state.
    entangling_layer:
        When ``True`` append a final layer of CNOT gates that connects the
        subsystems in a linear chain without disturbing the earlier
        independent evolution.

    Returns
    -------
    Circuit
        A circuit where the stabiliser preparation favours tableau
        simulation and the subsequent ``T`` layer triggers conversion to
        the decision-diagram backend.
    """

    if num_groups < 1:
        raise ValueError("num_groups must be at least one")
    if group_size < 3:
        raise ValueError("group_size must be at least three for GHZ preparation")

    gates: List[Gate] = []

    # Prepare independent GHZ states using only Clifford gates so the
    # partitioner groups them into tableau-friendly partitions.
    for group in range(num_groups):
        offset = group * group_size
        ghz = ghz_circuit(group_size)
        gates.extend(_shift_gates(ghz.gates, offset))

    # Apply identical non-Clifford rotations on each subsystem to force a
    # conversion from the tableau backend to the decision-diagram backend.
    for group in range(num_groups):
        base = group * group_size
        for qubit in range(group_size):
            gates.append(Gate("T", [base + qubit]))

    # Reapply a Clifford entangling chain within each subsystem so the
    # resulting DD partition maintains the original group boundaries while
    # still containing the non-Clifford ``T`` rotations.
    for group in range(num_groups):
        base = group * group_size
        for qubit in range(1, group_size):
            gates.append(Gate("CX", [base + qubit - 1, base + qubit]))

    # Optionally entangle neighbouring subsystems after the conversion
    # point so each group experiences the tableau->DD switch independently.
    if entangling_layer and num_groups > 1:
        for group in range(num_groups - 1):
            control = (group + 1) * group_size - 1
            target = (group + 1) * group_size
            gates.append(Gate("CX", [control, target]))

    return Circuit(gates, use_classical_simplification=False)


__all__ = ["mixed_backend_subsystems", "stim_to_dd_circuit"]
