"""Larger benchmark circuits composed from arithmetic primitives.

This module provides circuits intended to stress-test QuASAr's
partitioning and simulation on wider registers.  The primary entry point is
``ripple_carry_modular_circuit`` which either builds a ripple carry adder or a
naive modular multiplication circuit.  A small non-Clifford subroutine is
included in both cases to demonstrate hybrid behaviour.
"""

from __future__ import annotations

import networkx as nx
from random import Random
from typing import Iterable, List, Tuple

from quasar.circuit import Circuit, Gate

try:  # Allow execution when the module is imported as a script
    from .circuits import _cdkm_adder_gates, _vbe_adder_gates, _iqft_gates
except ImportError:  # pragma: no cover - fallback for script execution
    from circuits import _cdkm_adder_gates, _vbe_adder_gates, _iqft_gates  # type: ignore

# Names of gates forming the Clifford group used in benchmark filtering.
CLIFFORD_GATES = {
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "SDG",
    "CX",
    "CY",
    "CZ",
    "SWAP",
}


def is_clifford(circuit: Circuit) -> bool:
    """Return ``True`` if ``circuit`` contains only Clifford gates."""

    return all(g.gate in CLIFFORD_GATES for g in circuit.gates)


def ripple_carry_modular_circuit(
    bit_width: int, modulus: int | None = None, arithmetic: str = "cdkm"
) -> Circuit:
    """Build a ripple-carry adder or modular multiplication circuit.

    Parameters
    ----------
    bit_width:
        Number of bits in the arithmetic registers.
    modulus:
        Optional modulus.  When ``None`` a plain ripple-carry adder is
        generated.  If provided, a simple modular multiplication by the
        quantum register values is constructed.
    arithmetic:
        Selects the ripple-carry adder implementation.  Supported values are
        ``"cdkm"`` (Cuccaro--Draper--Kutin--Moulton) and ``"vbe"``
        (Vedral--Barenco--Ekert).

    Returns
    -------
    Circuit
        The assembled gate-level circuit ready for benchmarking.
    """

    if bit_width <= 0:
        return Circuit([])

    arithmetic = arithmetic.lower()
    if arithmetic == "cdkm":
        adder_gates = _cdkm_adder_gates(bit_width)
        helper_count = 0
    else:
        adder_gates = _vbe_adder_gates(bit_width)
        helper_count = max(0, bit_width - 1)

    if modulus is None:
        gates = list(adder_gates)
        gates.append(Gate("T", [0]))
        return Circuit(gates)

    n = bit_width
    gates: List[Gate] = []

    a = list(range(n))
    b = list(range(n, 2 * n))
    prod = list(range(2 * n, 3 * n))
    carry = 3 * n
    helpers = [carry + 1 + i for i in range(helper_count)]
    total_qubits = carry + 1 + helper_count

    for i in range(n):
        ctrl = b[i]
        targets = prod[i : i + n]
        if len(targets) < n:
            targets = targets + prod[: n - len(targets)]

        mapping = {0: carry}
        for j in range(n):
            mapping[1 + j] = a[j]
            mapping[1 + n + j] = targets[j]
        mapping[1 + 2 * n] = carry
        for idx, h in enumerate(helpers):
            mapping[1 + 2 * n + 1 + idx] = h

        for g in adder_gates:
            name = "C" + g.gate
            qubits = [ctrl] + [mapping[q] for q in g.qubits]
            gates.append(Gate(name, qubits, g.params))

    for q in range(min(3, total_qubits)):
        gates.append(Gate("T", [q]))

    mod_bits = bin(modulus % (1 << n))[2:].zfill(n)[::-1]
    for idx, bit in enumerate(mod_bits):
        if bit == "1":
            gates.append(Gate("X", [prod[idx]]))

    return Circuit(gates)


def _surface_patch_layout(distance: int, scheme: str = "surface") -> Tuple[int, int]:
    """Return the number of data and ancilla qubits for a stabiliser patch."""

    if distance <= 0:
        return 0, 0

    scheme = scheme.lower()
    if scheme == "repetition":
        data = distance
        ancilla = max(0, distance - 1)
    else:
        data = distance * distance
        ancilla = 2 * distance * (distance - 1)
    return data, ancilla


def _shift_gates(gates: Iterable[Gate], offset: int) -> List[Gate]:
    """Return ``gates`` with all qubits shifted by ``offset``."""

    return [Gate(g.gate, [q + offset for q in g.qubits], g.params) for g in gates]


def alternating_ladder_circuit(
    *,
    chain_length: int,
    dense_gadgets: int,
    gadget_width: int,
    ladder_layers: int,
    gadget_layers: int,
    seed: int = 0,
) -> Circuit:
    """Construct a ladder circuit alternating Clifford rails and dense gadgets.

    The generator arranges ``2 * chain_length`` qubits into two rails coupled via
    alternating ``CX``/``CZ`` interactions.  Segments of ``gadget_width`` rungs
    are selected as ``dense_gadgets`` injection sites where non-Clifford layers
    dominated by ``RY``/``RZ``/``T`` rotations and ``CCZ`` couplings are applied
    ``gadget_layers`` times.  The surrounding ladder rails remain comparatively
    light-weight to model regions that are efficiently simulated by stabiliser
    or low-rank methods.  Increasing ``gadget_layers`` therefore drives the
    Schmidt rank ``χ`` across the ladder boundaries while ``dense_gadgets``
    controls the spacing of these high-entanglement islands.

    Parameters
    ----------
    chain_length:
        Number of rungs in the ladder.  The circuit acts on ``2 * chain_length``
        qubits arranged into two rails.
    dense_gadgets:
        Number of dense gadget regions inserted along the ladder.  The regions
        are distributed as evenly as possible.
    gadget_width:
        Number of consecutive rungs forming each dense gadget.  The value is
        truncated to ``chain_length`` when larger.
    ladder_layers:
        Number of background ladder layers composed of alternating Clifford
        interactions.
    gadget_layers:
        Number of dense non-Clifford layers applied within each gadget region.
    seed:
        Random seed controlling angle selection inside the dense gadgets.

    Returns
    -------
    Circuit
        Gate-level circuit combining light-weight ladder rails with embedded
        dense gadgets.
    """

    if (
        chain_length <= 0
        or dense_gadgets < 0
        or gadget_width <= 0
        or ladder_layers <= 0
        or gadget_layers <= 0
    ):
        return Circuit([])

    rng = Random(seed)
    rung_count = chain_length
    total_qubits = rung_count * 2
    top_rail = [2 * idx for idx in range(rung_count)]
    bottom_rail = [2 * idx + 1 for idx in range(rung_count)]

    gates: List[Gate] = []

    # Background ladder composed mostly of Clifford operations to keep
    # stabiliser-friendly regions between the dense gadgets.
    for layer in range(ladder_layers):
        for qubit in range(total_qubits):
            if (layer + qubit) % 2 == 0:
                gates.append(Gate("H", [qubit]))
            else:
                gates.append(Gate("S", [qubit]))

        for idx in range(rung_count):
            control, target = (
                (top_rail[idx], bottom_rail[idx])
                if layer % 2 == 0
                else (bottom_rail[idx], top_rail[idx])
            )
            gates.append(Gate("CX", [control, target]))

        for rail in (top_rail, bottom_rail):
            for idx in range(len(rail) - 1):
                gates.append(Gate("CZ", [rail[idx], rail[idx + 1]]))

        if rung_count > 2:
            for idx in range(0, rung_count - 2, 2):
                gates.append(Gate("CX", [top_rail[idx], bottom_rail[idx + 2]]))

    width = min(gadget_width, rung_count)
    if dense_gadgets == 0 or width == 0:
        return Circuit(gates)

    window_count = rung_count - width + 1
    if window_count <= 0:
        return Circuit(gates)

    if dense_gadgets >= window_count:
        gadget_starts = list(range(window_count))
    else:
        gadget_starts = sorted(rng.sample(range(window_count), dense_gadgets))

    for start in gadget_starts:
        top_segment = top_rail[start : start + width]
        bottom_segment = bottom_rail[start : start + width]
        local_register = top_segment + bottom_segment

        for layer in range(gadget_layers):
            # Local single-qubit rotations drive non-Clifford behaviour.
            for qubit in local_register:
                gates.append(
                    Gate("RY", [qubit], {"theta": rng.uniform(0.2, 1.35)})
                )
                gates.append(
                    Gate("RZ", [qubit], {"phi": rng.uniform(0.15, 1.2)})
                )

            # Dense cross couplings between the rails.
            for control, target in zip(top_segment, bottom_segment):
                gates.append(
                    Gate("CRZ", [control, target], {"phi": rng.uniform(0.3, 1.1)})
                )
                gates.append(Gate("CX", [control, target]))

            # Multi-qubit entanglers to raise the Schmidt rank across the
            # gadget boundary.
            for idx in range(width - 1):
                gates.append(
                    Gate(
                        "CCZ",
                        [top_segment[idx], bottom_segment[idx], top_segment[idx + 1]],
                    )
                )
                gates.append(
                    Gate(
                        "CCZ",
                        [
                            bottom_segment[idx],
                            top_segment[idx + 1],
                            bottom_segment[idx + 1],
                        ],
                    )
                )

            focus = (start + layer) % width
            gates.append(Gate("T", [top_segment[focus]]))
            gates.append(Gate("T", [bottom_segment[focus]]))

    return Circuit(gates)


def surface_code_cycle(distance: int, rounds: int = 1, scheme: str = "surface") -> Circuit:
    """Construct repeated stabiliser cycles for simple error-correction codes.

    Parameters
    ----------
    distance:
        Code distance determining the number of data qubits.
    rounds:
        Number of stabiliser rounds to repeat.
    scheme:
        ``"surface"`` arranges qubits on a 2-D square lattice while ``"repetition"``
        uses a 1-D chain.

    Returns
    -------
    Circuit
        Circuit of explicit ``CX``/``CZ``/``H`` gates implementing ``rounds``
        cycles of parity-check interactions without measurements.
    """

    if distance <= 0 or rounds <= 0:
        return Circuit([])

    gates: List[Gate] = []
    scheme = scheme.lower()
    if scheme == "repetition":
        data_count = distance
        anc_count = distance - 1
        for _ in range(rounds):
            for i in range(anc_count):
                anc = data_count + i
                left = i
                right = i + 1
                gates.append(Gate("CX", [left, anc]))
                gates.append(Gate("CX", [right, anc]))
    else:  # surface code
        d = distance
        data_count = d * d
        anc_start = data_count
        for _ in range(rounds):
            a_offset = 0
            for row in range(d):
                for col in range(d - 1):
                    anc = anc_start + a_offset
                    a_offset += 1
                    q1 = row * d + col
                    q2 = row * d + col + 1
                    gates.append(Gate("H", [anc]))
                    gates.append(Gate("CZ", [anc, q1]))
                    gates.append(Gate("CZ", [anc, q2]))
                    gates.append(Gate("H", [anc]))
            for row in range(d - 1):
                for col in range(d):
                    anc = anc_start + a_offset
                    a_offset += 1
                    q1 = row * d + col
                    q2 = (row + 1) * d + col
                    gates.append(Gate("H", [anc]))
                    gates.append(Gate("CZ", [anc, q1]))
                    gates.append(Gate("CZ", [anc, q2]))
                    gates.append(Gate("H", [anc]))

    return Circuit(gates)


def dual_magic_injection_circuit(
    *,
    patch_distance: int,
    stabilizer_rounds: int,
    gadget_width: int,
    gadget: str = "ccz_bridge",
    scheme: str = "surface",
) -> Circuit:
    """Combine two stabiliser patches with a configurable non-Clifford gadget.

    The generator constructs two disjoint surface-code patches each executing
    ``stabilizer_rounds`` of :func:`surface_code_cycle`.  After the stabiliser
    evolution, a non-Clifford gadget acting on ``gadget_width`` data qubits from
    each patch is appended.  The gadget type is selected via ``gadget`` and is
    intended to model a magic-state injection interface between protected
    patches.

    Parameters
    ----------
    patch_distance:
        Distance of each surface-code patch; determines the number of data and
        ancilla qubits per patch.
    stabilizer_rounds:
        Number of stabiliser measurement rounds to execute on each patch before
        performing the injection gadget.
    gadget_width:
        Number of data qubit pairs (one per patch) involved in the gadget.
    gadget:
        Identifier for the non-Clifford gadget.  Supported values are
        ``"ccz_bridge"`` which applies a chain of ``CCZ`` operations across
        matched data qubits with dedicated magic ancillae, ``"tof_bridge"``
        which realises a Toffoli-based bridge with injected ``T`` rotations and
        ``"t_bridge"`` which couples the patches through controlled-NOT bridges
        and single-qubit ``T`` gates only.
    scheme:
        Layout of the stabiliser patch.  ``"surface"`` builds a square-lattice
        patch while ``"repetition"`` produces a linear chain with fewer
        ancillae for lighter-weight experiments.

    Returns
    -------
    Circuit
        The assembled benchmark circuit capturing stabiliser dynamics followed
        by a non-Clifford coupling gadget.
    """

    if patch_distance <= 0 or stabilizer_rounds <= 0 or gadget_width <= 0:
        return Circuit([])

    data_qubits, ancilla_qubits = _surface_patch_layout(patch_distance, scheme)
    patch_qubits = data_qubits + ancilla_qubits
    if patch_qubits == 0:
        return Circuit([])

    base_cycle = surface_code_cycle(patch_distance, stabilizer_rounds, scheme=scheme)
    left_patch = list(base_cycle.gates)
    right_patch = _shift_gates(base_cycle.gates, patch_qubits)

    width = min(gadget_width, data_qubits)
    if width == 0:
        return Circuit(left_patch + right_patch)

    left_data = list(range(width))
    right_data = [q + patch_qubits for q in range(width)]
    ancilla_start = patch_qubits * 2
    ancilla = list(range(ancilla_start, ancilla_start + width))

    gadget = gadget.lower()
    if gadget not in {"ccz_bridge", "tof_bridge", "t_bridge"}:
        raise ValueError(f"unsupported gadget '{gadget}'")

    gates: List[Gate] = []
    gates.extend(left_patch)
    gates.extend(right_patch)

    for left, right, anc in zip(left_data, right_data, ancilla):
        if gadget == "ccz_bridge":
            gates.append(Gate("H", [anc]))
            gates.append(Gate("T", [anc]))
            gates.append(Gate("CCZ", [left, right, anc]))
            gates.append(Gate("T", [left]))
            gates.append(Gate("T", [right]))
            gates.append(Gate("S", [left]))
            gates.append(Gate("S", [right]))
        elif gadget == "tof_bridge":
            gates.append(Gate("H", [anc]))
            gates.append(Gate("T", [anc]))
            gates.append(Gate("H", [anc]))
            gates.append(Gate("CCX", [left, right, anc]))
            gates.append(Gate("T", [anc]))
            gates.append(Gate("H", [anc]))
        else:
            gates.append(Gate("H", [anc]))
            gates.append(Gate("CX", [left, anc]))
            gates.append(Gate("T", [anc]))
            gates.append(Gate("CX", [right, anc]))
            gates.append(Gate("H", [anc]))
            gates.append(Gate("T", [anc]))
        gates.append(Gate("S", [left]))
        gates.append(Gate("S", [right]))

    return Circuit(gates)


def grover_with_oracle_circuit(
    n_qubits: int, oracle_depth: int, iterations: int = 1
) -> Circuit:
    """Build a Grover circuit with a configurable-depth oracle.

    Parameters
    ----------
    n_qubits:
        Number of search qubits.
    oracle_depth:
        Number of cascaded Toffoli/CNOT layers forming the oracle.
    iterations:
        Number of Grover iterations to apply.

    Returns
    -------
    Circuit
        The assembled circuit ready for benchmarking.
    """

    if n_qubits <= 0:
        return Circuit([])

    gates: list[Gate] = []

    for q in range(n_qubits):
        gates.append(Gate("H", [q]))

    controls = list(range(n_qubits - 1))
    target = n_qubits - 1
    mcx_name = "C" * len(controls) + "X" if controls else "X"

    for _ in range(iterations):
        # Oracle composed from cascaded Toffoli/CNOT layers.
        gates.append(Gate("H", [target]))
        for _ in range(oracle_depth):
            gates.append(Gate(mcx_name, controls + [target]))

            for q in range(n_qubits - 1):
                gates.append(Gate("CX", [q, q + 1]))
            for q in reversed(range(n_qubits - 1)):
                gates.append(Gate("CX", [q, q + 1]))
        gates.append(Gate("H", [target]))

        # Standard Grover diffusion operator.
        for q in range(n_qubits):
            gates.append(Gate("H", [q]))
        for q in range(n_qubits):
            gates.append(Gate("X", [q]))
        gates.append(Gate("H", [target]))
        gates.append(Gate(mcx_name, controls + [target]))
        gates.append(Gate("H", [target]))
        for q in range(n_qubits):
            gates.append(Gate("X", [q]))
        for q in range(n_qubits):
            gates.append(Gate("H", [q]))

    return Circuit(gates)


def deep_qaoa_circuit(graph: nx.Graph, p_layers: int) -> Circuit:
    """Construct a deep QAOA circuit for an input graph.

    The circuit alternates between problem-Hamiltonian layers of ``RZZ``
    interactions for each edge in the ``graph`` and mixing layers of single
    qubit ``RX`` rotations.  The number of alternations is controlled by
    ``p_layers`` and is independent of the graph size.

    Parameters
    ----------
    graph:
        Input graph defining the problem interactions.
    p_layers:
        Number of problem/mixing layer pairs to apply.

    Returns
    -------
    Circuit
        The assembled gate-level circuit ready for benchmarking.
    """

    if graph.number_of_nodes() == 0 or p_layers <= 0:
        return Circuit([])

    gates: List[Gate] = []
    n_qubits = graph.number_of_nodes()
    for _ in range(p_layers):
        for u, v in graph.edges():
            gates.append(Gate("RZZ", [u, v], {"theta": 0.5}))
        for q in range(n_qubits):
            gates.append(Gate("RX", [q], {"theta": 0.5}))
    return Circuit(gates)


def surface_corrected_qaoa(bit_width: int, distance: int, rounds: int) -> Circuit:
    """Interleave QAOA layers with surface-code error-correction cycles.

    The generator constructs a simple ring ``p``-layer QAOA circuit acting on
    ``bit_width`` data qubits.  After each problem/mixing layer pair a single
    round of :func:`surface_code_cycle` is appended to model an error-correction
    cycle on a ``distance`` x ``distance`` lattice.  Extra data qubits required
    by the surface code are included automatically and unused by the algorithmic
    portion.

    Parameters
    ----------
    bit_width:
        Number of problem qubits arranged on a cycle graph.
    distance:
        Code distance of the inserted surface-code cycles.  The lattice must
        contain at least ``bit_width`` data qubits.
    rounds:
        Number of QAOA problem/mixing layers to apply.  Each layer is followed
        by one surface-code stabiliser round.

    Returns
    -------
    Circuit
        The assembled gate-level circuit combining algorithmic and
        error-correction layers.
    """

    if bit_width <= 0 or distance <= 0 or rounds <= 0:
        return Circuit([])

    problem_graph = nx.cycle_graph(bit_width)
    correction = surface_code_cycle(distance).gates

    gates: List[Gate] = []
    for _ in range(rounds):
        for u, v in problem_graph.edges():
            gates.append(Gate("RZZ", [u, v], {"theta": 0.5}))
        for q in range(bit_width):
            gates.append(Gate("RX", [q], {"theta": 0.5}))
        gates.extend(correction)

    return Circuit(gates)


def phase_estimation_classical_unitary(
    eigen_qubits: int, precision_qubits: int, classical_depth: int
) -> Circuit:
    """Construct a phase-estimation circuit using a classical reversible unitary.

    The circuit uses ``precision_qubits`` qubits as the phase register and
    ``eigen_qubits`` as the target on which a unitary consisting of
    ``classical_depth`` layers of reversible logic (CNOT/Toffoli gates) is
    applied.  Controlled powers of this unitary are followed by an explicit
    inverse quantum Fourier transform implemented via ``CRZ`` and ``H`` gates.

    Parameters
    ----------
    eigen_qubits:
        Number of qubits for the eigenstate register.
    precision_qubits:
        Number of qubits in the phase-estimation register.
    classical_depth:
        Number of reversible logic layers forming the base unitary.

    Returns
    -------
    Circuit
        The assembled gate-level circuit ready for benchmarking.
    """

    if eigen_qubits <= 0 or precision_qubits <= 0 or classical_depth <= 0:
        return Circuit([])

    unitary: List[Gate] = []
    if eigen_qubits == 1:
        unitary.append(Gate("X", [0]))
    else:
        for _ in range(classical_depth):
            for i in range(eigen_qubits - 1):
                unitary.append(Gate("CX", [i, i + 1]))
            if eigen_qubits >= 3:
                for i in range(eigen_qubits - 2):
                    unitary.append(Gate("CCX", [i, i + 1, i + 2]))

    total_qubits = precision_qubits + eigen_qubits
    phase_reg = list(range(precision_qubits))
    eigen_reg = list(range(precision_qubits, total_qubits))

    gates: List[Gate] = []

    for q in phase_reg:
        gates.append(Gate("H", [q]))

    for j, ctrl in enumerate(phase_reg):
        reps = 2 ** j
        for _ in range(reps):
            for g in unitary:
                name = "C" + g.gate
                qubits = [ctrl] + [q + precision_qubits for q in g.qubits]
                gates.append(Gate(name, qubits, g.params))

    gates.extend(_iqft_gates(precision_qubits, 0))

    return Circuit(gates)


def dense_to_clifford_partition_circuit(
    dense_qubits: int,
    clifford_qubits: int,
    *,
    boundary: int = 6,
    schmidt_layers: int = 2,
    prefix_layers: int = 2,
    clifford_layers: int = 6,
    seed: int = 0,
) -> Circuit:
    """Model a dense-to-Clifford partition used in boundary sweeps.

    The construction mirrors the assumptions in
    ``docs/partitioning_thresholds.ipynb`` where an initial dense prefix is
    followed by a Clifford fragment that prefers tableau simulation.  The
    interface between the two fragments exposes ``boundary`` qubits and the
    ``schmidt_layers`` parameter controls how many rounds of cross-fragment
    entanglers are inserted to increase the effective Schmidt rank at the cut.

    Parameters
    ----------
    dense_qubits:
        Number of qubits in the dense, non-Clifford prefix.
    clifford_qubits:
        Number of qubits evolved by the trailing Clifford fragment.
    boundary:
        Number of qubits shared between the fragments.  The value must not
        exceed either register size.
    schmidt_layers:
        Number of cross-fragment entangling layers used to raise the Schmidt
        rank.  Each layer applies ``CRZ`` bridges and alternating ``CX`` gates
        across the boundary.
    prefix_layers:
        Number of dense non-Clifford layers applied before partitioning.
    clifford_layers:
        Number of tableau-friendly Clifford layers applied to the suffix.
    seed:
        Seed controlling the deterministic rotation angles.
    """

    if dense_qubits <= 0 or clifford_qubits <= 0:
        return Circuit([])
    if boundary <= 0:
        raise ValueError("boundary must be positive")
    if boundary > min(dense_qubits, clifford_qubits):
        raise ValueError("boundary must not exceed either fragment size")
    if schmidt_layers <= 0:
        raise ValueError("schmidt_layers must be positive")
    if prefix_layers <= 0:
        raise ValueError("prefix_layers must be positive")
    if clifford_layers <= 0:
        raise ValueError("clifford_layers must be positive")

    rng = Random(seed)
    gates: List[Gate] = []

    dense_offset = 0
    clifford_offset = dense_qubits

    # Dense prefix: alternating non-Clifford single-qubit rotations and
    # nearest-neighbour entanglers.
    for layer in range(prefix_layers):
        for q in range(dense_qubits):
            theta = rng.uniform(0.25, 1.35)
            phi = rng.uniform(0.2, 1.1)
            gates.append(Gate("RY", [dense_offset + q], {"theta": theta}))
            gates.append(Gate("RZ", [dense_offset + q], {"phi": phi}))
        for q in range(dense_qubits - 1):
            angle = rng.uniform(0.35, 1.25)
            gates.append(
                Gate(
                    "CRZ",
                    [dense_offset + q, dense_offset + q + 1],
                    {"phi": angle},
                )
            )
        # Introduce long-range entanglement so the prefix retains a dense
        # character even when partitioned away from the Clifford suffix.
        if dense_qubits > 2:
            target = (layer % (dense_qubits - 1)) + 1
            gates.append(
                Gate(
                    "CRZ",
                    [dense_offset, dense_offset + target],
                    {"phi": rng.uniform(0.3, 1.0)},
                )
            )
        for q in range(0, dense_qubits, 3):
            gates.append(Gate("T", [dense_offset + q % dense_qubits]))

    # Cross-fragment entanglement to expose a tunable conversion boundary.
    prefix_boundary = list(range(dense_offset + dense_qubits - boundary, dense_offset + dense_qubits))
    suffix_boundary = list(range(clifford_offset, clifford_offset + boundary))
    for layer in range(schmidt_layers):
        for src, dst in zip(prefix_boundary, suffix_boundary):
            angle = rng.uniform(0.25, 1.05)
            gates.append(Gate("CRZ", [src, dst], {"phi": angle}))
            if layer % 2 == 0:
                gates.append(Gate("CX", [src, dst]))
        for dst in suffix_boundary:
            gates.append(Gate("T", [dst]))

    # Tableau-friendly suffix dominated by Clifford operations.
    for _ in range(clifford_layers):
        for q in range(clifford_qubits):
            gates.append(Gate("H", [clifford_offset + q]))
        for q in range(clifford_qubits - 1):
            gates.append(Gate("CX", [clifford_offset + q, clifford_offset + q + 1]))
        if clifford_qubits > 2:
            gates.append(
                Gate(
                    "CX",
                    [clifford_offset + clifford_qubits - 1, clifford_offset],
                )
            )

    return Circuit(gates, use_classical_simplification=False)


def staged_partition_circuit(
    clifford_qubits: int,
    core_qubits: int,
    suffix_qubits: int,
    *,
    prefix_depth: int = 4,
    core_layers: int = 3,
    suffix_layers: int = 3,
    prefix_core_boundary: int = 6,
    core_suffix_boundary: int = 4,
    cross_layers: int = 2,
    suffix_sparsity: float = 0.6,
    seed: int = 1,
) -> Circuit:
    """Combine Clifford, dense and sparse fragments into a single workload.

    The generator reflects the three-fragment case study from
    ``docs/partitioning_tradeoffs.ipynb``.  A tableau-friendly Clifford
    initialisation is followed by a dense, non-local core that favours the
    statevector backend.  The final suffix remains mostly diagonal and sparse
    so that the decision-diagram backend can accelerate it.  Conversion
    boundaries between the fragments are governed by ``prefix_core_boundary``
    and ``core_suffix_boundary`` while ``cross_layers`` controls how many
    rounds of cross-fragment entanglement raise the effective Schmidt ranks.

    Parameters
    ----------
    clifford_qubits:
        Width of the initial Clifford fragment.
    core_qubits:
        Width of the dense middle fragment.
    suffix_qubits:
        Width of the sparse decision-diagram suffix.
    prefix_depth:
        Number of tableau-friendly layers used for the initial fragment.
    core_layers:
        Number of dense, non-Clifford layers in the middle fragment.
    suffix_layers:
        Number of sparse layers in the suffix.
    prefix_core_boundary:
        Number of qubits exposed between the first two fragments.
    core_suffix_boundary:
        Number of qubits exposed between the second and third fragments.
    cross_layers:
        Number of cross-fragment entangling rounds.
    suffix_sparsity:
        Fraction in ``[0, 1]`` describing how sparse the suffix remains.  A
        value near ``1`` keeps most operations diagonal, whereas ``0`` applies
        amplitude-mixing rotations to all suffix qubits.
    seed:
        Seed controlling deterministic angle selection.
    """

    if clifford_qubits <= 0 or core_qubits <= 0 or suffix_qubits <= 0:
        return Circuit([])
    if prefix_core_boundary <= 0 or core_suffix_boundary <= 0:
        raise ValueError("boundaries must be positive")
    if prefix_core_boundary > min(clifford_qubits, core_qubits):
        raise ValueError("prefix_core_boundary too large for fragment sizes")
    if core_suffix_boundary > min(core_qubits, suffix_qubits):
        raise ValueError("core_suffix_boundary too large for fragment sizes")
    if cross_layers <= 0:
        raise ValueError("cross_layers must be positive")
    if not 0.0 <= suffix_sparsity <= 1.0:
        raise ValueError("suffix_sparsity must be between 0 and 1")

    rng = Random(seed)
    gates: List[Gate] = []

    clifford_offset = 0
    core_offset = clifford_qubits
    suffix_offset = clifford_qubits + core_qubits

    # Clifford fragment: repeated GHZ-like layers to remain tableau friendly.
    for _ in range(prefix_depth):
        for q in range(clifford_qubits):
            gates.append(Gate("H", [clifford_offset + q]))
        for q in range(clifford_qubits - 1):
            gates.append(Gate("CX", [clifford_offset + q, clifford_offset + q + 1]))
        if clifford_qubits > 2:
            gates.append(Gate("CZ", [clifford_offset, clifford_offset + clifford_qubits - 1]))

    # Dense core: layers of non-local rotations and entanglers.
    for layer in range(core_layers):
        for q in range(core_qubits):
            theta = rng.uniform(0.2, 1.4)
            phi = rng.uniform(0.15, 1.2)
            gates.append(Gate("RY", [core_offset + q], {"theta": theta}))
            gates.append(Gate("RZ", [core_offset + q], {"phi": phi}))
        for q in range(core_qubits - 1):
            gates.append(
                Gate(
                    "CRZ",
                    [core_offset + q, core_offset + q + 1],
                    {"phi": rng.uniform(0.3, 1.1)},
                )
            )
        if core_qubits > 2:
            offset = (layer % (core_qubits - 1)) + 1
            gates.append(
                Gate(
                    "CRZ",
                    [core_offset, core_offset + offset],
                    {"phi": rng.uniform(0.25, 1.0)},
                )
            )
        gates.append(Gate("T", [core_offset + layer % core_qubits]))

    # Cross entanglement between Clifford prefix and dense core.
    prefix_boundary = list(range(clifford_offset + clifford_qubits - prefix_core_boundary, clifford_offset + clifford_qubits))
    core_left_boundary = list(range(core_offset, core_offset + prefix_core_boundary))
    for layer in range(cross_layers):
        for src, dst in zip(prefix_boundary, core_left_boundary):
            gates.append(
                Gate(
                    "CRZ",
                    [src, dst],
                    {"phi": rng.uniform(0.25, 1.0)},
                )
            )
            if layer % 2 == 0:
                gates.append(Gate("CX", [src, dst]))
        for dst in core_left_boundary:
            gates.append(Gate("T", [dst]))

    # Sparse suffix dominated by diagonal rotations.
    mix_qubits = max(0, round((1.0 - suffix_sparsity) * suffix_qubits))
    mix_indices = list(range(mix_qubits))
    for layer in range(suffix_layers):
        for q in range(suffix_qubits):
            phi = rng.uniform(0.05, 0.8)
            gates.append(Gate("RZ", [suffix_offset + q], {"phi": phi}))
        for idx in mix_indices:
            gates.append(
                Gate(
                    "RX",
                    [suffix_offset + (idx + layer) % suffix_qubits],
                    {"theta": rng.uniform(0.1, 1.0)},
                )
            )
        if suffix_qubits > 1:
            gates.append(
                Gate(
                    "CZ",
                    [suffix_offset, suffix_offset + suffix_qubits - 1],
                )
            )
        for q in range(0, suffix_qubits, 3):
            gates.append(Gate("T", [suffix_offset + q % suffix_qubits]))

    # Cross entanglement between dense core and sparse suffix.
    core_right_boundary = list(range(core_offset + core_qubits - core_suffix_boundary, core_offset + core_qubits))
    suffix_boundary = list(range(suffix_offset, suffix_offset + core_suffix_boundary))
    for layer in range(cross_layers):
        for src, dst in zip(core_right_boundary, suffix_boundary):
            gates.append(
                Gate(
                    "CRZ",
                    [src, dst],
                    {"phi": rng.uniform(0.15, 0.9)},
                )
            )
            if layer % 2 == 1:
                gates.append(Gate("CX", [src, dst]))
        for src in core_right_boundary:
            gates.append(Gate("T", [src]))

    return Circuit(gates, use_classical_simplification=False)
