"""Larger benchmark circuits composed from arithmetic primitives.

This module provides circuits intended to stress-test QuASAr's
partitioning and simulation on wider registers.  The primary entry point is
``ripple_carry_modular_circuit`` which either builds a ripple carry adder or a
naive modular multiplication circuit.  A small non-Clifford subroutine is
included in both cases to demonstrate hybrid behaviour.
"""

from __future__ import annotations

import networkx as nx
from typing import List

from quasar.circuit import Circuit, Gate
from .circuits import _cdkm_adder_gates, _vbe_adder_gates, _iqft_gates


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
        return Circuit(gates, use_classical_simplification=False)

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

    return Circuit(gates, use_classical_simplification=False)


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

    return Circuit(gates, use_classical_simplification=False)


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

    return Circuit(gates, use_classical_simplification=False)


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
