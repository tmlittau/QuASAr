"""Common benchmark circuits for QuASAr."""
from __future__ import annotations

import math
from typing import List

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    QFT,
    RealAmplitudes,
    EfficientSU2,
    TwoLocal,
    ZZFeatureMap,
    CDKMRippleCarryAdder,
    DraperQFTAdder,
    VBERippleCarryAdder,
)
from qiskit.circuit.random import random_circuit as qiskit_random_circuit

from quasar.circuit import Circuit, Gate


def ghz_circuit(
    n_qubits: int, *, use_classical_simplification: bool = False
) -> Circuit:
    """Create an ``n_qubits`` GHZ state preparation circuit."""
    gates: List[Gate] = []
    if n_qubits <= 0:
        return Circuit(gates, use_classical_simplification=use_classical_simplification)
    gates.append(Gate("H", [0]))
    for i in range(1, n_qubits):
        gates.append(Gate("CX", [i - 1, i]))
    return Circuit(gates, use_classical_simplification=use_classical_simplification)


def _qft_spec(n: int) -> List[Gate]:
    """Return a list of :class:`Gate` objects for the QFT circuit."""
    gates: List[Gate] = []
    for i in reversed(range(n)):
        gates.append(Gate("H", [i]))
        for j, q in enumerate(reversed(range(0, i))):
            gates.append(Gate("CP", [q, i], {"k": j + 1}))
    return gates


def qft_circuit(
    n_qubits: int, *, use_classical_simplification: bool = False
) -> Circuit:
    """Create an ``n_qubits`` quantum Fourier transform circuit."""
    return Circuit(
        _qft_spec(n_qubits),
        use_classical_simplification=use_classical_simplification,
    )


def qft_on_ghz_circuit(
    n_qubits: int, *, use_classical_simplification: bool = False
) -> Circuit:
    """Apply the QFT to a GHZ state."""
    ghz = ghz_circuit(
        n_qubits, use_classical_simplification=use_classical_simplification
    )
    qft = qft_circuit(
        n_qubits, use_classical_simplification=use_classical_simplification
    )
    return Circuit(
        list(ghz.gates) + list(qft.gates),
        use_classical_simplification=use_classical_simplification,
    )


def _w_state_spec(n: int) -> List[Gate]:

    """Return a gate list for a W state preparation circuit."""
    gates: List[Gate] = []
    gates.append(Gate("RY", [0], {"theta": 2 * math.acos(math.sqrt(1 / n))}))

    for q in range(1, n - 1):
        gates.append(
            Gate(
                "CRY",
                [q - 1, q],
                {"theta": 2 * math.acos(math.sqrt(1 / (n - q)))}
            )
        )
    for q in reversed(range(n - 1)):
        gates.append(Gate("CX", [q, q + 1]))
    gates.append(Gate("X", [0]))
    return gates


def w_state_circuit(
    n_qubits: int, *, use_classical_simplification: bool = False
) -> Circuit:
    """Create an ``n_qubits`` W state preparation circuit."""
    gates = _w_state_spec(n_qubits)
    return Circuit(gates, use_classical_simplification=use_classical_simplification)


def grover_circuit(n_qubits: int, n_iterations: int = 1) -> Circuit:
    """Create a Grover search circuit targeting the all-ones state.

    Args:
        n_qubits: Number of search qubits.
        n_iterations: Number of Grover iterations to apply.

    Returns:
        A :class:`Circuit` implementing the algorithm.
    """

    if n_qubits <= 0:
        return Circuit([])

    gates: List[Gate] = []

    # Initial Hadamards
    for q in range(n_qubits):
        gates.append(Gate("H", [q]))

    controls = list(range(n_qubits - 1))
    target = n_qubits - 1
    mcx_name = "C" * len(controls) + "X" if controls else "X"

    for _ in range(n_iterations):
        # Oracle marking the all-ones state
        gates.append(Gate("H", [target]))
        gates.append(Gate(mcx_name, controls + [target]))
        gates.append(Gate("H", [target]))

        # Diffuser
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


def bernstein_vazirani_circuit(
    n_qubits: int,
    secret: int = 0,
    *,
    use_classical_simplification: bool = False,
) -> Circuit:
    """Create a Bernstein-Vazirani circuit for a given secret string.

    Args:
        n_qubits: Number of secret bits.
        secret: Integer encoding the secret string (little-endian).

    Returns:
        A :class:`Circuit` implementing the algorithm.
    """

    gates: List[Gate] = []
    if n_qubits <= 0:
        return Circuit(gates, use_classical_simplification=use_classical_simplification)

    anc = n_qubits

    # Initial Hadamards on the search register.
    for q in range(n_qubits):
        gates.append(Gate("H", [q]))

    # Prepare the ancilla in |-> state.
    gates.append(Gate("X", [anc]))
    gates.append(Gate("H", [anc]))

    # Oracle marking the secret string using CNOTs.
    for i in range(n_qubits):
        if (secret >> i) & 1:
            gates.append(Gate("CX", [i, anc]))

    # Final Hadamards to decode the secret string.
    for q in range(n_qubits):
        gates.append(Gate("H", [q]))

    return Circuit(gates, use_classical_simplification=use_classical_simplification)


def amplitude_estimation_circuit(num_qubits: int, probability: float) -> Circuit:
    """Construct an amplitude estimation circuit.

    Args:
        num_qubits: Number of evaluation qubits.
        probability: Success probability encoded by the oracle.

    Returns:
        A :class:`Circuit` implementing a basic amplitude estimation routine.
    """

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    qc = QuantumCircuit(num_qubits + 1)
    theta = 2 * math.asin(math.sqrt(probability))
    qc.h(range(num_qubits))
    qc.ry(theta, num_qubits)
    for i in range(num_qubits):
        qc.crz(2 ** i * 2 * theta, i, num_qubits)
    qc.append(QFT(num_qubits, inverse=True), range(num_qubits))
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def bmw_quark_circuit(num_qubits: int, depth: int, kind: str = "cardinality") -> Circuit:
    """Generate BMW-QUARK ansatz circuits.

    Args:
        num_qubits: Number of qubits.
        depth: Number of alternating rotation/entangling layers.
        kind: ``"cardinality"`` or ``"circular"`` ansatz style.

    Returns:
        The requested ansatz circuit as a :class:`Circuit`.
    """

    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for q in range(num_qubits):
            qc.rx(np.pi / 2, q)
        if kind == "cardinality":
            for q in range(0, num_qubits - 1, 2):
                qc.rxx(np.pi / 2, q, q + 1)
            for q in range(1, num_qubits - 1, 2):
                qc.rxx(np.pi / 2, q, q + 1)
        else:  # circular
            for q in range(num_qubits):
                qc.rxx(np.pi / 2, q, (q + 1) % num_qubits)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def adder_circuit(num_qubits: int, kind: str = "cdkm") -> Circuit:
    """Construct CDKM, Draper or VBE adder circuits.

    Args:
        num_qubits: Number of bits in each addend.
        kind: ``"cdkm"``, ``"draper"`` or ``"vbe"``.
    """

    if kind == "cdkm":
        qc = CDKMRippleCarryAdder(num_qubits)
    elif kind == "draper":
        qc = DraperQFTAdder(num_qubits)
    elif kind == "vbe":
        qc = VBERippleCarryAdder(num_qubits)
    else:
        raise ValueError("Unknown adder kind")
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def deutsch_jozsa_circuit(num_qubits: int, balanced: bool = True) -> Circuit:
    """Construct a Deutsch-Jozsa circuit.

    Args:
        num_qubits: Number of input bits.
        balanced: Whether to use a balanced oracle; otherwise constant.
    """

    qc = QuantumCircuit(num_qubits + 1)
    qc.x(num_qubits)
    qc.h(range(num_qubits + 1))
    if balanced:
        for i in range(num_qubits):
            qc.cx(i, num_qubits)
    qc.h(range(num_qubits))
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def graph_state_circuit(num_qubits: int, degree: int, seed: int | None = None) -> Circuit:
    """Generate a random regular graph state circuit."""

    import networkx as nx

    if degree >= num_qubits:
        raise ValueError("degree must be < num_qubits")
    if (num_qubits * degree) % 2 != 0:
        raise ValueError("n * degree must be even for a regular graph")
    g = nx.random_regular_graph(degree, num_qubits, seed=seed)
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for u, v in g.edges():
        qc.cz(u, v)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def hhl_circuit(num_qubits: int) -> Circuit:
    """Create a simple HHL circuit. Requires ``num_qubits >= 3``."""

    if num_qubits < 3:
        raise ValueError("HHL requires at least 3 qubits")
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 2)
    qc.ry(math.pi / 4, 2)
    qc.cx(1, 2)
    qc.h(0)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def hrs_circuit(num_qubits: int) -> Circuit:
    """Construct a toy HRS arithmetic circuit."""

    if num_qubits % 2 != 0:
        raise ValueError("num_qubits must be even for HRS circuit")
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits, 2):
        target = (i + 2) % num_qubits
        qc.ccx(i, i + 1, target)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def qaoa_circuit(num_qubits: int, repetitions: int = 1, seed: int | None = None) -> Circuit:
    """Create a basic QAOA circuit on a ring graph."""

    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0, 2 * np.pi, size=repetitions)
    betas = rng.uniform(0, 2 * np.pi, size=repetitions)
    edges = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for p in range(repetitions):
        for u, v in edges:
            qc.rzz(gammas[p], u, v)
        for q in range(num_qubits):
            qc.rx(betas[p], q)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def qnn_circuit(num_qubits: int) -> Circuit:
    """Construct a simple quantum neural network circuit."""

    fm = ZZFeatureMap(num_qubits)
    ansatz = RealAmplitudes(num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.append(fm, range(num_qubits))
    qc.append(ansatz, range(num_qubits))
    params = {p: (i + 1) * 0.1 for i, p in enumerate(qc.parameters)}
    qc = qc.assign_parameters(params)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def qpe_circuit(num_qubits: int, inexact: bool = False) -> Circuit:
    """Quantum phase estimation circuit.

    Args:
        num_qubits: Number of counting qubits.
        inexact: Whether to use an inexact eigenphase.
    """

    qc = QuantumCircuit(num_qubits + 1)
    theta = 2 * np.pi / (2**num_qubits)
    if inexact:
        theta *= 1.1
    qc.h(range(num_qubits))
    for j in range(num_qubits):
        qc.cp(2 ** j * theta, j, num_qubits)
    qc.append(QFT(num_qubits, inverse=True), range(num_qubits))
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def quantum_walk_circuit(num_qubits: int, depth: int) -> Circuit:
    """Construct a simple quantum walk circuit."""

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for _ in range(depth):
        if num_qubits > 1:
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(range(num_qubits))
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def random_circuit(
    num_qubits: int,
    seed: int | None = None,
    *,
    use_classical_simplification: bool = False,
) -> Circuit:
    """Generate a random circuit of depth ``2*num_qubits``."""

    qc = qiskit_random_circuit(num_qubits, 2 * num_qubits, seed=seed)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(
        qc, use_classical_simplification=use_classical_simplification
    )


def shor_circuit(circuit_size: int) -> Circuit:
    """Create a toy Shor factoring circuit of a given size."""

    qc = QuantumCircuit(circuit_size)
    qc.h(range(circuit_size))
    if circuit_size > 1:
        qc.cx(0, circuit_size - 1)
    qc.append(QFT(circuit_size), range(circuit_size))
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def real_amplitudes_circuit(
    num_qubits: int, reps: int = 1, entanglement: str | List[int] | List[List[int]] = "full"
) -> Circuit:
    """Construct a ``RealAmplitudes`` ansatz with bound parameters."""

    qc = RealAmplitudes(num_qubits, reps=reps, entanglement=entanglement)
    params = {p: 0.1 * (i + 1) for i, p in enumerate(qc.parameters)}
    qc = qc.assign_parameters(params)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def efficient_su2_circuit(
    num_qubits: int, reps: int = 1, entanglement: str | List[int] | List[List[int]] = "full"
) -> Circuit:
    """Construct an ``EfficientSU2`` ansatz with bound parameters."""

    qc = EfficientSU2(num_qubits, reps=reps, entanglement=entanglement)
    params = {p: 0.1 * (i + 1) for i, p in enumerate(qc.parameters)}
    qc = qc.assign_parameters(params)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def two_local_circuit(
    num_qubits: int, reps: int = 1, entanglement: str | List[int] | List[List[int]] = "full"
) -> Circuit:
    """Construct a ``TwoLocal`` ansatz with bound parameters."""

    qc = TwoLocal(num_qubits, reps=reps, entanglement=entanglement)
    params = {p: 0.1 * (i + 1) for i, p in enumerate(qc.parameters)}
    qc = qc.assign_parameters(params)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def clifford_ec_circuit() -> Circuit:
    """Three-qubit bit-flip error-correction circuit using Clifford gates.

    Measurement operations are omitted so the circuit contains only unitary gates.
    """
    qc = QuantumCircuit(5)
    # Encode logical qubit into three data qubits
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    # Syndrome extraction with two ancilla qubits
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(1, 4)
    qc.cx(2, 4)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h"])
    return Circuit.from_qiskit(qc)


def ripple_add_circuit(num_bits: int = 4) -> Circuit:
    """Ripple-carry adder for two ``num_bits``-bit registers."""
    adder = VBERippleCarryAdder(num_bits)
    qc = QuantumCircuit(adder.num_qubits)
    qc.append(adder, range(adder.num_qubits))
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"], optimization_level=0)
    return Circuit.from_qiskit(qc)


def vqe_chain_circuit(num_qubits: int = 6, depth: int = 2) -> Circuit:
    """Parameterized VQE ansatz with linear entanglement chain."""
    qc = EfficientSU2(num_qubits, reps=depth, entanglement="linear")
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x", "ry", "rz"])
    # Disable classical simplification to avoid issues with unbound parameters
    return Circuit.from_qiskit(qc, use_classical_simplification=False)


def random_hybrid_circuit(num_qubits: int = 6, depth: int = 10, seed: int | None = None) -> Circuit:
    """Random circuit mixing Clifford and non-Clifford gates."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for q in range(num_qubits):
            gate = rng.choice(["h", "s"])
            getattr(qc, gate)(q)
        a, b = rng.choice(num_qubits, 2, replace=False)
        if rng.random() < 0.5:
            qc.cx(int(a), int(b))
        else:
            qc.cz(int(a), int(b))
        qc.t(int(rng.integers(num_qubits)))
    qc = transpile(qc, basis_gates=["u", "p", "cx", "cz", "h", "s", "t"])
    return Circuit.from_qiskit(qc)


def recur_subroutine_circuit(num_qubits: int = 4, depth: int = 3) -> Circuit:
    """Circuit invoking a repeated subroutine across layers."""
    sub = QuantumCircuit(num_qubits, name="sub")
    for i in range(num_qubits - 1):
        sub.cx(i, i + 1)
    sub.h(range(num_qubits))
    inst = sub.to_instruction()
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        qc.append(inst, range(num_qubits))
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)
