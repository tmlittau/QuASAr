"""Common benchmark circuits for QuASAr."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

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


def _iqft_spec(n: int) -> List[Gate]:
    """Return a list of :class:`Gate` objects for the inverse QFT."""
    gates: List[Gate] = []
    for gate in reversed(_qft_spec(n)):
        if gate.gate == "H":
            gates.append(Gate("H", gate.qubits))
        elif gate.gate == "CP":
            k = float(gate.params.get("k", 0))
            phi = -2 * math.pi / (2**k)
            gates.append(Gate("CRZ", gate.qubits, {"phi": phi}))
        else:
            raise ValueError(f"Unsupported gate {gate.gate} in QFT spec")
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

    theta = 2 * math.asin(math.sqrt(probability))
    gates: List[Gate] = []
    for q in range(num_qubits):
        gates.append(Gate("H", [q]))
    gates.append(Gate("RY", [num_qubits], {"theta": theta}))
    for i in range(num_qubits):
        phi = (2 ** i) * 2 * theta
        gates.append(Gate("CRZ", [i, num_qubits], {"phi": phi}))
    gates.extend(_iqft_spec(num_qubits))
    return Circuit(gates)


def bmw_quark_circuit(num_qubits: int, depth: int, kind: str = "cardinality") -> Circuit:
    """Generate BMW-QUARK ansatz circuits.

    Args:
        num_qubits: Number of qubits.
        depth: Number of alternating rotation/entangling layers.
        kind: ``"cardinality"`` or ``"circular"`` ansatz style.

    Returns:
        The requested ansatz circuit as a :class:`Circuit`.
    """

    gates: List[Gate] = []
    theta = np.pi / 2
    for _ in range(depth):
        for q in range(num_qubits):
            gates.append(Gate("RX", [q], {"theta": theta}))
        if kind == "cardinality":
            for q in range(0, num_qubits - 1, 2):
                gates.append(Gate("RXX", [q, q + 1], {"theta": theta}))
            for q in range(1, num_qubits - 1, 2):
                gates.append(Gate("RXX", [q, q + 1], {"theta": theta}))
        else:  # circular
            for q in range(num_qubits):
                gates.append(Gate("RXX", [q, (q + 1) % num_qubits], {"theta": theta}))
    return Circuit(gates)


def _cdkm_adder_gates(n: int) -> List[Gate]:
    """Gate sequence for the CDKM ripple-carry adder.

    This implements the ``full`` variant of the adder from
    Cuccaro et al. [quant-ph/0410184], acting on two ``n``-qubit
    registers ``a`` and ``b`` with an additional carry qubit at the
    beginning and the end of the layout.  The result is stored in
    ``b`` while ``a`` is restored to its input value.
    """

    if n <= 0:
        return []

    gates: List[Gate] = []
    cin = 0
    a_start = 1
    b_start = 1 + n
    cout = 1 + 2 * n

    def maj(a: int, b: int, c: int) -> None:
        gates.append(Gate("CX", [a, b]))
        gates.append(Gate("CX", [a, c]))
        gates.append(Gate("CCX", [c, b, a]))

    def uma(a: int, b: int, c: int) -> None:
        gates.append(Gate("CCX", [c, b, a]))
        gates.append(Gate("CX", [a, c]))
        gates.append(Gate("CX", [c, b]))

    maj(a_start, b_start, cin)
    for i in range(n - 1):
        maj(a_start + i + 1, b_start + i + 1, a_start + i)

    gates.append(Gate("CX", [a_start + n - 1, cout]))

    for i in reversed(range(n - 1)):
        uma(a_start + i + 1, b_start + i + 1, a_start + i)
    uma(a_start, b_start, cin)

    return gates


def _iqft_gates(n: int, offset: int) -> List[Gate]:
    """Inverse QFT on ``n`` qubits starting at ``offset``."""

    gates: List[Gate] = []
    for j in reversed(range(n)):
        qubit = offset + j
        for k in range(j):
            ctrl = offset + k
            angle = -math.pi / (2 ** (j - k))
            gates.append(Gate("CRZ", [ctrl, qubit], {"theta": angle}))
        gates.append(Gate("H", [qubit]))
    return gates


def _qft_gates(n: int, offset: int) -> List[Gate]:
    """QFT on ``n`` qubits starting at ``offset``."""

    gates: List[Gate] = []
    for j in range(n):
        qubit = offset + j
        gates.append(Gate("H", [qubit]))
        for k in range(j + 1, n):
            ctrl = offset + k
            angle = math.pi / (2 ** (k - j))
            gates.append(Gate("CRZ", [ctrl, qubit], {"theta": angle}))
    return gates


def _draper_adder_gates(n: int) -> List[Gate]:
    """Gate sequence for the Draper QFT adder (fixed-point variant)."""

    if n <= 0:
        return []

    gates: List[Gate] = []
    a_start = 0
    b_start = n

    gates.extend(_qft_gates(n, b_start))

    for j in range(n):
        for k in range(n - j):
            ctrl = a_start + j
            tgt = b_start + j + k
            angle = math.pi / (2 ** k)
            gates.append(Gate("CRZ", [ctrl, tgt], {"theta": angle}))

    gates.extend(_iqft_gates(n, b_start))
    return gates


def _vbe_adder_gates(n: int) -> List[Gate]:
    """Gate sequence for the VBE ripple-carry adder."""

    if n <= 0:
        return []

    gates: List[Gate] = []
    cin = 0
    a_start = 1
    b_start = 1 + n
    cout = 1 + 2 * n
    helpers = [cout + 1 + i for i in range(max(0, n - 1))]
    carries = [cin] + helpers + [cout]

    i = 0
    for inp, out in zip(carries[:-1], carries[1:]):
        a_i = a_start + i
        b_i = b_start + i
        gates.append(Gate("CCX", [a_i, b_i, out]))
        gates.append(Gate("CX", [a_i, b_i]))
        gates.append(Gate("CCX", [inp, b_i, out]))
        i += 1

    gates.append(Gate("CX", [a_start + n - 1, b_start + n - 1]))
    if len(carries) > 1:
        inp = carries[-2]
        a_i = a_start + n - 1
        b_i = b_start + n - 1
        gates.append(Gate("CX", [a_i, b_i]))
        gates.append(Gate("CX", [inp, b_i]))

    i -= 2
    for j, (inp, out) in enumerate(
        zip(reversed(carries[:-1]), reversed(carries[1:]))
    ):
        if j == 0:
            continue
        a_i = a_start + i
        b_i = b_start + i
        gates.append(Gate("CCX", [inp, b_i, out]))
        gates.append(Gate("CX", [a_i, b_i]))
        gates.append(Gate("CCX", [a_i, b_i, out]))
        gates.append(Gate("CX", [inp, b_i]))
        gates.append(Gate("CX", [a_i, b_i]))
        i -= 1

    return gates


def adder_circuit(num_qubits: int, kind: str = "cdkm") -> Circuit:
    """Construct CDKM, Draper or VBE adder circuits.

    Args:
        num_qubits: Number of bits in each addend.
        kind: ``"cdkm"``, ``"draper"`` or ``"vbe"``.
    """

    kind = kind.lower()
    if kind == "cdkm":
        gates = _cdkm_adder_gates(num_qubits)
    elif kind == "draper":
        gates = _draper_adder_gates(num_qubits)
    elif kind == "vbe":
        gates = _vbe_adder_gates(num_qubits)
    else:
        raise ValueError("Unknown adder kind")
    return Circuit(gates, use_classical_simplification=False)


def deutsch_jozsa_circuit(num_qubits: int, balanced: bool = True) -> Circuit:
    """Construct a Deutsch-Jozsa circuit.

    Args:
        num_qubits: Number of input bits.
        balanced: Whether to use a balanced oracle; otherwise constant.
    """

    gates: List[Gate] = []
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")
    # Prepare ancilla in |-> state
    anc = num_qubits
    gates.append(Gate("X", [anc]))
    for q in range(num_qubits + 1):
        gates.append(Gate("H", [q]))
    # Balanced oracle uses CNOTs from each input to ancilla
    if balanced:
        for i in range(num_qubits):
            gates.append(Gate("CX", [i, anc]))
    # Decode the result
    for q in range(num_qubits):
        gates.append(Gate("H", [q]))
    return Circuit(gates)


def graph_state_circuit(num_qubits: int, degree: int, seed: int | None = None) -> Circuit:
    """Generate a random regular graph state circuit."""

    import networkx as nx

    if degree >= num_qubits:
        raise ValueError("degree must be < num_qubits")
    if (num_qubits * degree) % 2 != 0:
        raise ValueError("n * degree must be even for a regular graph")
    g = nx.random_regular_graph(degree, num_qubits, seed=seed)
    gates: List[Gate] = []
    for q in range(num_qubits):
        gates.append(Gate("H", [q]))
    for u, v in g.edges():
        gates.append(Gate("CZ", [u, v]))
    return Circuit(gates)


def hhl_circuit(num_qubits: int) -> Circuit:
    """Create a simple HHL circuit. Requires ``num_qubits >= 3``."""

    if num_qubits < 3:
        raise ValueError("HHL requires at least 3 qubits")
    gates: List[Gate] = []
    gates.append(Gate("H", [0]))
    gates.append(Gate("CX", [0, 1]))
    gates.append(Gate("H", [0]))
    gates.append(Gate("CX", [1, 2]))
    gates.append(Gate("RY", [2], {"theta": math.pi / 4}))
    gates.append(Gate("CX", [1, 2]))
    gates.append(Gate("H", [0]))
    return Circuit(gates)


def hrs_circuit(num_qubits: int) -> Circuit:
    """Construct a toy HRS arithmetic circuit."""

    if num_qubits % 2 != 0:
        raise ValueError("num_qubits must be even for HRS circuit")
    gates: List[Gate] = []
    for i in range(0, num_qubits, 2):
        target = (i + 2) % num_qubits
        gates.append(Gate("CCX", [i, i + 1, target]))
    return Circuit(gates)


def qaoa_circuit(num_qubits: int, repetitions: int = 1, seed: int | None = None) -> Circuit:
    """Create a basic QAOA circuit on a ring graph."""

    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0, 2 * np.pi, size=repetitions)
    betas = rng.uniform(0, 2 * np.pi, size=repetitions)
    edges = [(i, (i + 1) % num_qubits) for i in range(num_qubits)]
    gates: List[Gate] = []
    for q in range(num_qubits):
        gates.append(Gate("H", [q]))
    for p in range(repetitions):
        for u, v in edges:
            gates.append(Gate("RZZ", [u, v], {"theta": float(gammas[p])}))
        for q in range(num_qubits):
            gates.append(Gate("RX", [q], {"theta": float(betas[p])}))
    return Circuit(gates)


def qnn_circuit(num_qubits: int) -> Circuit:
    """Construct a simple quantum neural network circuit."""

    gates: List[Gate] = []
    param = 0
    # ZZFeatureMap: Hadamards followed by pairwise ZZ interactions
    for q in range(num_qubits):
        gates.append(Gate("H", [q]))
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            param += 1
            gates.append(Gate("RZZ", [i, j], {"theta": 0.1 * param}))
    # RealAmplitudes ansatz with a single repetition
    for q in range(num_qubits):
        param += 1
        gates.append(Gate("RY", [q], {"theta": 0.1 * param}))
    for i in range(num_qubits - 1):
        gates.append(Gate("CX", [i, i + 1]))
    for q in range(num_qubits):
        param += 1
        gates.append(Gate("RY", [q], {"theta": 0.1 * param}))
    return Circuit(gates)


def qpe_circuit(num_qubits: int, inexact: bool = False) -> Circuit:
    """Quantum phase estimation circuit.

    Args:
        num_qubits: Number of counting qubits.
        inexact: Whether to use an inexact eigenphase.
    """

    gates: List[Gate] = []
    theta = 2 * math.pi / (2**num_qubits)
    if inexact:
        theta *= 1.1
    for q in range(num_qubits):
        gates.append(Gate("H", [q]))
    for j in range(num_qubits):
        phi = (2**j) * theta
        gates.append(Gate("CRZ", [j, num_qubits], {"phi": phi}))
    gates.extend(_iqft_spec(num_qubits))
    return Circuit(gates)


def quantum_walk_circuit(num_qubits: int, depth: int) -> Circuit:
    """Construct a simple quantum walk circuit."""

    gates: List[Gate] = []
    for q in range(num_qubits):
        gates.append(Gate("H", [q]))
    mcx_name = "C" * (num_qubits - 1) + "X" if num_qubits > 1 else "X"
    for _ in range(depth):
        if num_qubits > 1:
            qubits = list(range(num_qubits))
            gates.append(Gate(mcx_name, qubits))
        for q in range(num_qubits):
            gates.append(Gate("H", [q]))
    return Circuit(gates)


def random_circuit(
    num_qubits: int,
    seed: int | None = None,
    *,
    use_classical_simplification: bool = False,
) -> Circuit:
    """Generate a random circuit of depth ``2*num_qubits``."""

    rng = np.random.default_rng(seed)
    gates: List[Gate] = []
    depth = 2 * num_qubits
    single_gates = ["H", "RX", "RY"]
    two_gates = ["CX", "CZ", "RZZ"]
    for _ in range(depth):
        for q in range(num_qubits):
            gate = rng.choice(single_gates)
            if gate == "H":
                gates.append(Gate("H", [q]))
            else:
                angle = float(rng.uniform(0, 2 * np.pi))
                gates.append(Gate(gate, [q], {"theta": angle}))
        if num_qubits > 1:
            num_two = int(rng.integers(1, num_qubits))
            used: set[Tuple[int, int]] = set()
            for _ in range(num_two):
                a, b = rng.choice(num_qubits, 2, replace=False)
                if (a, b) in used or (b, a) in used:
                    continue
                used.add((a, b))
                gate = rng.choice(two_gates)
                if gate == "RZZ":
                    angle = float(rng.uniform(0, 2 * np.pi))
                    gates.append(Gate("RZZ", [int(a), int(b)], {"theta": angle}))
                else:
                    gates.append(Gate(gate, [int(a), int(b)]))
    return Circuit(gates, use_classical_simplification=use_classical_simplification)


def shor_circuit(circuit_size: int) -> Circuit:
    """Create a toy Shor factoring circuit of a given size."""

    gates: List[Gate] = []
    for q in range(circuit_size):
        gates.append(Gate("H", [q]))
    if circuit_size > 1:
        gates.append(Gate("CX", [0, circuit_size - 1]))
    gates.extend(_qft_spec(circuit_size))
    return Circuit(gates)


def _entanglement_pairs(
    num_qubits: int, entanglement: str | List[int] | List[List[int]]
) -> List[Tuple[int, int]]:
    """Return pairs of qubits to entangle for ansatz circuits."""

    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        if entanglement == "linear":
            return [(i, i + 1) for i in range(num_qubits - 1)]
        raise ValueError("Unsupported entanglement pattern")
    pairs: List[Tuple[int, int]] = []
    if not entanglement:
        return pairs
    if isinstance(entanglement[0], (list, tuple)):
        for a, b in entanglement:
            pairs.append((int(a), int(b)))
    else:
        for a, b in zip(entanglement[:-1], entanglement[1:]):
            pairs.append((int(a), int(b)))
    return pairs


def _two_local_gates(
    num_qubits: int,
    reps: int,
    entanglement: str | List[int] | List[List[int]],
    rotation_blocks: List[str],
) -> List[Gate]:
    """Return gates for a generic two-local style ansatz."""

    pairs = _entanglement_pairs(num_qubits, entanglement)
    gates: List[Gate] = []
    param = 0
    # Initial rotation layer
    for q in range(num_qubits):
        for block in rotation_blocks:
            param += 1
            name = "phi" if block == "RZ" else "theta"
            gates.append(Gate(block, [q], {name: 0.1 * param}))
    for _ in range(reps):
        for a, b in pairs:
            gates.append(Gate("CX", [a, b]))
        for q in range(num_qubits):
            for block in rotation_blocks:
                param += 1
                name = "phi" if block == "RZ" else "theta"
                gates.append(Gate(block, [q], {name: 0.1 * param}))
    return gates


def real_amplitudes_circuit(
    num_qubits: int, reps: int = 1, entanglement: str | List[int] | List[List[int]] = "full"
) -> Circuit:
    """Construct a ``RealAmplitudes`` ansatz with bound parameters."""

    gates = _two_local_gates(num_qubits, reps, entanglement, ["RY"])
    return Circuit(gates)


def efficient_su2_circuit(
    num_qubits: int, reps: int = 1, entanglement: str | List[int] | List[List[int]] = "full"
) -> Circuit:
    """Construct an ``EfficientSU2`` ansatz with bound parameters."""

    gates = _two_local_gates(num_qubits, reps, entanglement, ["RY", "RZ"])
    return Circuit(gates)


def two_local_circuit(
    num_qubits: int, reps: int = 1, entanglement: str | List[int] | List[List[int]] = "full"
) -> Circuit:
    """Construct a ``TwoLocal`` ansatz with bound parameters."""

    gates = _two_local_gates(num_qubits, reps, entanglement, ["RY", "RZ"])
    return Circuit(gates)


def clifford_ec_circuit() -> Circuit:
    """Three-qubit bit-flip error-correction circuit using Clifford gates.

    Measurement operations are omitted so the circuit contains only unitary gates.
    """
    gates: List[Gate] = []
    # Encode logical qubit into three data qubits
    gates.append(Gate("H", [0]))
    gates.append(Gate("CX", [0, 1]))
    gates.append(Gate("CX", [0, 2]))
    # Syndrome extraction with two ancilla qubits
    gates.append(Gate("CX", [0, 3]))
    gates.append(Gate("CX", [1, 3]))
    gates.append(Gate("CX", [1, 4]))
    gates.append(Gate("CX", [2, 4]))
    return Circuit(gates)


def ripple_add_circuit(num_bits: int = 4) -> Circuit:
    """Ripple-carry adder for two ``num_bits``-bit registers."""

    return adder_circuit(num_bits, kind="vbe")


def vqe_chain_circuit(num_qubits: int = 6, depth: int = 2) -> Circuit:
    """Parameterized VQE ansatz with linear entanglement chain."""

    gates = _two_local_gates(num_qubits, depth, "linear", ["RY", "RZ"])
    return Circuit(gates, use_classical_simplification=False)


def random_hybrid_circuit(num_qubits: int = 6, depth: int = 10, seed: int | None = None) -> Circuit:
    """Random circuit mixing Clifford and non-Clifford gates."""
    rng = np.random.default_rng(seed)
    gates: List[Gate] = []
    for _ in range(depth):
        for q in range(num_qubits):
            gate = rng.choice(["H", "S"])
            gates.append(Gate(gate, [q]))
        if num_qubits > 1:
            a, b = rng.choice(num_qubits, 2, replace=False)
            two_gate = "CX" if rng.random() < 0.5 else "CZ"
            gates.append(Gate(two_gate, [int(a), int(b)]))
        gates.append(Gate("T", [int(rng.integers(num_qubits))]))
    return Circuit(gates)


def recur_subroutine_circuit(num_qubits: int = 4, depth: int = 3) -> Circuit:
    """Circuit invoking a repeated subroutine across layers."""

    gates: List[Gate] = []
    for _ in range(depth):
        for i in range(num_qubits - 1):
            gates.append(Gate("CX", [i, i + 1]))
        for q in range(num_qubits):
            gates.append(Gate("H", [q]))
    return Circuit(gates)
