"""Larger benchmark circuits composed from arithmetic primitives.

This module provides circuits intended to stress-test QuASAr's
partitioning and simulation on wider registers.  The primary entry point is
``ripple_carry_modular_circuit`` which either builds a ripple carry adder or a
naive modular multiplication circuit.  A small non-Clifford subroutine is
included in both cases to demonstrate hybrid behaviour.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import VBERippleCarryAdder, CDKMRippleCarryAdder

from quasar.circuit import Circuit


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
        The assembled circuit ready for benchmarking.
    """

    if bit_width <= 0:
        return Circuit([])

    arithmetic = arithmetic.lower()
    adder_cls = CDKMRippleCarryAdder if arithmetic == "cdkm" else VBERippleCarryAdder

    if modulus is None:
        # Plain ripple-carry adder on two ``bit_width``-wide registers.
        adder = adder_cls(bit_width)
        qc = QuantumCircuit(adder.num_qubits)
        qc.append(adder, range(adder.num_qubits))
        # Small non-Clifford section.
        qc.t(0)
    else:
        n = bit_width
        # ``a`` and ``b`` act as inputs; ``p`` accumulates the product.
        # Total qubits: 3n for the registers plus one extra for carry in the
        # composed adder below.
        qc = QuantumCircuit(3 * n + 1)
        a = list(range(n))
        b = list(range(n, 2 * n))
        prod = list(range(2 * n, 3 * n))
        carry = 3 * n

        # Schoolbook multiplication using controlled additions of the ``a``
        # register into the ``prod`` register conditioned on bits of ``b``.
        adder = adder_cls(n)
        for i in range(n):
            ctrl = b[i]
            # Shifted targets for addition of ``a`` << i
            targets = prod[i: i + n]
            if len(targets) < n:
                # wrap around into higher bits (mod 2**n) for simplicity
                targets = targets + prod[: n - len(targets)]
            qc.compose(
                adder.to_gate().control(1),
                [ctrl, carry, *a, *targets],
                inplace=True,
            )
        # Insert a few T gates as a tiny non-Clifford routine.
        for q in range(min(3, qc.num_qubits)):
            qc.t(q)

        # Simple modular reduction by classically subtracting ``modulus`` once.
        mod_bits = bin(modulus % (1 << n))[2:].zfill(n)[::-1]
        for idx, bit in enumerate(mod_bits):
            if bit == "1":
                qc.x(prod[idx])

    qc = transpile(qc, basis_gates=["u", "p", "cx", "ccx", "h", "x", "t"])
    return Circuit.from_qiskit(qc)


def surface_code_cycle(distance: int, rounds: int = 1, scheme: str = "surface") -> Circuit:
    """Construct repeated stabiliser-measurement cycles for simple error-correction codes.

    Parameters
    ----------
    distance:
        Code distance determining the number of data qubits.
    rounds:
        Number of stabiliser-measurement rounds to repeat.
    scheme:
        ``"surface"`` arranges qubits on a 2-D square lattice while ``"repetition"``
        uses a 1-D chain.

    Returns
    -------
    Circuit
        Circuit implementing ``rounds`` cycles of parity-check measurements.
    """

    if distance <= 0 or rounds <= 0:
        return Circuit([])

    scheme = scheme.lower()
    if scheme == "repetition":
        data_count = distance
        anc_count = distance - 1
        total_qubits = data_count + anc_count
        qc = QuantumCircuit(total_qubits, anc_count * rounds)
        for r in range(rounds):
            for i in range(anc_count):
                anc = data_count + i
                left = i
                right = i + 1
                qc.cx(left, anc)
                qc.cx(right, anc)
                qc.measure(anc, r * anc_count + i)
    else:  # surface code
        d = distance
        data_count = d * d
        horiz = d * (d - 1)
        vert = d * (d - 1)
        anc_count = horiz + vert
        total_qubits = data_count + anc_count
        qc = QuantumCircuit(total_qubits, anc_count * rounds)

        def data_index(row: int, col: int) -> int:
            return row * d + col

        anc_start = data_count
        for r in range(rounds):
            a_offset = 0
            # Horizontal parity checks
            for row in range(d):
                for col in range(d - 1):
                    anc = anc_start + a_offset
                    a_offset += 1
                    q1 = data_index(row, col)
                    q2 = data_index(row, col + 1)
                    qc.h(anc)
                    qc.cz(anc, q1)
                    qc.cz(anc, q2)
                    qc.h(anc)
                    qc.measure(anc, r * anc_count + (a_offset - 1))
            # Vertical parity checks
            for row in range(d - 1):
                for col in range(d):
                    anc = anc_start + a_offset
                    a_offset += 1
                    q1 = data_index(row, col)
                    q2 = data_index(row + 1, col)
                    qc.h(anc)
                    qc.cz(anc, q1)
                    qc.cz(anc, q2)
                    qc.h(anc)
                    qc.measure(anc, r * anc_count + (a_offset - 1))

    qc = transpile(qc, basis_gates=["u", "p", "cx", "ccx", "h", "x", "t", "cz", "measure"])
    return Circuit.from_qiskit(qc)


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

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    controls = list(range(n_qubits - 1))
    target = n_qubits - 1

    for _ in range(iterations):
        # Oracle composed from cascaded Toffoli/CNOT layers.
        qc.h(target)
        for _ in range(oracle_depth):
            if n_qubits > 1:
                qc.mcx(controls, target)
            else:
                qc.x(target)
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
            for q in reversed(range(n_qubits - 1)):
                qc.cx(q, q + 1)
        qc.h(target)

        # Standard Grover diffusion operator.
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(target)
        if n_qubits > 1:
            qc.mcx(controls, target)
        qc.h(target)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc = transpile(qc, basis_gates=["u", "p", "cx", "ccx", "h", "x", "t"])
    return Circuit.from_qiskit(qc)
