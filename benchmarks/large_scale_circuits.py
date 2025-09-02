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
