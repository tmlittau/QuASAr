"""Common benchmark circuits for QuASAr."""
from __future__ import annotations

import math
from typing import List

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT

from quasar.circuit import Circuit, Gate


def ghz_circuit(n_qubits: int) -> Circuit:
    """Create an ``n_qubits`` GHZ state preparation circuit."""
    gates: List[Gate] = []
    if n_qubits <= 0:
        return Circuit(gates)
    gates.append(Gate("H", [0]))
    for i in range(1, n_qubits):
        gates.append(Gate("CX", [i - 1, i]))
    return Circuit(gates)


def qft_circuit(n_qubits: int) -> Circuit:
    """Create an ``n_qubits`` quantum Fourier transform circuit."""
    qc = QFT(n_qubits)
    qc = transpile(qc, basis_gates=["u", "p", "cx", "swap", "h"])
    return Circuit.from_qiskit(qc)


def qft_on_ghz_circuit(n_qubits: int) -> Circuit:
    """Apply the QFT to a GHZ state."""
    ghz = ghz_circuit(n_qubits)
    qft = qft_circuit(n_qubits)
    return Circuit(list(ghz.gates) + list(qft.gates))


def w_state_circuit(n_qubits: int) -> Circuit:
    """Create an ``n_qubits`` W state preparation circuit."""
    qc = QuantumCircuit(n_qubits)
    state = np.zeros(2**n_qubits)
    for i in range(n_qubits):
        state[1 << i] = 1 / math.sqrt(n_qubits)
    qc.initialize(state, range(n_qubits))
    qc = transpile(qc, basis_gates=["u", "cx"])
    new_qc = QuantumCircuit(n_qubits)
    for inst, qargs, cargs in qc.data:
        if inst.name == "reset":
            continue
        new_qc.append(inst, qargs, cargs)
    return Circuit.from_qiskit(new_qc)


def grover_circuit(n_qubits: int, n_iterations: int = 1) -> Circuit:
    """Create a Grover search circuit targeting the all-ones state.

    Args:
        n_qubits: Number of search qubits.
        n_iterations: Number of Grover iterations to apply.

    Returns:
        A :class:`Circuit` implementing the algorithm.
    """

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for _ in range(n_iterations):
        # Oracle marking the all-ones state
        qc.h(n_qubits - 1)
        if n_qubits > 1:
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

        # Diffuser
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        if n_qubits > 1:
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)


def bernstein_vazirani_circuit(n_qubits: int, secret: int = 0) -> Circuit:
    """Create a Bernstein-Vazirani circuit for a given secret string.

    Args:
        n_qubits: Number of secret bits.
        secret: Integer encoding the secret string (little-endian).

    Returns:
        A :class:`Circuit` implementing the algorithm.
    """

    qc = QuantumCircuit(n_qubits + 1)
    qc.h(range(n_qubits))
    qc.x(n_qubits)
    qc.h(n_qubits)

    for i in range(n_qubits):
        if (secret >> i) & 1:
            qc.cx(i, n_qubits)

    qc.h(range(n_qubits))
    qc = transpile(qc, basis_gates=["u", "p", "cx", "h", "x"])
    return Circuit.from_qiskit(qc)
