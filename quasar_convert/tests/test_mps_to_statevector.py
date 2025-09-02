import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import quasar_convert as qc


def test_mps_to_statevector_contracts_to_expected_statevector():
    eng = qc.ConversionEngine()

    inv_sqrt2 = 1 / np.sqrt(2)
    t1 = [inv_sqrt2, 0, 0, inv_sqrt2]
    t2 = [1.0, 0.0, 0.0, 1.0]
    mps = qc.MPS(tensors=[t1, t2], bond_dims=[1, 2, 1])

    state = eng.mps_to_statevector(mps)

    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0, 1)
    expected = Statevector.from_instruction(circ).data
    expected = np.asarray(expected).reshape([2, 2]).transpose(1, 0).reshape(-1)

    assert np.allclose(state, expected)
