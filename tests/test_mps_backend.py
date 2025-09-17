from __future__ import annotations

import numpy as np

from quasar.backends.mps import MPSBackend, tensor_product
from quasar.backends.statevector import StatevectorBackend


def test_tensor_product_matches_statevector() -> None:
    left = MPSBackend()
    left.load(1)
    left.apply_gate("H", [0])
    left_state = left._run()
    left_vec = left._mps_to_statevector(left_state)

    right = MPSBackend()
    right.load(1)
    right.apply_gate("X", [0])
    right_state = right._run()
    right_vec = right._mps_to_statevector(right_state)

    combined = tensor_product(left_state, right_state)
    combo_backend = MPSBackend()
    combo_backend.num_qubits = 2
    combo_vec = combo_backend._mps_to_statevector(combined)

    np.testing.assert_allclose(combo_vec, np.kron(left_vec, right_vec))


def test_ingest_mps_with_mapping_matches_reference_statevector() -> None:
    source = MPSBackend()
    source.load(2)
    source.apply_gate("H", [0])
    source.apply_gate("CX", [0, 1])
    bell_state = source._run()

    backend = MPSBackend()
    backend.ingest(bell_state, num_qubits=4, mapping=[1, 3])
    backend_vec = backend.statevector()

    reference = StatevectorBackend()
    reference.load(4)
    reference.apply_gate("H", [1])
    reference.apply_gate("CX", [1, 3])
    expected = reference.statevector()

    np.testing.assert_allclose(backend_vec, expected)
