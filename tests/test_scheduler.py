import numpy as np

from quasar.scheduler import _tensor_statevectors


def _reference_tensor_statevectors(
    left_state: np.ndarray,
    left_qubits: tuple[int, ...],
    right_state: np.ndarray,
    right_qubits: tuple[int, ...],
    merged_qubits: tuple[int, ...],
) -> np.ndarray:
    """Replicate the historical Python implementation for regression checks."""

    num_qubits = len(merged_qubits)
    result = np.zeros(1 << num_qubits, dtype=complex)
    left_positions = [merged_qubits.index(q) for q in left_qubits]
    right_positions = [merged_qubits.index(q) for q in right_qubits]

    left_state = np.asarray(left_state, dtype=complex)
    right_state = np.asarray(right_state, dtype=complex)

    for basis in range(1 << num_qubits):
        bits = [(basis >> i) & 1 for i in range(num_qubits)]
        left_index = 0
        for offset, pos in enumerate(left_positions):
            left_index |= bits[pos] << offset
        right_index = 0
        for offset, pos in enumerate(right_positions):
            right_index |= bits[pos] << offset
        amplitude_left = left_state[left_index] if left_positions else 1.0
        amplitude_right = right_state[right_index] if right_positions else 1.0
        result[basis] = amplitude_left * amplitude_right

    return result


def test_tensor_statevectors_matches_reference():
    cases = [
        (
            np.array([1 + 1j, 2 - 1j]),
            (0,),
            np.array([3 + 0.5j, -1j, 0.25 + 2j, 4 - 3j]),
            (2, 1),
            (0, 2, 1),
        ),
        (
            np.array([2 - 1j, -0.5 + 0.25j, 3 + 4j, 0.75 - 2j]),
            (2, 0),
            np.array([1 - 2j, 0.5 + 0.5j]),
            (1,),
            (2, 0, 1),
        ),
        (
            np.array([0.5 - 0.25j]),
            tuple(),
            np.array([1 + 0j, -1j, 0.75 + 0.5j, 2 - 0.75j]),
            (3, 1),
            (3, 1),
        ),
        (
            np.array([1 + 0.25j, -0.5 - 0.5j, 2 + 3j, -1 + 0.75j]),
            (1, 3),
            np.array([0.25 - 1j]),
            tuple(),
            (1, 3),
        ),
    ]

    for left_state, left_qubits, right_state, right_qubits, merged in cases:
        expected = _reference_tensor_statevectors(
            left_state, left_qubits, right_state, right_qubits, merged
        )
        result = _tensor_statevectors(
            left_state, left_qubits, right_state, right_qubits, merged
        )
        np.testing.assert_allclose(result, expected)
