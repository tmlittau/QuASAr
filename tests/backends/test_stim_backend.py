"""Tests for the Stim backend initialisation."""

from quasar.backends.stim_backend import StimBackend


def test_load_and_apply_highest_qubit() -> None:
    """Loading three qubits allows operations on the highest index."""
    backend = StimBackend()
    backend.load(3)
    backend.apply_gate("X", [2])
    assert backend.simulator is not None
    assert backend.simulator.num_qubits == 3

