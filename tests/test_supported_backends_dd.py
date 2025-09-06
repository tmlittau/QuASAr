from benchmarks.circuits import random_circuit, w_state_circuit, qft_circuit
from quasar.cost import Backend
from quasar.planner import _supported_backends


def test_supported_backends_sparse_adds_dd():
    circ = w_state_circuit(5)
    backends = _supported_backends(circ.gates, sparsity=circ.sparsity)
    assert Backend.DECISION_DIAGRAM in backends


def test_supported_backends_random_excludes_dd():
    circ = random_circuit(5, seed=123)
    backends = _supported_backends(circ.gates, sparsity=circ.sparsity)
    assert Backend.DECISION_DIAGRAM not in backends


def test_supported_backends_qft_rotation_diversity():
    circ = qft_circuit(5)
    backends = _supported_backends(
        circ.gates,
        sparsity=circ.sparsity,
        phase_rotation_diversity=circ.phase_rotation_diversity,
        amplitude_rotation_diversity=circ.amplitude_rotation_diversity,
    )
    assert Backend.DECISION_DIAGRAM not in backends
