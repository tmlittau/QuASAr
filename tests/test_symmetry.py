from benchmarks.circuits import qft_circuit, w_state_circuit, random_circuit


def test_qft_symmetry_high():
    assert qft_circuit(5).symmetry > 0.3


def test_w_state_symmetry_high():
    assert w_state_circuit(5).symmetry > 0.1


def test_random_circuit_symmetry_low():
    assert random_circuit(5, seed=123).symmetry < 0.05


def test_qft_rotation_diversity_count():
    assert qft_circuit(5).rotation_diversity == 4
