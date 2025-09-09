from benchmarks.circuits import ghz_circuit, w_state_circuit, is_clifford as is_clifford_small
from benchmarks.large_scale_circuits import (
    ripple_carry_modular_circuit,
    surface_code_cycle,
    is_clifford as is_clifford_large,
)


def test_is_clifford_small_circuits():
    assert is_clifford_small(ghz_circuit(3))
    assert not is_clifford_small(w_state_circuit(3))


def test_is_clifford_large_circuits():
    assert not is_clifford_large(ripple_carry_modular_circuit(2, arithmetic="cdkm"))
    assert is_clifford_large(surface_code_cycle(3, rounds=1, scheme="surface"))

