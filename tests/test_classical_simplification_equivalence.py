from benchmarks.circuits import ghz_circuit, qft_circuit, w_state_circuit
from quasar.circuit import Gate


def test_classical_simplification_equivalence() -> None:
    n = 4

    ghz = ghz_circuit(n)
    ghz_original = list(ghz.gates)
    ghz.enable_classical_simplification()
    assert ghz.gates == ghz_original

    w_state = w_state_circuit(n)
    w_original = list(w_state.gates)
    w_state.enable_classical_simplification()
    assert w_state.gates == w_original

    qft = qft_circuit(n)
    qft_original = list(qft.gates)
    qft.enable_classical_simplification()
    assert qft.gates != qft_original
    expected = [Gate("H", [i]) for i in reversed(range(n))]
    assert qft.gates == expected
