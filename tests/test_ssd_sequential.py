from quasar.circuit import Circuit, Gate
from quasar.cost import Backend
from quasar.ssd import build_hierarchical_ssd


def build_circuit(*gates: Gate) -> Circuit:
    return Circuit(list(gates), use_classical_simplification=False)


def test_first_fragment_prefers_tableau_for_initial_clifford() -> None:
    circuit = build_circuit(Gate("H", [0]), Gate("X", [1]))

    ssd = build_hierarchical_ssd(circuit)

    assert ssd.partitions
    first = ssd.partitions[0]
    assert first.backend == Backend.TABLEAU
    assert first.history == ("H",)


def test_disjoint_qubits_form_independent_partitions() -> None:
    circuit = build_circuit(Gate("H", [0]), Gate("X", [1]))

    ssd = build_hierarchical_ssd(circuit)

    assert len(ssd.partitions) == 2
    first, second = ssd.partitions

    assert set(first.qubits) == {0}
    assert set(second.qubits) == {1}
    assert first.history == ("H",)
    assert second.history == ("X",)
    assert first.backend == Backend.TABLEAU
    assert second.backend == Backend.TABLEAU
