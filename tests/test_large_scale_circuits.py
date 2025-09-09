import networkx as nx

from quasar.circuit import Gate
from benchmarks.large_scale_circuits import (
    ripple_carry_modular_circuit,
    surface_code_cycle,
    deep_qaoa_circuit,
    phase_estimation_classical_unitary,
    surface_corrected_qaoa,
)
from benchmarks.circuits import _cdkm_adder_gates


def test_ripple_carry_adder_cdkm():
    n = 2
    circ = ripple_carry_modular_circuit(n, arithmetic="cdkm")
    expected = _cdkm_adder_gates(n) + [Gate("T", [0])]
    assert circ.gates == expected


def test_ripple_carry_modular_has_controls():
    n = 2
    circ = ripple_carry_modular_circuit(n, modulus=3, arithmetic="cdkm")
    assert any(g.gate.startswith("C") for g in circ.gates)
    prod_start = 2 * n
    assert any(g.gate == "X" and g.qubits[0] >= prod_start for g in circ.gates)


def test_surface_code_cycle_repetition():
    circ = surface_code_cycle(3, rounds=1, scheme="repetition")
    expected = [
        Gate("CX", [0, 3]),
        Gate("CX", [1, 3]),
        Gate("CX", [1, 4]),
        Gate("CX", [2, 4]),
    ]
    assert circ.gates == expected


def test_surface_code_cycle_surface():
    circ = surface_code_cycle(2, rounds=1, scheme="surface")
    assert len(circ.gates) == 16
    assert [g.gate for g in circ.gates[:4]] == ["H", "CZ", "CZ", "H"]


def test_deep_qaoa_circuit_structure():
    g = nx.cycle_graph(3)
    circ = deep_qaoa_circuit(g, p_layers=2)
    assert len(circ.gates) == 12
    assert all(gate.gate in {"RZZ", "RX"} for gate in circ.gates)


def test_surface_corrected_qaoa_interleaves_cycles():
    circ = surface_corrected_qaoa(3, distance=2, rounds=2)
    assert len(circ.gates) == 44
    assert [g.gate for g in circ.gates[:6]] == ["RZZ", "RZZ", "RZZ", "RX", "RX", "RX"]
    assert circ.gates[6].gate == "H"
    assert circ.gates[22].gate == "RZZ"


def test_phase_estimation_gate_count():
    circ = phase_estimation_classical_unitary(2, 2, 1)
    assert len(circ.gates) == 8
    assert [g.gate for g in circ.gates[:2]] == ["H", "H"]
    assert circ.gates[2].gate.startswith("C")

