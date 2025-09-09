import math

from benchmarks.circuits import amplitude_estimation_circuit


def test_amplitude_estimation_gate_sequence():
    circ = amplitude_estimation_circuit(2, 0.25)
    theta = 2 * math.asin(math.sqrt(0.25))
    assert [g.gate for g in circ.gates] == [
        "H",
        "H",
        "RY",
        "CRZ",
        "CRZ",
        "H",
        "CRZ",
        "H",
    ]
    assert math.isclose(circ.gates[2].params["theta"], theta)
    assert math.isclose(circ.gates[3].params["phi"], 2 * theta)
    assert math.isclose(circ.gates[4].params["phi"], 4 * theta)
    assert math.isclose(circ.gates[6].params["phi"], -math.pi)
    assert circ.gates[3].qubits == [0, 2]
    assert circ.gates[4].qubits == [1, 2]
    assert circ.gates[6].qubits == [0, 1]
