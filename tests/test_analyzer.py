import pytest

from quasar import Circuit, CircuitAnalyzer, Backend, Cost


@pytest.fixture
def sample_circuit():
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "CX", "qubits": [1, 2]},
        {"gate": "H", "qubits": [2]},
    ]
    return Circuit.from_dict(gates)


def test_gate_distribution(sample_circuit):
    analyzer = CircuitAnalyzer(sample_circuit)
    dist = analyzer.gate_distribution()
    assert dist["H"] == 2
    assert dist["CX"] == 2
    assert sum(dist.values()) == 4


def test_entanglement_metrics(sample_circuit):
    analyzer = CircuitAnalyzer(sample_circuit)
    metrics = analyzer.entanglement_metrics()
    assert metrics["multi_qubit_gate_count"] == 2
    # CX gates connect qubits in a chain -> single component of size 3
    assert metrics["connected_components"] == 1
    assert metrics["max_connected_size"] == 3


def test_resource_estimates(sample_circuit):
    analyzer = CircuitAnalyzer(sample_circuit)
    estimates = analyzer.resource_estimates()
    # Ensure estimates for all backends are returned
    assert set(estimates.keys()) == {
        Backend.STATEVECTOR,
        Backend.TABLEAU,
        Backend.MPS,
        Backend.DECISION_DIAGRAM,
    }
    # Basic sanity: costs should be non-negative numbers
    for cost in estimates.values():
        assert isinstance(cost, Cost)
        assert cost.time >= 0
        assert cost.memory >= 0


def test_temporal_metrics():
    circ = Circuit.from_dict(
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "H", "qubits": [1]},
            {"gate": "CX", "qubits": [0, 1]},
        ]
    )
    analysis = CircuitAnalyzer(circ).analyze()
    assert analysis.parallel_layers == [[0, 1], [2]]
    assert analysis.critical_path_length == 2
