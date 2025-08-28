from quasar import Circuit, SimulationEngine, SSD


def test_simulation_engine_simulate_returns_metrics():
    circuit = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "X", "qubits": [1]},
    ])
    engine = SimulationEngine()
    result = engine.simulate(circuit)
    assert isinstance(result.ssd, SSD)
    # The analysis must record all gates
    assert sum(result.analysis.gate_distribution.values()) == 3
    # Planner produced at least one step
    assert result.plan.steps


def test_partition_state_extraction():
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    engine = SimulationEngine()
    result = engine.simulate(circuit)
    states = [result.ssd.extract_state(p) for p in result.ssd.partitions]
    assert any(s is not None for s in states)
