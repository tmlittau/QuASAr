import pytest

from quasar import Circuit, SimulationEngine, SSD, Backend


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
    # Timing metrics are recorded
    assert isinstance(result.analysis_time, float)
    assert isinstance(result.planning_time, float)
    assert isinstance(result.execution_time, float)
    assert isinstance(result.backend_switches, int)
    assert isinstance(result.conversion_durations, list)
    assert isinstance(result.plan_cache_hits, int)


def test_partition_state_extraction():
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    engine = SimulationEngine()
    result = engine.simulate(circuit)
    states = [result.ssd.extract_state(p) for p in result.ssd.partitions]
    assert any(s is not None for s in states)


def test_memory_threshold_triggers_adaptive_plan():
    circuit = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "X", "qubits": [1]},
    ])
    high = SimulationEngine().simulate(circuit, memory_threshold=1000)
    low = SimulationEngine().simulate(circuit, memory_threshold=1)
    assert len(low.plan.steps) <= len(high.plan.steps)


def test_backend_selection():
    circuit = Circuit([
        {"gate": "H", "qubits": [0]},
        {"gate": "CX", "qubits": [0, 1]},
        {"gate": "S", "qubits": [1]},
    ])
    engine = SimulationEngine()
    result = engine.simulate(circuit, backend=Backend.TABLEAU)
    assert all(step.backend == Backend.TABLEAU for step in result.plan.steps)


def test_simulate_uses_supplied_plan(monkeypatch):
    circuit = Circuit([{ "gate": "H", "qubits": [i] } for i in range(13)])
    engine = SimulationEngine()
    calls = {"count": 0}

    original_plan = engine.planner.plan

    def counting_plan(*args, **kwargs):
        calls["count"] += 1
        return original_plan(*args, **kwargs)

    monkeypatch.setattr(engine.planner, "plan", counting_plan)
    engine.simulate(circuit)
    assert calls["count"] == 1


def test_simulate_forwards_constraints(monkeypatch):
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    engine = SimulationEngine()
    received: dict[str, float | int | None] = {}

    original_plan = engine.planner.plan

    def plan_spy(*args, **kwargs):
        received["target_accuracy"] = kwargs.get("target_accuracy")
        received["max_time"] = kwargs.get("max_time")
        received["optimization_level"] = kwargs.get("optimization_level")
        return original_plan(*args, **kwargs)

    monkeypatch.setattr(engine.planner, "plan", plan_spy)

    original_run = engine.scheduler.run

    def run_spy(*args, **kwargs):
        received["run_max_time"] = kwargs.get("max_time")
        return original_run(*args, **kwargs)

    monkeypatch.setattr(engine.scheduler, "run", run_spy)

    engine.simulate(
        circuit,
        target_accuracy=0.9,
        max_time=1000.0,
        optimization_level=2,
    )

    assert received["target_accuracy"] == 0.9
    assert received["max_time"] == 1000.0
    assert received["optimization_level"] == 2
    assert received["run_max_time"] == 1000.0


def test_simulation_enforces_max_time():
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    engine = SimulationEngine()
    with pytest.raises(ValueError):
        engine.simulate(circuit, max_time=1e-9)


def test_plan_cache_hits_accumulate():
    circuit = Circuit([{ "gate": "H", "qubits": [0] }])
    engine = SimulationEngine()
    first = engine.simulate(circuit, backend=Backend.STATEVECTOR, optimization_level=0)
    second = engine.simulate(circuit, backend=Backend.STATEVECTOR, optimization_level=0)
    assert first.plan_cache_hits == 0
    assert second.plan_cache_hits >= 1


def test_fidelity_against_reference():
    circuit = Circuit([{"gate": "H", "qubits": [0]}])
    engine = SimulationEngine()
    import numpy as np

    ref = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    result = engine.simulate(circuit, backend=Backend.STATEVECTOR, reference_state=ref)
    assert result.fidelity is not None and abs(result.fidelity - 1.0) < 1e-6
