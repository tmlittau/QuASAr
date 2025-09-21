import pytest

from quasar import (
    Planner,
    Circuit,
    Gate,
    Backend,
    NoFeasibleBackendError,
    SimulationEngine,
)


def _t_gate_circuit():
    # Single non-Clifford gate to avoid tableau fallback
    c = Circuit([Gate("T", [0])], use_classical_simplification=False)
    # Force metrics that make decision diagrams unattractive
    c.sparsity = 0.0
    c.phase_rotation_diversity = 100
    c.amplitude_rotation_diversity = 100
    return c


def test_planner_raises_no_feasible_backend_for_tight_memory():
    planner = Planner()
    circuit = _t_gate_circuit()
    with pytest.raises(NoFeasibleBackendError):
        planner.plan(circuit, max_memory=1000)


def test_planner_selects_backend_when_memory_sufficient():
    planner = Planner()
    circuit = _t_gate_circuit()
    result = planner.plan(circuit, max_memory=10**8)
    assert result.steps[0].backend == Backend.STATEVECTOR


def test_simulation_engine_respects_detected_memory_limit(monkeypatch):
    monkeypatch.setenv("QUASAR_STATEVECTOR_MAX_MEMORY_BYTES", "64")
    engine = SimulationEngine()
    assert engine.memory_threshold == pytest.approx(64.0)

    # Three-qubit circuit exceeds the 64-byte budget required for dense
    # statevectors (128 bytes for amplitudes).
    circuit = Circuit(
        [
            Gate("H", [0]),
            Gate("H", [1]),
            Gate("H", [2]),
            Gate("CX", [0, 1]),
            Gate("T", [2]),
        ],
        use_classical_simplification=False,
    )

    called = {"run": False}

    def _fail_run(*args, **kwargs):  # pragma: no cover - should not execute
        called["run"] = True
        raise AssertionError("scheduler.run should not be invoked when planning fails")

    monkeypatch.setattr(engine.scheduler, "run", _fail_run)

    with pytest.raises(NoFeasibleBackendError):
        engine.simulate(circuit, backend=Backend.STATEVECTOR)

    assert called["run"] is False
