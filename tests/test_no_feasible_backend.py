import pytest

from quasar import Planner, Circuit, Gate, Backend, NoFeasibleBackendError


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
