from benchmarks.circuits import ghz_circuit, qft_circuit, w_state_circuit
from quasar import Backend, SimulationEngine
from quasar.circuit import Circuit
from quasar.planner import Planner
from quasar.cost import CostEstimator
import quasar.config as config


def test_planner_selects_tableau_for_ghz():
    circuit = ghz_circuit(4)
    engine = SimulationEngine()
    plan = engine.planner.plan(circuit)
    assert plan.final_backend == Backend.TABLEAU
    assert {s.backend for s in plan.steps} == {Backend.TABLEAU}


def test_planner_selects_mps_for_qft():
    circuit = qft_circuit(5)
    circuit.symmetry = 0.0
    circuit.sparsity = 0.0
    est = CostEstimator(
        coeff={
            "sv_gate_1q": 50.0,
            "sv_gate_2q": 50.0,
            "mps_base_time": 0.0,
            "mps_base_mem": 0.0,
        }
    )
    engine = SimulationEngine(estimator=est)
    plan = engine.planner.plan(circuit)
    assert plan.final_backend == Backend.MPS
    assert Backend.MPS in {s.backend for s in plan.steps}


def test_planner_selects_dd_when_sparsity_weighted(monkeypatch):
    circuit = w_state_circuit(5)
    monkeypatch.setattr(config.DEFAULT, "dd_sparsity_weight", 1.2)
    monkeypatch.setattr(config.DEFAULT, "dd_metric_threshold", 0.9)
    engine = SimulationEngine()
    plan = engine.planner.plan(circuit)
    assert plan.final_backend == Backend.DECISION_DIAGRAM


def test_w_state_amplitude_threshold_scales(monkeypatch):
    base = config.DEFAULT.dd_amplitude_rotation_diversity_threshold
    circuit = w_state_circuit(base + 5)
    # Skew the estimator to strongly prefer decision diagrams over MPS.
    est = CostEstimator(
        coeff={
            "dd_gate": 1e-6,
            "dd_mem": 1e-6,
            "mps_gate_1q": 1000.0,
            "mps_gate_2q": 1000.0,
            "mps_mem": 1000.0,
            "mps_temp_mem": 1000.0,
        }
    )
    engine = SimulationEngine(estimator=est)
    plan = engine.planner.plan(circuit)
    assert plan.final_backend == Backend.DECISION_DIAGRAM


def test_mps_target_fidelity_controls_selection(monkeypatch):
    circuit = qft_circuit(7)
    circuit.symmetry = 0.0
    circuit.sparsity = 0.0
    est = CostEstimator(
        coeff={
            "sv_gate_1q": 50.0,
            "sv_gate_2q": 50.0,
            "mps_base_time": 0.0,
            "mps_base_mem": 0.0,
        }
    )
    engine = SimulationEngine(estimator=est)

    monkeypatch.setattr(config.DEFAULT, "mps_target_fidelity", 1.0)
    plan = engine.planner.plan(circuit, max_memory=26000, use_cache=False)
    assert Backend.MPS not in {s.backend for s in plan.steps}

    monkeypatch.setattr(config.DEFAULT, "mps_target_fidelity", 0.9)
    plan = engine.planner.plan(circuit, max_memory=26000, use_cache=False)
    assert plan.final_backend == Backend.MPS


def test_memory_threshold_limits_mps(monkeypatch):
    circuit = qft_circuit(5)
    circuit.symmetry = 0.0
    circuit.sparsity = 0.0
    est = CostEstimator(
        coeff={
            "sv_gate_1q": 50.0,
            "sv_gate_2q": 50.0,
            "mps_base_time": 0.0,
            "mps_base_mem": 0.0,
        }
    )
    engine = SimulationEngine(estimator=est)

    monkeypatch.setattr(config.DEFAULT, "mps_target_fidelity", 0.9)
    plan = engine.planner.plan(circuit, max_memory=1000, use_cache=False)
    assert plan.final_backend == Backend.MPS

    plan = engine.planner.plan(circuit, max_memory=4, use_cache=False)
    assert Backend.MPS not in {s.backend for s in plan.steps}


def test_rotation_diversity_discourages_dd(monkeypatch):
    circuit = qft_circuit(5)
    monkeypatch.setattr(config.DEFAULT, "dd_sparsity_threshold", 0.0)
    monkeypatch.setattr(config.DEFAULT, "dd_nnz_threshold", 10_000_000)
    monkeypatch.setattr(config.DEFAULT, "dd_phase_rotation_diversity_threshold", 3)
    monkeypatch.setattr(config.DEFAULT, "dd_amplitude_rotation_diversity_threshold", 3)
    engine = SimulationEngine(planner=Planner(perf_prio="time"))
    plan = engine.planner.plan(circuit)
    assert plan.final_backend == Backend.STATEVECTOR
    assert Backend.DECISION_DIAGRAM not in {s.backend for s in plan.steps}


def test_planner_selects_tableau_for_classically_simplified_qft():
    unsimplified = qft_circuit(3)
    circuit = Circuit(unsimplified.gates, use_classical_simplification=True)
    assert [g.gate for g in circuit.gates] == ["H", "H", "H"]
    engine = SimulationEngine()
    plan = engine.planner.plan(circuit)
    assert plan.final_backend == Backend.TABLEAU
    assert {s.backend for s in plan.steps} == {Backend.TABLEAU}

