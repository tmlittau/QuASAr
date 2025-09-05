from benchmarks.circuits import qft_circuit, random_circuit, w_state_circuit
from quasar import Backend, Scheduler
from quasar.planner import Planner
from quasar.partitioner import Partitioner
from quasar.config import DEFAULT


def test_w_state_selects_decision_diagram_via_sparsity():
    circ = w_state_circuit(5)
    assert circ.sparsity >= DEFAULT.dd_sparsity_threshold
    assert circ.symmetry < DEFAULT.dd_symmetry_threshold
    scheduler = Scheduler()
    scheduler.prepare_run(circ)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.DECISION_DIAGRAM


def test_qft_selects_decision_diagram_via_symmetry():
    circ = qft_circuit(5)
    assert circ.symmetry >= DEFAULT.dd_symmetry_threshold
    assert circ.sparsity < DEFAULT.dd_sparsity_threshold
    scheduler = Scheduler()
    scheduler.prepare_run(circ)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.DECISION_DIAGRAM


def test_random_circuit_stays_statevector_when_metrics_low():
    circ = random_circuit(5, seed=123)
    assert circ.sparsity < DEFAULT.dd_sparsity_threshold
    assert circ.symmetry < DEFAULT.dd_symmetry_threshold
    scheduler = Scheduler()
    scheduler.prepare_run(circ)
    part = circ.ssd.partitions[0]
    assert part.backend == Backend.STATEVECTOR


def test_w_state_planner_prefers_decision_diagram():
    circ = w_state_circuit(5)
    planner = Planner()
    plan = planner.plan(circ)
    assert plan.final_backend == Backend.DECISION_DIAGRAM


def test_w_state_partitioner_prefers_decision_diagram():
    circ = w_state_circuit(5)
    part = Partitioner().partition(circ)
    assert part.partitions[0].backend == Backend.DECISION_DIAGRAM
