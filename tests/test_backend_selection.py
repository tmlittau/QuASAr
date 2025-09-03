from benchmarks.circuits import ghz_circuit
from quasar import SimulationEngine, Backend


def test_planner_selects_tableau_for_ghz():
    circuit = ghz_circuit(4)
    engine = SimulationEngine()
    plan = engine.planner.plan(circuit)
    assert plan.final_backend == Backend.TABLEAU
