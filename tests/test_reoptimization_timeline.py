
from quasar.planner import Planner
from benchmarks.circuits import ghz_circuit, qft_circuit


def test_reoptimization_timeline() -> None:
    """Planner should react to circuit perturbations and recover."""
    planner = Planner()

    baseline = ghz_circuit(4)
    base_plan = planner.plan(baseline)

    perturbed = qft_circuit(4)
    pert_plan = planner.plan(perturbed)

    # Planner should detect change in circuit and adjust backend.
    assert pert_plan.final_backend != base_plan.final_backend

    # Recovery: planning the original circuit again should match initial backend.
    recovered = planner.plan(baseline)
    assert recovered.final_backend == base_plan.final_backend
