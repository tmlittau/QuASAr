
from quasar.planner import Planner
from quasar.circuit import Circuit
from benchmarks.circuits import ghz_circuit, _qft_spec


def test_reoptimization_timeline() -> None:
    """Planner should react to circuit perturbations and recover."""
    planner = Planner()

    baseline = ghz_circuit(4)
    base_plan = planner.plan(baseline)

    perturbed = Circuit(_qft_spec(4), use_classical_simplification=False)
    pert_plan = planner.plan(perturbed)

    # Planner should detect change in circuit and adjust backend.
    assert pert_plan.final_backend != base_plan.final_backend

    # Recovery: planning the original circuit again should match initial backend.
    recovered = planner.plan(baseline)
    assert recovered.final_backend == base_plan.final_backend
