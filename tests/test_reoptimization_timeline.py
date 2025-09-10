import time

from quasar.planner import Planner
from benchmarks.circuits import ghz_circuit, qft_circuit


def test_reoptimization_timeline() -> None:
    """Planner should react to circuit perturbations and recover quickly."""
    planner = Planner()

    baseline = ghz_circuit(4)
    start = time.perf_counter()
    base_plan = planner.plan(baseline)
    baseline_time = time.perf_counter() - start
    assert baseline_time < 1.0

    perturbed = qft_circuit(4)
    start = time.perf_counter()
    pert_plan = planner.plan(perturbed)
    perturb_time = time.perf_counter() - start
    assert perturb_time < 1.0

    # Planner should detect change in circuit and adjust backend.
    assert pert_plan.final_backend != base_plan.final_backend

    # Recovery: planning the original circuit again should match initial backend.
    start = time.perf_counter()
    recovered = planner.plan(baseline)
    recovery_time = time.perf_counter() - start
    assert recovery_time < 1.0
    assert recovered.final_backend == base_plan.final_backend
