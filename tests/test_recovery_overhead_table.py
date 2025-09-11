import time

import pytest

from quasar.planner import Planner
from quasar.circuit import Circuit, Gate
from benchmarks.circuits import ghz_circuit, qft_circuit

BASELINES = {
    "minor": (2.0e-05, 1.4),
    "major": (2.6e-05, 9.0),
}


def compute_metrics() -> dict[str, tuple[float, float]]:
    planner = Planner()
    baseline = ghz_circuit(5)
    start = time.perf_counter()
    planner.plan(baseline)
    baseline_time = time.perf_counter() - start

    minor = Circuit(list(baseline.gates) + [Gate("H", [0])], use_classical_simplification=False)
    major = qft_circuit(5)
    perturbations = {"minor": minor, "major": major}

    metrics: dict[str, tuple[float, float]] = {}
    for name, pert in perturbations.items():
        start = time.perf_counter(); planner.plan(pert); perturb_time = time.perf_counter() - start
        start = time.perf_counter(); planner.plan(baseline); recovery_time = time.perf_counter() - start
        overhead = (perturb_time + recovery_time) / baseline_time
        metrics[name] = (recovery_time, overhead)
    return metrics


@pytest.mark.parametrize("name", ["minor", "major"])
def test_recovery_overhead_table(name: str) -> None:
    metrics = compute_metrics()[name]
    expected_recovery, expected_overhead = BASELINES[name]
    recovery, overhead = metrics
    assert recovery == pytest.approx(expected_recovery, rel=0.5, abs=1e-4)
    assert overhead == pytest.approx(expected_overhead, rel=0.5)
