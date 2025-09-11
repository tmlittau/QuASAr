import pytest

from quasar.planner import Planner
from quasar.circuit import Circuit, Gate
from quasar.cost import Backend
from benchmarks.circuits import ghz_circuit, qft_circuit


def compute_backends() -> dict[str, tuple[Backend, Backend, Backend]]:
    """Return perturbed and recovered backends for minor and major changes."""
    planner = Planner()
    baseline = ghz_circuit(5)
    baseline_backend = planner.plan(baseline).final_backend

    minor = Circuit(
        list(baseline.gates) + [Gate("H", [0])], use_classical_simplification=False
    )
    major = qft_circuit(5)
    perturbations = {"minor": minor, "major": major}

    metrics: dict[str, tuple[Backend, Backend, Backend]] = {}
    for name, pert in perturbations.items():
        pert_backend = planner.plan(pert).final_backend
        recovered_backend = planner.plan(baseline).final_backend
        metrics[name] = (pert_backend, recovered_backend, baseline_backend)
    return metrics


@pytest.mark.parametrize("name", ["minor", "major"])
def test_recovery_backends(name: str) -> None:
    pert_backend, recovered_backend, baseline_backend = compute_backends()[name]
    assert recovered_backend == baseline_backend
    if name == "major":
        assert pert_backend != baseline_backend
