import pytest

from quasar.circuit import Circuit
from quasar.cost import Backend
from quasar import config
from quasar.scheduler import Scheduler


class ForceStatevectorScheduler(Scheduler):
    """Scheduler that always selects the statevector quick path."""

    def select_backend(
        self,
        circuit: Circuit,
        *,
        backend: Backend | None = None,
        max_time: float | None = None,
        optimization_level: int | None = None,
    ) -> Backend | None:
        return Backend.STATEVECTOR


def _local_chain_circuit() -> Circuit:
    return Circuit(
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "CX", "qubits": [1, 2]},
            {"gate": "RZ", "qubits": [2], "params": {"theta": 0.5}},
            {"gate": "CX", "qubits": [2, 3]},
            {"gate": "CX", "qubits": [3, 4]},
        ],
        use_classical_simplification=False,
    )


def test_prepare_run_respects_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quick-path execution must fall back when exceeding the memory budget."""

    # Disable the decision diagram metric so that MPS becomes the preferred
    # alternative backend once planning is triggered.
    monkeypatch.setattr(config.DEFAULT, "dd_metric_threshold", 2.0)

    circuit = _local_chain_circuit()
    scheduler = ForceStatevectorScheduler(
        backend_order=[
            Backend.STATEVECTOR,
            Backend.MPS,
            Backend.DECISION_DIAGRAM,
            Backend.TABLEAU,
        ],
        quick_max_qubits=10,
        quick_max_gates=32,
        quick_max_depth=32,
    )

    plan = scheduler.prepare_run(circuit, max_memory=60_000)

    assert plan.final_backend != Backend.STATEVECTOR
    assert all(step.backend != Backend.STATEVECTOR for step in plan.steps)
    assert plan.step_costs
    assert max(cost.memory for cost in plan.step_costs) <= 60_000
