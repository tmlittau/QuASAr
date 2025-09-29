"""Tests for MPS initialisation in the scheduler."""

from quasar.circuit import Circuit, Gate
from quasar.cost import Backend, Cost, CostEstimator
from quasar.planner import PlanResult, PlanStep, Planner
from quasar.scheduler import Scheduler


class RecordingMPSBackend:
    """Minimal MPS backend capturing the ``chi`` argument passed to ``load``."""

    backend = Backend.MPS
    instances: list["RecordingMPSBackend"] = []

    def __init__(self) -> None:
        self.recorded_chi: int | None = None
        self.num_qubits: int | None = None
        RecordingMPSBackend.instances.append(self)

    def load(self, num_qubits: int, **kwargs) -> None:  # pragma: no cover - trivial
        self.num_qubits = num_qubits
        self.recorded_chi = kwargs.get("chi")

    def apply_gate(self, gate: str, qubits, params) -> None:  # pragma: no cover - stub
        del gate, qubits, params

    def extract_ssd(self):  # pragma: no cover - stub
        return None


def test_scheduler_initialises_mps_with_planned_chi() -> None:
    """Scheduler should forward the planner's ``chi_max`` to MPS backends."""

    chi_cap = 23
    estimator = CostEstimator(chi_max=chi_cap)
    planner = Planner(estimator=estimator)

    RecordingMPSBackend.instances.clear()
    scheduler = Scheduler(
        planner=planner,
        backends={Backend.MPS: RecordingMPSBackend()},
    )

    gate = Gate("H", [0])
    circuit = Circuit([gate], use_classical_simplification=False)
    gates = circuit.gates
    plan = PlanResult(
        table=[],
        final_backend=Backend.MPS,
        gates=gates,
        explicit_steps=[PlanStep(start=0, end=1, backend=Backend.MPS)],
        explicit_conversions=[],
        step_costs=[Cost(time=0.0, memory=0.0)],
    )

    scheduler.run(circuit, plan=plan)

    assert RecordingMPSBackend.instances, "Scheduler did not instantiate the backend"
    recorded = RecordingMPSBackend.instances[-1].recorded_chi
    assert recorded == planner.estimator.chi_max
