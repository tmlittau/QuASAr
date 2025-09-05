from __future__ import annotations

from quasar.circuit import Circuit
from quasar.planner import Planner
from quasar.cost import CostEstimator
from quasar.scheduler import Scheduler
from quasar.cost import Backend
from quasar.ssd import SSD, SSDPartition


class DummyBackend:
    """Minimal backend used for testing conversions."""

    backend: Backend

    def __init__(self) -> None:
        self.num_qubits = 0
        self.history: list[str] = []

    def load(self, n: int, **_: dict) -> None:  # pragma: no cover - trivial
        self.num_qubits = n

    def apply_gate(self, gate: str, qubits, params) -> None:  # pragma: no cover - trivial
        self.history.append(gate)

    def ingest(self, state) -> None:  # pragma: no cover - used to force conversion
        raise RuntimeError("force conversion")

    def extract_ssd(self) -> SSD:  # pragma: no cover - trivial
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
        )
        return SSD([part])

    def statevector(self):  # pragma: no cover - not used
        return [0] * (2 ** self.num_qubits)


class DummyTableauBackend(DummyBackend):
    backend = Backend.TABLEAU


class DummyStatevectorBackend(DummyBackend):
    backend = Backend.STATEVECTOR


class RecordingEngine:
    """Conversion engine that records boundaries and forbids planning-time conversion."""

    def __init__(self) -> None:
        self.boundaries: list[tuple[int, ...]] = []

    def convert(self, *args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("convert should not be called")

    def convert_boundary_to_statevector(self, ssd):  # pragma: no cover - trivial
        self.boundaries.append(tuple(ssd.boundary_qubits))
        return object()

    def convert_boundary_to_tableau(self, ssd):  # pragma: no cover - unused
        self.boundaries.append(tuple(ssd.boundary_qubits))
        return object()

    def convert_boundary_to_dd(self, ssd):  # pragma: no cover - unused
        self.boundaries.append(tuple(ssd.boundary_qubits))
        return object()

    def extract_local_window(self, state, boundary):  # pragma: no cover - unused
        self.boundaries.append(tuple(boundary))
        return object()

    def build_bridge_tensor(self, *args, **kwargs):  # pragma: no cover - unused
        return object()


def test_planner_conversions_used():
    """Planner-provided conversions are honoured by the scheduler."""

    circ = Circuit(
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "T", "qubits": [0]},
        ]
    )
    planner = Planner(
        estimator=CostEstimator(
            coeff={"sv_gate_1q": 1e6, "sv_gate_2q": 1e6, "sv_meas": 1e6}
        ),
        conversion_cost_multiplier=0.0,
    )
    planner.plan(circ)
    assert len(circ.ssd.conversions) == 0

    engine = RecordingEngine()
    sched = Scheduler(
        planner=planner,
        conversion_engine=engine,
        backends={
            Backend.TABLEAU: DummyTableauBackend(),
            Backend.STATEVECTOR: DummyStatevectorBackend(),
            Backend.MPS: DummyStatevectorBackend(),
            Backend.DECISION_DIAGRAM: DummyStatevectorBackend(),
        },
    )

    plan = sched.prepare_run(circ)
    result = sched.run(circ, plan)

    assert result.conversions == []
