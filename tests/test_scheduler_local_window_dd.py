from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pytest

from quasar.circuit import Circuit, Gate
from quasar.cost import Backend, Cost
from quasar.planner import PlanResult, PlanStep
from quasar.scheduler import Scheduler
from quasar.ssd import ConversionLayer, SSD, SSDPartition
from quasar.backends.mqt_dd import DecisionDiagramBackend
from quasar_convert import ConversionEngine


@dataclass
class DummyStatevectorBackend:
    """Minimal statevector backend used for scheduler integration tests."""

    backend: Backend = Backend.STATEVECTOR
    num_qubits: int = 0
    history: list[str] = field(default_factory=list)
    state: np.ndarray | None = None

    def load(self, num_qubits: int, **_: dict) -> None:
        self.num_qubits = num_qubits
        self.state = np.zeros(1 << num_qubits, dtype=complex)
        if num_qubits:
            self.state[0] = 1.0 + 0j
        self.history.clear()

    def ingest(
        self,
        state: Sequence[complex],
        *,
        num_qubits: int | None = None,
        mapping: Sequence[int] | None = None,
    ) -> None:
        data = np.asarray(state, dtype=complex)
        k = int(np.log2(len(data))) if len(data) else 0
        if mapping is None:
            mapping = list(range(k))
        if num_qubits is None:
            num_qubits = max(mapping, default=-1) + 1 if mapping else k
        full = np.zeros(1 << num_qubits, dtype=complex)
        for local in range(len(data)):
            idx = 0
            for bit, qubit in enumerate(mapping):
                if (local >> bit) & 1:
                    idx |= 1 << qubit
            full[idx] = data[local]
        self.num_qubits = num_qubits
        self.state = full

    def apply_gate(self, name: str, qubits: Sequence[int], params: dict | None = None) -> None:
        raise AssertionError(f"Unexpected gate application: {name} on {qubits}")

    def extract_ssd(self) -> SSD:
        state = None if self.state is None else np.array(self.state, copy=True)
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
            state=state,
        )
        return SSD([part])

    def statevector(self) -> np.ndarray:
        if self.state is None:
            return np.array([], dtype=complex)
        return np.array(self.state, copy=True)


def test_local_window_conversion_avoids_dense_export(monkeypatch: pytest.MonkeyPatch) -> None:
    mqt_core = pytest.importorskip("mqt.core")
    dd = mqt_core.dd

    engine = ConversionEngine()
    if not hasattr(engine, "extract_local_window_dd"):
        pytest.skip("Decision diagram helper not available")

    scheduler = Scheduler(
        conversion_engine=engine,
        backends={
            Backend.DECISION_DIAGRAM: DecisionDiagramBackend(),
            Backend.STATEVECTOR: DummyStatevectorBackend(),
        },
    )

    circuit = Circuit([Gate("H", [0])], use_classical_simplification=False)

    plan = PlanResult(
        table=[],
        final_backend=Backend.STATEVECTOR,
        gates=circuit.gates,
        explicit_steps=[
            PlanStep(start=0, end=1, backend=Backend.DECISION_DIAGRAM),
            PlanStep(start=1, end=1, backend=Backend.STATEVECTOR),
        ],
        explicit_conversions=[
            ConversionLayer(
                boundary=(0,),
                source=Backend.DECISION_DIAGRAM,
                target=Backend.STATEVECTOR,
                rank=2,
                frontier=1,
                primitive="LW",
                cost=Cost(time=0.0, memory=0.0),
            )
        ],
        step_costs=[Cost(time=0.0, memory=0.0), Cost(time=0.0, memory=0.0)],
    )

    def fail_get_vector(self: dd.VectorDD) -> None:
        raise AssertionError("Dense decision diagram export was invoked")

    monkeypatch.setattr(dd.VectorDD, "get_vector", fail_get_vector)

    def fail_extract_local_window(*args, **kwargs):
        raise AssertionError("Dense helper used")

    monkeypatch.setattr(engine, "extract_local_window", fail_extract_local_window)

    result = scheduler.run(circuit, plan=plan)
    assert isinstance(result, SSD)

    state = result.extract_state(0)
    assert isinstance(state, np.ndarray)
    np.testing.assert_allclose(
        state,
        np.array([1 / np.sqrt(2.0), 1 / np.sqrt(2.0)], dtype=complex),
        atol=1e-12,
    )
