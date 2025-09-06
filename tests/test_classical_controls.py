from __future__ import annotations

"""Tests for classical control simplifications in Scheduler."""

from quasar import Circuit, Scheduler
from quasar.cost import Backend
from quasar.ssd import SSD, SSDPartition


class RecordingBackend:
    """Backend prototype that records gate applications at the class level."""

    backend = Backend.STATEVECTOR
    history: list[str] = []

    def __init__(self) -> None:
        self.num_qubits = 0

    def load(self, n: int, **_: dict) -> None:
        self.num_qubits = n

    def ingest(self, state, **_: dict) -> None:  # pragma: no cover - trivial
        self.state = state

    def apply_gate(self, gate: str, qubits, params) -> None:  # pragma: no cover - trivial
        type(self).history.append(gate)

    def extract_ssd(self) -> SSD:  # pragma: no cover - trivial
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(type(self).history),
            backend=self.backend,
        )
        return SSD([part])


def run_circuit(gates: list[dict]) -> tuple[list[str], Circuit]:
    RecordingBackend.history = []
    circ = Circuit(gates)
    sched = Scheduler(backends={Backend.STATEVECTOR: RecordingBackend()})
    plan = sched.prepare_run(circ, backend=Backend.STATEVECTOR)
    sched.run(circ, plan)
    return RecordingBackend.history, circ


def test_bit_flip_chain_remains_classical() -> None:
    history, circ = run_circuit(
        [
            {"gate": "X", "qubits": [0]},
            {"gate": "X", "qubits": [0]},
            {"gate": "X", "qubits": [0]},
        ]
    )
    assert history == []
    assert circ.classical_state == [1]


def test_quantum_control_preserves_gate() -> None:
    history, circ = run_circuit(
        [
            {"gate": "H", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ]
    )
    assert history == ["H", "CX"]
    assert circ.classical_state == [None, None]


def test_classical_control_one_reduces_gate() -> None:
    history, circ = run_circuit(
        [
            {"gate": "X", "qubits": [0]},
            {"gate": "H", "qubits": [1]},
            {"gate": "CX", "qubits": [0, 1]},
        ]
    )
    assert history == ["H", "X"]
    assert circ.classical_state == [1, None]


def test_classical_control_zero_skips_gate() -> None:
    history, circ = run_circuit([
        {"gate": "CX", "qubits": [0, 1]},
    ])
    assert history == []
    assert circ.classical_state == [0, 0]
