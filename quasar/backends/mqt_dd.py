from __future__ import annotations

"""Wrapper for the MQT decision diagram simulators."""

from dataclasses import dataclass, field
from typing import Dict, Sequence

from mqt.core.ir import QuantumComputation
import mqt.ddsim as ddsim

from ..ssd import SSD, SSDPartition
from ..cost import Backend as BackendType
from .base import Backend


@dataclass
class DecisionDiagramBackend(Backend):
    backend: BackendType = BackendType.DECISION_DIAGRAM
    circuit: QuantumComputation | None = field(default=None, init=False)
    num_qubits: int = field(default=0, init=False)
    history: list[str] = field(default_factory=list, init=False)
    state: object | None = field(default=None, init=False)

    _ALIASES: Dict[str, str] = field(default_factory=lambda: {"SDG": "sdg", "U1": "p"})

    def load(self, num_qubits: int, **_: dict) -> None:
        self.circuit = QuantumComputation(num_qubits)
        self.num_qubits = num_qubits
        self.history.clear()
        self.state = None

    def ingest(self, state: object) -> None:
        """Initialise the backend from an existing decision diagram state."""
        if isinstance(state, tuple) and len(state) == 2:
            n, ptr = state
            self.num_qubits = int(n)
            self.state = {"ptr": ptr, "num_qubits": self.num_qubits}
            self.circuit = QuantumComputation(self.num_qubits)
            self.history.clear()
            return

        n = getattr(state, "num_qubits", None)
        if n is None:
            raise TypeError("Unsupported state for decision diagram backend")
        self.num_qubits = int(n)
        self.state = state
        self.circuit = QuantumComputation(self.num_qubits)
        self.history.clear()

    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        if self.circuit is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        lname = self._ALIASES.get(name.upper(), name.lower())
        func = getattr(self.circuit, lname, None)
        if func is None:
            raise ValueError(f"Unsupported MQT DD gate {name}")
        args = [float(v) for v in params.values()] if params else []
        func(*args, *qubits)
        self.history.append(name.upper())

    def extract_ssd(self) -> SSD:
        if self.circuit is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        simulator = ddsim.CircuitSimulator(self.circuit)
        self.state = simulator.get_constructed_dd()
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
        )
        return SSD([part])

    # ------------------------------------------------------------------
    def statevector(self) -> Sequence[complex]:
        """Return a dense statevector for the DD backend.

        The decision diagram package does not currently expose an efficient
        statevector extraction method, hence this is left unimplemented.
        """
        raise NotImplementedError("Statevector extraction not supported")
