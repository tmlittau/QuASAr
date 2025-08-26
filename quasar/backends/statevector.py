from __future__ import annotations

"""Dense statevector simulation using NumPy."""

from dataclasses import dataclass, field
from typing import Dict, Sequence
import numpy as np

from ..ssd import SSD, SSDPartition
from ..cost import Backend as BackendType
from .base import Backend


@dataclass
class StatevectorBackend(Backend):
    """Simple statevector simulator.

    The implementation is intentionally minimal and optimised for clarity
    rather than performance.  It supports a subset of common gates used in
    the QuASAr tests.
    """

    backend: BackendType = BackendType.STATEVECTOR
    state: np.ndarray | None = field(default=None, init=False)
    num_qubits: int = field(default=0, init=False)
    history: list[str] = field(default_factory=list, init=False)

    _GATES: Dict[str, np.ndarray] = field(default_factory=lambda: {
        "I": np.eye(2, dtype=complex),
        "ID": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        "H": 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex),
        "S": np.array([[1, 0], [0, 1j]], dtype=complex),
        "SDG": np.array([[1, 0], [0, -1j]], dtype=complex),
        "CX": np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        ),
        "CZ": np.diag([1, 1, 1, -1]).astype(complex),
        "SWAP": np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=complex,
        ),
    }, init=False)

    def load(self, num_qubits: int, **_: dict) -> None:
        self.num_qubits = num_qubits
        self.state = np.zeros(2 ** num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.history.clear()

    # ------------------------------------------------------------------
    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        if self.state is None:
            raise RuntimeError("Backend not initialised; call 'load' first")

        gate = self._GATES.get(name.upper())
        if gate is None:
            raise ValueError(f"Unsupported gate {name}")

        self.history.append(name.upper())
        k = len(qubits)
        order = list(qubits) + [i for i in range(self.num_qubits) if i not in qubits]
        state = self.state.reshape([2] * self.num_qubits).transpose(order)
        state = state.reshape(2 ** k, -1)
        state = gate @ state
        state = state.reshape([2] * self.num_qubits).transpose(np.argsort(order))
        self.state = state.reshape(-1)

    # ------------------------------------------------------------------
    def extract_ssd(self) -> SSD:
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
        )
        return SSD([part])
