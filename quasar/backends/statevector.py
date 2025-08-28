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
        "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
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

    def ingest(self, state: np.ndarray | Sequence[complex]) -> None:
        """Initialise the backend from an external statevector."""
        array = np.asarray(state, dtype=complex)
        dim = len(array)
        n = int(np.log2(dim))
        if 2 ** n != dim:
            raise TypeError("Statevector length is not a power of two")
        self.num_qubits = n
        self.state = array.reshape(-1).astype(complex)
        self.history.clear()

    # ------------------------------------------------------------------
    def _param(self, params: Dict[str, float] | None, idx: int) -> float:
        if not params:
            return 0.0
        key = f"param{idx}"
        if key in params:
            return float(params[key])
        values = list(params.values())
        return float(values[idx]) if idx < len(values) else 0.0

    def _gate_matrix(self, name: str, params: Dict[str, float] | None) -> np.ndarray:
        gate = self._GATES.get(name)
        if gate is not None:
            return gate

        p0 = self._param(params, 0)
        if name == "RX":
            c = np.cos(p0 / 2)
            s = np.sin(p0 / 2)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        if name == "RY":
            c = np.cos(p0 / 2)
            s = np.sin(p0 / 2)
            return np.array([[c, -s], [s, c]], dtype=complex)
        if name == "RZ":
            return np.diag([np.exp(-1j * p0 / 2), np.exp(1j * p0 / 2)]).astype(complex)
        if name in {"P", "U1"}:
            return np.diag([1.0, np.exp(1j * p0)]).astype(complex)
        if name == "RZZ":
            return np.diag(
                [
                    np.exp(-1j * p0 / 2),
                    np.exp(1j * p0 / 2),
                    np.exp(1j * p0 / 2),
                    np.exp(-1j * p0 / 2),
                ]
            ).astype(complex)
        if name == "U2":
            p1 = self._param(params, 1)
            return (1 / np.sqrt(2)) * np.array(
                [
                    [1.0, -np.exp(1j * p1)],
                    [np.exp(1j * p0), np.exp(1j * (p0 + p1))],
                ],
                dtype=complex,
            )
        if name in {"U", "U3"}:
            p1 = self._param(params, 1)
            p2 = self._param(params, 2)
            c = np.cos(p0 / 2)
            s = np.sin(p0 / 2)
            return np.array(
                [
                    [c, -np.exp(1j * p2) * s],
                    [np.exp(1j * p1) * s, np.exp(1j * (p1 + p2)) * c],
                ],
                dtype=complex,
            )
        raise NotImplementedError(f"Unsupported gate {name}")

    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        if self.state is None:
            raise RuntimeError("Backend not initialised; call 'load' first")

        lname = name.upper()
        gate = self._gate_matrix(lname, params)

        self.history.append(lname)
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
            state=self.statevector(),
        )
        return SSD([part])

    # ------------------------------------------------------------------
    def statevector(self) -> np.ndarray:
        """Return a dense statevector of the current simulator state."""
        if self.state is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        return self.state.copy()
