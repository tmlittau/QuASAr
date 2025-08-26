from __future__ import annotations

"""Matrix product state (MPS) simulator."""

from dataclasses import dataclass, field
from typing import Dict, Sequence
import numpy as np

from ..ssd import SSD, SSDPartition
from ..cost import Backend as BackendType
from .base import Backend


@dataclass
class MPSBackend(Backend):
    """Lightweight MPS simulator for local circuits.

    The implementation supports single-qubit gates and two-qubit gates on
    neighbouring qubits.  Bond dimensions are truncated to ``chi`` during
    two-qubit updates.
    """

    backend: BackendType = BackendType.MPS
    tensors: list[np.ndarray] = field(default_factory=list, init=False)
    num_qubits: int = field(default=0, init=False)
    chi: int = field(default=16, init=False)
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

    def load(self, num_qubits: int, **kwargs: dict) -> None:
        self.num_qubits = num_qubits
        self.chi = int(kwargs.get("chi", 16))
        self.tensors = [np.zeros((1, 2, 1), dtype=complex) for _ in range(num_qubits)]
        for tensor in self.tensors:
            tensor[0, 0, 0] = 1.0
        self.history.clear()

    # ------------------------------------------------------------------
    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        lname = name.upper()
        if lname == "BRIDGE":
            # Bridge tensors are identities for this reference backend.
            self.history.append(lname)
            return

        gate = self._GATES.get(lname)
        if gate is None:
            raise ValueError(f"Unsupported gate {name}")
        self.history.append(lname)

        if len(qubits) == 1:
            i = qubits[0]
            A = self.tensors[i]
            self.tensors[i] = np.tensordot(gate, A, axes=(1, 1)).transpose(1, 0, 2)
            return

        if len(qubits) == 2:
            q0, q1 = qubits
            if abs(q0 - q1) != 1:
                raise NotImplementedError("MPS backend supports only nearest-neighbour gates")
            i = min(q0, q1)
            j = i + 1
            left = self.tensors[i]
            right = self.tensors[j]
            theta = np.tensordot(left, right, axes=(2, 0))  # l,2,2,r
            theta = np.tensordot(gate.reshape(2, 2, 2, 2), theta, axes=([2, 3], [1, 2]))
            theta = theta.transpose(2, 0, 1, 3)
            l, _, _, r = theta.shape
            theta = theta.reshape(l * 2, 2 * r)
            u, s, vh = np.linalg.svd(theta, full_matrices=False)
            chi = min(self.chi, len(s))
            u = u[:, :chi]
            s = s[:chi]
            vh = vh[:chi, :]
            self.tensors[i] = u.reshape(l, 2, chi)
            self.tensors[j] = (np.diag(s) @ vh).reshape(chi, 2, r)
            return

        raise NotImplementedError("Gate arity beyond 2 is not supported")

    # ------------------------------------------------------------------
    def extract_ssd(self) -> SSD:
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
        )
        return SSD([part])
