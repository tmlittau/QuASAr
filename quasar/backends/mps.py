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

    def ingest(self, state: Sequence[complex] | list[np.ndarray]) -> None:
        """Initialise the MPS from a statevector or list of tensors."""
        if isinstance(state, list):
            self.tensors = [np.array(t, dtype=complex) for t in state]
            self.num_qubits = len(self.tensors)
            self.history.clear()
            return

        vec = np.asarray(state, dtype=complex)
        dim = len(vec)
        n = int(np.log2(dim))
        if 2 ** n != dim:
            raise TypeError("Statevector length is not a power of two")

        self.num_qubits = n
        self.tensors = []
        psi = vec.reshape(1, dim)
        chi_left = 1
        for _ in range(n - 1):
            psi = psi.reshape(chi_left * 2, -1)
            u, s, vh = np.linalg.svd(psi, full_matrices=False)
            chi = min(self.chi, len(s))
            u = u[:, :chi]
            s = s[:chi]
            vh = vh[:chi, :]
            self.tensors.append(u.reshape(chi_left, 2, chi))
            psi = np.diag(s) @ vh
            chi_left = chi
        self.tensors.append(psi.reshape(chi_left, 2, 1))
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
        raise ValueError(f"Unsupported gate {name}")

    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        lname = name.upper()
        gate = self._gate_matrix(lname, params)
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

    # ------------------------------------------------------------------
    def statevector(self) -> np.ndarray:
        """Return a dense statevector corresponding to the MPS."""
        if not self.tensors:
            raise RuntimeError("Backend not initialised; call 'load' first")
        psi = self.tensors[0]
        for tensor in self.tensors[1:]:
            psi = np.tensordot(psi, tensor, axes=(2, 0))
        return psi.reshape(-1).copy()
