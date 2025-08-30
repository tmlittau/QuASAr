from __future__ import annotations

"""Statevector backend powered by Qiskit Aer."""

from dataclasses import dataclass, field
from typing import Dict, Sequence, List, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import U2Gate, UGate
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from ..ssd import SSD, SSDPartition
from ..cost import Backend as BackendType
from .base import Backend


@dataclass
class StatevectorBackend(Backend):
    """Backend that builds a ``QuantumCircuit`` and evaluates via Aer.

    Parameters
    ----------
    method:
        Aer simulation method to use.  The default is ``"statevector"``.  A
        :class:`ValueError` is raised if an unsupported method is requested.
    """

    backend: BackendType = BackendType.STATEVECTOR
    method: str = "statevector"
    circuit: QuantumCircuit | None = field(default=None, init=False)
    num_qubits: int = field(default=0, init=False)
    history: list[str] = field(default_factory=list, init=False)
    _benchmark_mode: bool = field(default=False, init=False)
    _benchmark_ops: List[Tuple[str, Sequence[int], Dict[str, float] | None]] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        available = AerSimulator().available_methods()
        if self.method not in available:
            raise ValueError(
                f"Unsupported Aer method '{self.method}'. Available: {available}"
            )

    # ------------------------------------------------------------------
    def load(self, num_qubits: int, **_: dict) -> None:
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)
        self.history.clear()

    def ingest(self, state: Sequence[complex] | Statevector) -> None:
        """Initialise from an external statevector."""
        if isinstance(state, Statevector):
            data = state.data
            n = int(np.log2(len(data)))
        else:
            data = np.asarray(state, dtype=complex)
            n = int(np.log2(len(data)))
        if 2**n != len(data):
            raise TypeError("Statevector length is not a power of two")
        self.num_qubits = n
        self.circuit = QuantumCircuit(n)
        # convert from little-endian to Qiskit's big-endian ordering
        be = data.reshape([2] * n).transpose(*reversed(range(n))).reshape(-1)
        self.circuit.initialize(be, range(n))
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

    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        if self._benchmark_mode:
            self._benchmark_ops.append((name, tuple(qubits), params))
            return
        if self.circuit is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        lname = name.upper()
        self.history.append(lname)
        if lname == "RX":
            self.circuit.rx(self._param(params, 0), qubits[0])
        elif lname == "RY":
            self.circuit.ry(self._param(params, 0), qubits[0])
        elif lname == "RZ":
            self.circuit.rz(self._param(params, 0), qubits[0])
        elif lname in {"P", "U1"}:
            self.circuit.p(self._param(params, 0), qubits[0])
        elif lname == "RZZ":
            self.circuit.rzz(self._param(params, 0), qubits[0], qubits[1])
        elif lname == "U2":
            gate = U2Gate(self._param(params, 0), self._param(params, 1))
            self.circuit.append(gate, [qubits[0]])
        elif lname in {"U", "U3"}:
            gate = UGate(
                self._param(params, 0),
                self._param(params, 1),
                self._param(params, 2),
            )
            self.circuit.append(gate, [qubits[0]])
        else:
            method = getattr(self.circuit, lname.lower(), None)
            if method is None:
                raise NotImplementedError(f"Unsupported gate {name}")
            method(*qubits)

    # ------------------------------------------------------------------
    def _run(self) -> np.ndarray:
        if self.circuit is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        sim = AerSimulator(method=self.method)
        circuit = self.circuit.copy()
        circuit.save_statevector()
        result = sim.run(circuit).result()
        vec = result.get_statevector()
        n = self.num_qubits
        # convert from Qiskit's big-endian to little-endian
        return np.asarray(vec).reshape([2] * n).transpose(*reversed(range(n))).reshape(-1)

    def extract_ssd(self) -> SSD:
        state = self._run()
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
            state=state,
        )
        return SSD([part])

    # ------------------------------------------------------------------
    def statevector(self) -> np.ndarray:
        """Return a dense statevector of the current simulator state."""
        return self._run()


class AerStatevectorBackend(StatevectorBackend):
    """Convenience backend using Aer with the ``statevector`` method."""

    def __init__(self, **kwargs):
        kwargs.setdefault("method", "statevector")
        super().__init__(**kwargs)
