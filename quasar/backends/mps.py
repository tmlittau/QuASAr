from __future__ import annotations

"""Matrix product state backend using Qiskit's MPS simulator."""

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, List, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import U2Gate, UGate, standard_gates
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.library import set_matrix_product_state

from ..ssd import SSD, SSDPartition
from ..cost import Backend as BackendType
from .base import Backend


@dataclass
class MPSBackend(Backend):
    """Backend wrapping the Aer simulator.

    Parameters
    ----------
    method:
        Aer simulation method to use.  Defaults to
        ``"matrix_product_state"``.  A :class:`ValueError` is raised if an
        unsupported method is requested.
    """

    backend: BackendType = BackendType.MPS
    method: str = "matrix_product_state"
    circuit: QuantumCircuit | None = field(default=None, init=False)
    num_qubits: int = field(default=0, init=False)
    chi: int = field(default=16, init=False)
    history: list[str] = field(default_factory=list, init=False)
    _benchmark_mode: bool = field(default=False, init=False)
    _benchmark_ops: List[Tuple[str, Sequence[int], Dict[str, float] | None]] = field(
        default_factory=list, init=False
    )
    _cached_state: object | None = field(default=None, init=False)
    _cached_statevector: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        available = AerSimulator().available_methods()
        if self.method not in available:
            raise ValueError(
                f"Unsupported Aer method '{self.method}'. Available: {available}"
            )

    # ------------------------------------------------------------------
    def load(self, num_qubits: int, **kwargs: dict) -> None:
        self.num_qubits = num_qubits
        self.chi = int(kwargs.get("chi", 16))
        self.circuit = QuantumCircuit(num_qubits)
        self.history.clear()
        self._cached_state = None
        self._cached_statevector = None

    def ingest(
        self,
        state: object,
        *,
        num_qubits: int | None = None,
        mapping: Sequence[int] | None = None,
    ) -> None:
        """Ingest an initial state in either dense or MPS form."""
        data: np.ndarray | None
        n: int
        if isinstance(state, Statevector):
            data = state.data
            n = int(np.log2(len(data)))
        else:
            try:
                data = np.asarray(state, dtype=complex)
            except (TypeError, ValueError):
                data = None
            if data is not None and data.ndim == 1:
                n = int(np.log2(len(data)))
                if 2**n != len(data):
                    raise TypeError("Statevector length is not a power of two")
            else:
                data = None
        if data is not None:
            if num_qubits is None:
                num_qubits = n
            if mapping is None:
                if n != num_qubits:
                    raise ValueError("num_qubits does not match state size")
                mapping = list(range(n))
            if len(mapping) != n:
                raise ValueError("Mapping length does not match state size")
            self.num_qubits = num_qubits
            self.circuit = QuantumCircuit(num_qubits)
            be = data.reshape([2] * n).transpose(*reversed(range(n))).reshape(-1)
            self.circuit.initialize(be, mapping)
            self.history.clear()
            self._cached_state = None
            self._cached_statevector = None
            return
        # assume native MPS representation
        if mapping is not None:
            raise NotImplementedError("Mapping is not supported for MPS ingestion")
        if num_qubits is None:
            num_qubits = len(state)  # type: ignore[arg-type]
        self.num_qubits = num_qubits
        self.circuit = QuantumCircuit(num_qubits)
        set_matrix_product_state(self.circuit, state)
        self.history.clear()
        self._cached_state = state
        self._cached_statevector = None

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
        if lname == "CCX":
            raise NotImplementedError("CCX gates must be decomposed before execution")
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
        elif lname == "CP":
            k = float(params.get("k", 0)) if params else 0.0
            theta = 2 * np.pi / (2 ** k)
            self.circuit.cp(theta, qubits[0], qubits[1])
        elif lname.startswith("C"):
            base = lname[1:]
            num_params = len(params) if params else 0
            pvals = [self._param(params, i) for i in range(num_params)]
            gate_cls = getattr(standard_gates, f"{base}Gate", None)
            if gate_cls is None:
                raise NotImplementedError(f"Unsupported gate {name}")
            gate = gate_cls(*pvals).control()
            self.circuit.unitary(gate.to_matrix(), qubits)
        else:
            method = getattr(self.circuit, lname.lower(), None)
            if method is None:
                raise NotImplementedError(f"Unsupported gate {name}")
            method(*qubits)
        self._cached_state = None
        self._cached_statevector = None

    # ------------------------------------------------------------------
    def prepare_benchmark(self, circuit: Any | None = None) -> None:
        """Prepare the backend for benchmarking.

        Gates applied after this call are immediately appended to the
        internal circuit and the simulation is executed exactly once during
        :meth:`run_benchmark`.
        """
        if circuit is not None:
            self.circuit = circuit.copy()
            self.num_qubits = circuit.num_qubits
        elif self.circuit is None:
            if not self.num_qubits:
                raise RuntimeError("Backend not initialised; call 'load' first")
            self.circuit = QuantumCircuit(self.num_qubits)
        self.history.clear()
        self._benchmark_mode = False
        self._benchmark_ops = []
        self._cached_state = None
        self._cached_statevector = None

    def run_benchmark(self, *, return_state: bool = False) -> object | None:
        """Execute the prepared circuit once and optionally return the state."""
        self._benchmark_mode = False
        state = self._run()
        self._cached_state = state
        self._cached_statevector = None
        return state if return_state else None

    # ------------------------------------------------------------------
    def _run(self) -> object:
        if self.circuit is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        sim = AerSimulator(method=self.method)
        if self.method == "matrix_product_state":
            sim.set_options(matrix_product_state_max_bond_dimension=self.chi)
        circuit = self.circuit.copy()
        circuit.save_matrix_product_state()
        result = sim.run(circuit).result()
        return result.data(0)["matrix_product_state"]

    def extract_ssd(self) -> SSD:
        state = self._cached_state if self._cached_state is not None else self._run()
        self._cached_state = state
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
            state=state,
        )
        return SSD([part])

    # ------------------------------------------------------------------
    def _mps_to_statevector(self, mps: object) -> np.ndarray:
        """Convert a matrix product state to a dense statevector."""
        sim = AerSimulator(method="matrix_product_state")
        circuit = QuantumCircuit(self.num_qubits)
        set_matrix_product_state(circuit, mps)
        circuit.save_statevector()
        result = sim.run(circuit).result()
        vec = result.get_statevector()
        n = self.num_qubits
        return (
            np.asarray(vec)
            .reshape([2] * n)
            .transpose(*reversed(range(n)))
            .reshape(-1)
        )

    def statevector(self) -> np.ndarray:
        """Return a dense statevector corresponding to the circuit."""
        if self._cached_statevector is not None:
            return self._cached_statevector
        if self._cached_state is None:
            self._cached_state = self._run()
        vec = self._mps_to_statevector(self._cached_state)
        self._cached_statevector = vec
        return vec


class AerMPSBackend(MPSBackend):
    """Convenience backend using Aer with the ``matrix_product_state`` method."""

    def __init__(self, **kwargs):
        kwargs.setdefault("method", "matrix_product_state")
        super().__init__(**kwargs)
