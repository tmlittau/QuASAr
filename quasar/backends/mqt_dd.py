"""Decision diagram backend based on :class:`mqt.core.dd.DDPackage`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import numpy as np

from mqt.core import dd
from mqt.core.ir import operations

from ..cost import Backend as BackendType
from ..ssd import SSD, SSDPartition
from .base import Backend


@dataclass
class DecisionDiagramBackend(Backend):
    """Simulation backend using MQT's decision diagram package."""

    backend: BackendType = BackendType.DECISION_DIAGRAM
    num_qubits: int = field(default=0, init=False)
    history: list[str] = field(default_factory=list, init=False)
    state: dd.VectorDD | None = field(default=None, init=False)
    package: dd.DDPackage | None = field(default=None, init=False)
    _benchmark_mode: bool = field(default=False, init=False)
    _benchmark_ops: List[Tuple[str, Sequence[int], Dict[str, float] | None]] = field(
        default_factory=list, init=False
    )
    _benchmark_state: dd.VectorDD | None = field(default=None, init=False)

    _ALIASES: Dict[str, str] = field(
        default_factory=lambda: {"SDG": "sdg", "SXDG": "sxdg", "TDG": "tdg", "VDG": "vdg", "U1": "p"}
    )

    # ------------------------------------------------------------------
    def load(self, num_qubits: int, **_: dict) -> None:
        """Initialise a new zero state for ``num_qubits`` qubits."""

        self.package = dd.DDPackage(num_qubits)
        self.state = self.package.zero_state(num_qubits)
        self.package.inc_ref_vec(self.state)
        self.num_qubits = num_qubits
        self.history.clear()
        self._benchmark_state = None

    def ingest(
        self,
        state: object,
        *,
        num_qubits: int | None = None,
        mapping: Sequence[int] | None = None,
    ) -> None:
        """Initialise the backend from an existing ``(n, VectorDD)`` pair."""

        if not (isinstance(state, tuple) and len(state) == 2):
            raise TypeError("Unsupported state for decision diagram backend")
        n, vec = state
        n = int(n)
        if num_qubits is None:
            num_qubits = n
        if mapping is None:
            mapping = list(range(n))
        if len(mapping) != n:
            raise ValueError("Mapping length does not match state size")
        if mapping != list(range(n)):
            raise NotImplementedError("Qubit mapping not supported for decision diagrams")
        self.num_qubits = num_qubits
        if self.package is None:
            self.package = dd.DDPackage(self.num_qubits)
        elif isinstance(self.state, dd.VectorDD):
            self.package.dec_ref_vec(self.state)
        if isinstance(vec, dd.VectorDD) and num_qubits == n:
            self.package.inc_ref_vec(vec)
            self.state = vec
        else:  # stub environments provide integer handles; ignore and start from |0>
            self.state = self.package.zero_state(self.num_qubits)
            self.package.inc_ref_vec(self.state)
        self.history.clear()
        self._benchmark_state = None

    # ------------------------------------------------------------------
    def _standard_operation(
        self, name: str, qubits: Sequence[int], params: Sequence[float]
    ) -> operations.Operation:
        name_l = self._ALIASES.get(name.upper(), name.lower())
        if name_l.startswith("c"):
            base = name_l[1:]
            if hasattr(operations.OpType, base):
                control = operations.Control(qubits[0])
                target = qubits[1]
                op_type = getattr(operations.OpType, base)
                return operations.StandardOperation({control}, target, op_type, list(params))
        op_type = getattr(operations.OpType, name_l)
        return operations.StandardOperation(list(qubits), op_type, list(params))

    # ------------------------------------------------------------------
    def apply_gate(
        self,
        name: str,
        qubits: Sequence[int],
        params: Dict[str, float] | None = None,
    ) -> None:
        if self._benchmark_mode:
            self._benchmark_ops.append((name, tuple(qubits), params))
            return
        self._benchmark_state = None
        if self.package is None or self.state is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        if not isinstance(self.state, dd.VectorDD):
            raise TypeError("Backend state is not a VectorDD")

        lname = name.upper()
        if lname in {"CCX", "CCZ", "MCX"}:
            raise NotImplementedError(
                "CCX, CCZ and MCX gates must be decomposed before execution"
            )
        if lname == "CP" and params and "k" in params:
            theta = 2 * np.pi / (2 ** float(params["k"]))
            param_list = [theta]
        else:
            param_list = list(params.values()) if params else []
        op = self._standard_operation(name, qubits, param_list)
        new_state = self.package.apply_unitary_operation(self.state, op)
        self.package.inc_ref_vec(new_state)
        self.package.dec_ref_vec(self.state)
        self.state = new_state
        self.history.append(name.upper())

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Apply any operations queued during benchmark preparation."""

        if not self._benchmark_ops:
            return
        ops = self._benchmark_ops
        self._benchmark_ops = []
        self._benchmark_mode = False
        for name, qubits, params in ops:
            self.apply_gate(name, qubits, params)

    # ------------------------------------------------------------------
    def run_benchmark(self, *, return_state: bool = False) -> SSD | None:
        """Apply queued gates and optionally return the final state."""

        self.run()
        self._benchmark_state = self.state if isinstance(self.state, dd.VectorDD) else None
        if return_state:
            return self.extract_ssd()
        return None

    # ------------------------------------------------------------------
    def extract_ssd(self) -> SSD:
        state = self._benchmark_state
        if state is None:
            self.run()
            state = self.state
            self._benchmark_state = state if isinstance(state, dd.VectorDD) else None
        if self.package is None or state is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
            state=(self.num_qubits, state),
        )
        return SSD([part])

    # ------------------------------------------------------------------
    def statevector(self) -> np.ndarray:
        """Return a dense statevector for the current decision diagram state."""

        self.run()
        if self.package is None or self.state is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        if not isinstance(self.state, dd.VectorDD):
            raise TypeError("Backend state is not a VectorDD")
        vec = self.state.get_vector()
        return np.array(vec, dtype=complex)

