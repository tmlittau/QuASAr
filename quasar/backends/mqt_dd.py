"""Decision diagram backend based on :class:`mqt.core.dd.DDPackage`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

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

    def ingest(self, state: object) -> None:
        """Initialise the backend from an existing ``(n, VectorDD)`` pair."""

        if not (isinstance(state, tuple) and len(state) == 2):
            raise TypeError("Unsupported state for decision diagram backend")
        n, vec = state
        self.num_qubits = int(n)
        self.package = dd.DDPackage(self.num_qubits)
        if isinstance(vec, dd.VectorDD):
            self.package.inc_ref_vec(vec)
            self.state = vec
        else:  # stub environments provide integer handles; ignore and start from |0>
            self.state = self.package.zero_state(self.num_qubits)
            self.package.inc_ref_vec(self.state)
        self.history.clear()

    # ------------------------------------------------------------------
    def _standard_operation(
        self, name: str, qubits: Sequence[int], params: Sequence[float]
    ) -> operations.Operation:
        name_l = self._ALIASES.get(name.upper(), name.lower())
        if name_l.startswith("c") and len(name_l) == 2:
            # simple single-control gate like CX, CZ, CY
            control = operations.Control(qubits[0])
            target = qubits[1]
            op_type = getattr(operations.OpType, name_l[1:])
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

        if self.package is None or self.state is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        if not isinstance(self.state, dd.VectorDD):
            raise TypeError("Backend state is not a VectorDD")

        op = self._standard_operation(name, qubits, list(params.values()) if params else [])
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
    def extract_ssd(self) -> SSD:
        self.run()
        if self.package is None or self.state is None:
            raise RuntimeError("Backend not initialised; call 'load' first")
        part = SSDPartition(
            subsystems=(tuple(range(self.num_qubits)),),
            history=tuple(self.history),
            backend=self.backend,
            state=(self.num_qubits, self.state),
        )
        return SSD([part])

    # ------------------------------------------------------------------
    def statevector(self) -> Sequence[complex]:
        self.run()
        raise NotImplementedError("Statevector extraction not supported")

