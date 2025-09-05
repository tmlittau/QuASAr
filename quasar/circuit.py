"""Circuit representation and loading utilities for QuASAr."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List
import json
import os

from qiskit.circuit import QuantumCircuit
from qiskit_qasm3_import import api as qasm3_api

from .ssd import SSD, SSDPartition
from .cost import Cost
from .symmetry import symmetry_score


@dataclass
class Gate:
    """Simple gate description used when constructing circuits."""

    gate: str
    qubits: List[int]
    params: Dict[str, Any] = field(default_factory=dict)


class Circuit:
    """High level circuit container.

    Parameters
    ----------
    gates:
        Iterable of :class:`Gate` or dictionaries describing gates.
    """

    def __init__(self, gates: Iterable[Dict[str, Any] | Gate]):
        self.gates: List[Gate] = [g if isinstance(g, Gate) else Gate(**g) for g in gates]
        self._num_gates = len(self.gates)
        self._num_qubits = self._infer_qubit_count()
        self._depth = self._compute_depth()
        self.symmetry = symmetry_score(self)
        self.ssd = self._create_ssd()
        self.cost_estimates = self._estimate_costs()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, gates: Iterable[Dict[str, Any]]):
        """Build a circuit from an iterable of gate dictionaries."""
        return cls(gates)

    @classmethod
    def from_json(cls, path: str):
        """Load a circuit from a JSON file.

        The JSON file must contain a list of gate dictionaries.
        """
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_qiskit(cls, circuit: QuantumCircuit) -> "Circuit":
        """Build a :class:`Circuit` from a Qiskit ``QuantumCircuit``.

        Parameters
        ----------
        circuit:
            The input Qiskit circuit to convert.
        """
        gates = []
        for ci in circuit.data:
            op = ci.operation
            qubits = [q._index for q in ci.qubits]
            params: Dict[str, Any] = {}
            if getattr(op, "params", None):
                for i, val in enumerate(op.params):
                    params[f"param{i}"] = float(val) if isinstance(val, (int, float)) else val
            gates.append({"gate": op.name.upper(), "qubits": qubits, "params": params})
        return cls(gates)

    @classmethod
    def from_qasm(cls, path_or_str: str) -> "Circuit":
        """Build a :class:`Circuit` from an OpenQASM 3 string or file.

        Parameters
        ----------
        path_or_str:
            Either a filesystem path to an OpenQASM 3 file or a string
            containing the OpenQASM program.
        """
        if os.path.exists(path_or_str):
            with open(path_or_str, "r", encoding="utf8") as f:
                qasm = f.read()
        else:
            qasm = path_or_str
        qc = qasm3_api.parse(qasm)
        return cls.from_qiskit(qc)

    # ------------------------------------------------------------------
    def _infer_qubit_count(self) -> int:
        if not self.gates:
            return 0
        qubit_indices = [q for gate in self.gates for q in gate.qubits]
        min_q = min(qubit_indices)
        max_q = max(qubit_indices)
        return max_q - min_q + 1

    def _compute_depth(self) -> int:
        """Compute the circuit depth in a single pass over gates."""
        qubit_levels: Dict[int, int] = {}
        depth = 0
        for gate in self.gates:
            start = max((qubit_levels.get(q, 0) for q in gate.qubits), default=0)
            level = start + 1
            for q in gate.qubits:
                qubit_levels[q] = level
            if level > depth:
                depth = level
        return depth

    def _create_ssd(self) -> SSD:
        """Construct the initial subsystem descriptor."""
        if self._num_qubits == 0:
            return SSD([])
        part = SSDPartition(subsystems=(tuple(range(self._num_qubits)),))
        return SSD([part])

    def _estimate_costs(self) -> Dict[str, Cost]:
        """Estimate simulation costs for standard backends."""

        from .analyzer import CircuitAnalyzer

        analyzer = CircuitAnalyzer(self)
        estimates = analyzer.resource_estimates()
        return {backend.name.lower(): cost for backend, cost in estimates.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def depth(self) -> int:
        """Total circuit depth."""
        return self._depth

    @property
    def num_gates(self) -> int:
        """Number of gates in the circuit."""
        return self._num_gates

    @property
    def num_qubits(self) -> int:
        """Number of qubits spanned by the circuit."""
        return self._num_qubits
