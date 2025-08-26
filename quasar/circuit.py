"""Circuit representation and loading utilities for QuASAr."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List
import json
import os

from qiskit.circuit import QuantumCircuit
from qiskit_qasm3_import import api as qasm3_api

from .partitioner import Partitioner
from .ssd import SSD


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
        self.num_qubits = self._infer_qubit_count()
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

    def _create_ssd(self) -> SSD:
        """Construct the initial subsystem descriptor."""
        return Partitioner().partition(self)

    def _estimate_costs(self) -> Dict[str, float]:
        """Summarise estimated simulation and conversion costs.

        The subsystem descriptor produced by :class:`Partitioner` stores a
        :class:`~quasar.cost.Cost` object for each partition and conversion
        layer.  This routine aggregates those estimates to provide a concise
        circuit level view.  Runtime contributions are summed whereas memory
        requirements track the largest individual peak, mirroring the
        ``_add_cost`` helper used by the planner.

        Returns
        -------
        Dict[str, float]
            Mapping containing per-backend runtime and memory summaries as
            well as overall totals.  Keys follow ``"{backend}_time"`` and
            ``"{backend}_mem"`` naming conventions using backend short names
            (``sv``, ``tab``, ``mps`` and ``dd``).
        """

        if not self.ssd.partitions and not self.ssd.conversions:
            return {}

        estimates: Dict[str, float] = {}
        total_time = 0.0
        peak_memory = 0.0

        # Aggregate cost for each simulation backend.
        for backend, parts in self.ssd.by_backend().items():
            time = sum(p.cost.time * p.multiplicity for p in parts)
            memory = max((p.cost.memory for p in parts), default=0.0)

            estimates[f"{backend.value}_time"] = time
            estimates[f"{backend.value}_mem"] = memory

            total_time += time
            peak_memory = max(peak_memory, memory)

        # Include conversion layers between partitions.
        conv_time = sum(c.cost.time for c in self.ssd.conversions)
        conv_mem = max((c.cost.memory for c in self.ssd.conversions), default=0.0)
        if conv_time > 0.0:
            estimates["conversion_time"] = conv_time
            total_time += conv_time
        if conv_mem > 0.0:
            estimates["conversion_mem"] = conv_mem
            peak_memory = max(peak_memory, conv_mem)

        estimates["total_time"] = total_time
        estimates["peak_memory"] = peak_memory
        return estimates
