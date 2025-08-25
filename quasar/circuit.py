"""Circuit representation and loading utilities for QuASAr.

This module defines a minimal :class:`Circuit` abstraction used by
QuASAr to perform cost estimation and prepare inputs for the
conversion engine. The implementation currently focuses on circuit
I/O; cost estimation and SSD creation are represented by placeholder
methods to be fleshed out in later stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import json


@dataclass
class Gate:
    """Lightweight container describing a quantum gate.

    Attributes
    ----------
    name:
        The symbolic name of the gate, e.g. ``"H"`` or ``"CX"``.
    qubits:
        List of qubit indices the gate acts on.
    params:
        Optional parameter dictionary, for gates such as rotations.
    """

    name: str
    qubits: List[int]
    params: Dict[str, Any] = field(default_factory=dict)


class Circuit:
    """In-memory representation of a quantum circuit."""

    def __init__(self, gates: Iterable[Gate]):
        self.gates: List[Gate] = list(gates)
        self.num_qubits: int = 0
        self._ssd = None
        self._cost = 0
        self._infer_qubit_range()
        self._ssd = self._create_ssd()
        self._cost = self._estimate_cost()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Iterable[Dict[str, Any]]) -> "Circuit":
        """Construct a circuit from an iterable of gate dictionaries."""
        gates = [Gate(d["gate"], list(d["qubits"]), dict(d.get("params", {}))) for d in data]
        return cls(gates)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Circuit":
        """Construct a circuit from a JSON file."""
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _infer_qubit_range(self) -> None:
        """Infer the number of qubits from the gate list."""
        if not self.gates:
            self.num_qubits = 0
            return
        qubit_indices = [q for gate in self.gates for q in gate.qubits]
        if not qubit_indices:
            self.num_qubits = 0
            return
        self.num_qubits = max(qubit_indices) - min(qubit_indices) + 1

    # Placeholder methods -------------------------------------------------
    def _create_ssd(self) -> Optional[object]:
        """Placeholder for the Semi-Stabilizer Diagram (SSD) creation."""
        return None

    def _estimate_cost(self) -> int:
        """Placeholder for circuit cost estimation."""
        return 0

    # Public accessors ----------------------------------------------------
    @property
    def ssd(self) -> Optional[object]:
        """Return the underlying SSD representation, if available."""
        return self._ssd

    @property
    def cost(self) -> int:
        """Return the estimated cost of simulating this circuit."""
        return self._cost


def load_circuit(source: Union[Iterable[Dict[str, Any]], str, Path]) -> Circuit:
    """Load a circuit from either an iterable of dictionaries or a JSON file path."""
    if isinstance(source, (str, Path)):
        return Circuit.from_json(source)
    return Circuit.from_dict(source)
