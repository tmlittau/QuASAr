"""Circuit representation and loading utilities for QuASAr."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List
import json


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
        # Placeholders for future SSD and cost estimation logic.
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

    # ------------------------------------------------------------------
    def _infer_qubit_count(self) -> int:
        if not self.gates:
            return 0
        qubit_indices = [q for gate in self.gates for q in gate.qubits]
        min_q = min(qubit_indices)
        max_q = max(qubit_indices)
        return max_q - min_q + 1

    def _create_ssd(self) -> Dict[str, Any]:
        """Placeholder for SSD construction.

        Returns
        -------
        dict
            Currently returns an empty dict. This will be replaced by
            a call into the conversion engine once implemented.
        """
        return {}

    def _estimate_costs(self) -> Dict[str, float]:
        """Placeholder cost estimation routine.

        Returns
        -------
        dict
            A mapping from backend identifiers to estimated costs.
        """
        return {}
