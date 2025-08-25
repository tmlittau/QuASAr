"""Python stub for the C++ conversion engine.

This module provides a lightweight Python implementation that mirrors the
API of the intended C++ backend.  It allows unit tests to exercise the
conversion heuristics without requiring a compiled extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


@dataclass
class SSD:
    boundary_qubits: List[int] | None = None
    top_s: int = 0


class Backend(Enum):
    StimTableau = 0
    DecisionDiagram = 1


class Primitive(Enum):
    B2B = 0
    LW = 1
    ST = 2
    Full = 3


@dataclass
class ConversionResult:
    primitive: Primitive
    cost: float


class ConversionEngine:
    def estimate_cost(self, fragment_size: int, backend: Backend) -> Tuple[float, float]:
        time_cost = float(fragment_size)
        mem_cost = fragment_size * 0.1
        if backend == Backend.DecisionDiagram:
            time_cost *= 1.5
        return time_cost, mem_cost

    def extract_ssd(self, qubits: List[int], s: int) -> SSD:
        return SSD(boundary_qubits=list(qubits), top_s=s)

    def convert(self, ssd: SSD) -> ConversionResult:
        boundary = len(ssd.boundary_qubits or [])
        rank = ssd.top_s

        if rank <= 4 and boundary <= 6:
            primitive = Primitive.B2B
            cost = rank ** 3
        elif boundary <= 10:
            primitive = Primitive.LW
            cost = 2 ** min(boundary, 4)
        elif rank <= 16:
            primitive = Primitive.ST
            chi = min(rank, 8)
            cost = chi ** 3
        else:
            primitive = Primitive.Full
            cost = 2 ** min(boundary, 16)

        return ConversionResult(primitive=primitive, cost=float(cost))

    # Optional helpers -------------------------------------------------
    def convert_boundary_to_tableau(self, ssd: SSD):
        class Tableau:
            def __init__(self, n: int):
                self.num_qubits = n

        return Tableau(len(ssd.boundary_qubits or []))

    def convert_boundary_to_dd(self, ssd: SSD):
        return object()


__all__ = [
    "SSD",
    "Backend",
    "Primitive",
    "ConversionResult",
    "ConversionEngine",
]

