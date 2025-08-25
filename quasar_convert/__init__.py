"""Python stub for the C++ conversion engine.

This module provides a lightweight Python implementation that mirrors the
API of the intended C++ backend.  It allows unit tests to exercise the
conversion heuristics without requiring a compiled extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
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

    def extract_boundary_ssd(self, bridges: List[Tuple[int, int]], s: int) -> SSD:
        boundary = sorted({a for a, _ in bridges})
        return SSD(boundary_qubits=boundary, top_s=s)

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
    def extract_local_window(self, state: List[complex], window_qubits: List[int]) -> List[complex]:
        n = int(math.log2(len(state)))
        k = len(window_qubits)
        dim = 1 << k
        window = [0j] * dim
        for idx, amp in enumerate(state):
            match = True
            for q in range(n):
                if q not in window_qubits and ((idx >> q) & 1):
                    match = False
                    break
            if not match:
                continue
            local = 0
            for i, q in enumerate(window_qubits):
                if (idx >> q) & 1:
                    local |= 1 << i
            window[local] = amp
        return window

    def build_bridge_tensor(self, left: SSD, right: SSD) -> List[complex]:
        total = len(left.boundary_qubits or []) + len(right.boundary_qubits or [])
        dim = 1 << total
        tensor = [0j] * dim
        if dim:
            tensor[0] = 1.0 + 0j
        return tensor

    def convert_boundary_to_tableau(self, ssd: SSD):
        class Tableau:
            def __init__(self, n: int):
                self.num_qubits = n

        return Tableau(len(ssd.boundary_qubits or []))

    def convert_boundary_to_dd(self, ssd: SSD):
        return object()

    def learn_stabilizer(self, state: List[complex]):
        if not state:
            return None
        dim = len(state)
        n = int(math.log2(dim))
        # |0...0>
        if abs(state[0] - 1) < 1e-9 and all(abs(a) < 1e-9 for a in state[1:]):
            class Tableau:
                def __init__(self, n: int):
                    self.num_qubits = n

            return Tableau(n)
        target = 1 / math.sqrt(dim)
        if all(abs(abs(a) - target) < 1e-9 for a in state):
            class Tableau:
                def __init__(self, n: int):
                    self.num_qubits = n

            return Tableau(n)
        return None


__all__ = [
    "SSD",
    "Backend",
    "Primitive",
    "ConversionResult",
    "ConversionEngine",
]

