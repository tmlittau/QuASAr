"""Python interface to the optional native conversion engine.

The package ships with a C++ implementation exposed via pybind11.  When the
compiled extension is unavailable (for example on platforms without a
compiler) a lightweight Python stub is used instead so that the rest of the
package continues to function.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised when the extension is available
    from ._conversion_engine import (  # type: ignore[attr-defined]
        SSD,
        Backend,
        Primitive,
        ConversionResult,
        ConversionEngine as _CEngine,
    )

    class ConversionEngine:
        """Thin Python wrapper around the C++ implementation.

        The wrapper lazily constructs the underlying C++ object which allows
        subclasses to omit a call to ``super().__init__`` in their ``__init__``
        methods.  All public methods simply forward to the native instance.
        """

        def _ensure_impl(self) -> None:
            if "_impl" not in self.__dict__:
                self.__dict__["_impl"] = _CEngine()

        def estimate_cost(self, *args, **kwargs):
            self._ensure_impl()
            return self._impl.estimate_cost(*args, **kwargs)

        def extract_ssd(self, *args, **kwargs):
            self._ensure_impl()
            return self._impl.extract_ssd(*args, **kwargs)

        def extract_boundary_ssd(self, *args, **kwargs):
            self._ensure_impl()
            return self._impl.extract_boundary_ssd(*args, **kwargs)

        def extract_local_window(self, *args, **kwargs):
            self._ensure_impl()
            return self._impl.extract_local_window(*args, **kwargs)

        def convert(self, *args, **kwargs):
            self._ensure_impl()
            return self._impl.convert(*args, **kwargs)

        def build_bridge_tensor(self, *args, **kwargs):
            self._ensure_impl()
            return self._impl.build_bridge_tensor(*args, **kwargs)

        if hasattr(_CEngine, "convert_boundary_to_tableau"):
            def convert_boundary_to_tableau(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.convert_boundary_to_tableau(*args, **kwargs)

        if hasattr(_CEngine, "convert_boundary_to_dd"):
            def convert_boundary_to_dd(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.convert_boundary_to_dd(*args, **kwargs)

        if hasattr(_CEngine, "learn_stabilizer"):
            def learn_stabilizer(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.learn_stabilizer(*args, **kwargs)

    __all__ = [
        "SSD",
        "Backend",
        "Primitive",
        "ConversionResult",
        "ConversionEngine",
    ]
except Exception:  # pragma: no cover - exercised when extension missing
    from dataclasses import dataclass
    from enum import Enum
    import math
    from typing import List, Tuple

    @dataclass
    class SSD:
        boundary_qubits: List[int] | None = None
        top_s: int = 0
        vectors: List[List[float]] | None = None

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
            n = len(qubits)
            k = min(s, n)
            vecs = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(k)]
            return SSD(boundary_qubits=list(qubits), top_s=k, vectors=vecs)

        def extract_boundary_ssd(self, bridges: List[Tuple[int, int]], s: int) -> SSD:
            import numpy as np

            boundary = sorted({a for a, _ in bridges})
            remote = sorted({b for _, b in bridges})
            mat = np.zeros((len(boundary), len(remote)))
            for a, b in bridges:
                i = boundary.index(a)
                j = remote.index(b)
                mat[i, j] += 1.0
            u, _s, _v = np.linalg.svd(mat, full_matrices=False)
            k = min(s, u.shape[1])
            vecs = [u[:, i].tolist() for i in range(k)]
            return SSD(boundary_qubits=boundary, top_s=k, vectors=vecs)

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
            k = len(window_qubits)
            dim = 1 << k
            window = [0j] * dim
            for local in range(dim):
                global_index = 0
                for i, q in enumerate(window_qubits):
                    if (local >> i) & 1:
                        global_index |= 1 << q
                window[local] = state[global_index]
            return window

        def build_bridge_tensor(self, left: SSD, right: SSD) -> List[complex]:
            m = len(left.boundary_qubits or [])
            n = len(right.boundary_qubits or [])
            dim = 1 << (m + n)
            tensor = [0j] * dim
            mask = (1 << min(m, n)) - 1
            for l in range(1 << m):
                for r in range(1 << n):
                    if (l & mask) == (r & mask):
                        tensor[(l << n) | r] = 1.0 + 0j
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
            try:
                import stim

                return stim.Tableau.from_state_vector(state)
            except Exception:
                pass
            dim = len(state)
            n = int(math.log2(dim))
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

