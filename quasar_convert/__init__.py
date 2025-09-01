"""Python interface to the optional native conversion engine.

The package ships with a C++ implementation exposed via pybind11.  When the
compiled extension is unavailable (for example on platforms without a
compiler) a lightweight Python stub is used instead so that the rest of the
package continues to function.
"""

from __future__ import annotations

from collections import OrderedDict

try:  # pragma: no cover - exercised when the extension is available
    from ._conversion_engine import (  # type: ignore[attr-defined]
        SSD,
        Backend,
        Primitive,
        ConversionResult,
        StnTensor,
        ConversionEngine as _CEngine,
    )

    class ConversionEngine:
        """Thin Python wrapper around the C++ implementation with caching."""

        def __init__(self, cache_limit: int | None = None) -> None:
            self._cache_limit = cache_limit
            self._ssd_cache: OrderedDict[tuple, SSD] = OrderedDict()
            self._boundary_cache: OrderedDict[tuple, SSD] = OrderedDict()
            self._bridge_cache: OrderedDict[tuple, list] = OrderedDict()

        def _ensure_impl(self) -> None:
            if "_impl" not in self.__dict__:
                self.__dict__["_impl"] = _CEngine()

        # Cache helpers -------------------------------------------------
        def _trim_cache(self, cache: OrderedDict) -> None:
            if self._cache_limit is not None:
                while len(cache) > self._cache_limit:
                    cache.popitem(last=False)

        def clear_cache(self) -> None:
            self._ssd_cache.clear()
            self._boundary_cache.clear()
            self._bridge_cache.clear()

        def set_cache_limit(self, limit: int | None) -> None:
            self._cache_limit = limit
            for c in (self._ssd_cache, self._boundary_cache, self._bridge_cache):
                if limit is not None:
                    while len(c) > limit:
                        c.popitem(last=False)

        # Forwarding methods with caching -------------------------------
        def estimate_cost(self, *args, **kwargs):
            self._ensure_impl()
            return self._impl.estimate_cost(*args, **kwargs)

        def extract_ssd(self, qubits, s):
            key = (tuple(qubits), s)
            if key not in self._ssd_cache:
                self._ensure_impl()
                self._ssd_cache[key] = self._impl.extract_ssd(qubits, s)
                self._trim_cache(self._ssd_cache)
            return self._ssd_cache[key]

        def extract_boundary_ssd(self, bridges, s):
            key = (tuple(tuple(b) for b in bridges), s)
            if key not in self._boundary_cache:
                self._ensure_impl()
                self._boundary_cache[key] = self._impl.extract_boundary_ssd(bridges, s)
                self._trim_cache(self._boundary_cache)
            return self._boundary_cache[key]

        def extract_local_window(self, *args, **kwargs):
            self._ensure_impl()
            return self._impl.extract_local_window(*args, **kwargs)

        def convert(self, *args, **kwargs):
            self._ensure_impl()
            return self._impl.convert(*args, **kwargs)

        def build_bridge_tensor(self, left, right):
            key = (tuple(left.boundary_qubits or []), tuple(right.boundary_qubits or []))
            if key not in self._bridge_cache:
                self._ensure_impl()
                self._bridge_cache[key] = self._impl.build_bridge_tensor(left, right)
                self._trim_cache(self._bridge_cache)
            return self._bridge_cache[key]

        def convert_boundary_to_statevector(self, *args, **kwargs):  # type: ignore[override]
            self._ensure_impl()
            return self._impl.convert_boundary_to_statevector(*args, **kwargs)

        if hasattr(_CEngine, "convert_boundary_to_stn"):

            def convert_boundary_to_stn(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.convert_boundary_to_stn(*args, **kwargs)

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
        "StnTensor",
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
        fidelity: float

    @dataclass
    class StnTensor:
        amplitudes: List[complex]
        tableau: object | None = None

    class ConversionEngine:
        def __init__(self, cache_limit: int | None = None) -> None:
            self._cache_limit = cache_limit
            self._ssd_cache: OrderedDict[tuple, SSD] = OrderedDict()
            self._boundary_cache: OrderedDict[tuple, SSD] = OrderedDict()
            self._bridge_cache: OrderedDict[tuple, list] = OrderedDict()

        # Cache utilities -----------------------------------------------
        def _trim_cache(self, cache: OrderedDict) -> None:
            if self._cache_limit is not None:
                while len(cache) > self._cache_limit:
                    cache.popitem(last=False)

        def clear_cache(self) -> None:
            self._ssd_cache.clear()
            self._boundary_cache.clear()
            self._bridge_cache.clear()

        def set_cache_limit(self, limit: int | None) -> None:
            self._cache_limit = limit
            for c in (self._ssd_cache, self._boundary_cache, self._bridge_cache):
                if limit is not None:
                    while len(c) > limit:
                        c.popitem(last=False)

        # Core behaviour with caching ----------------------------------
        def estimate_cost(self, fragment_size: int, backend: Backend) -> Tuple[float, float]:
            time_cost = float(fragment_size)
            mem_cost = fragment_size * 0.1
            if backend == Backend.DecisionDiagram:
                time_cost *= 1.5
            return time_cost, mem_cost

        def _extract_ssd_impl(self, qubits: List[int], s: int) -> SSD:
            n = len(qubits)
            k = min(s, n)
            vecs = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(k)]
            return SSD(boundary_qubits=list(qubits), top_s=k, vectors=vecs)

        def extract_ssd(self, qubits: List[int], s: int) -> SSD:
            key = (tuple(qubits), s)
            if key not in self._ssd_cache:
                self._ssd_cache[key] = self._extract_ssd_impl(qubits, s)
                self._trim_cache(self._ssd_cache)
            return self._ssd_cache[key]

        def _extract_boundary_ssd_impl(self, bridges: List[Tuple[int, int]], s: int) -> SSD:
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

        def extract_boundary_ssd(self, bridges: List[Tuple[int, int]], s: int) -> SSD:
            key = (tuple(tuple(b) for b in bridges), s)
            if key not in self._boundary_cache:
                self._boundary_cache[key] = self._extract_boundary_ssd_impl(bridges, s)
                self._trim_cache(self._boundary_cache)
            return self._boundary_cache[key]

        def _build_bridge_tensor_impl(self, left: SSD, right: SSD) -> list[complex]:
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

        def build_bridge_tensor(self, left: SSD, right: SSD) -> list[complex]:
            key = (tuple(left.boundary_qubits or []), tuple(right.boundary_qubits or []))
            if key not in self._bridge_cache:
                self._bridge_cache[key] = self._build_bridge_tensor_impl(left, right)
                self._trim_cache(self._bridge_cache)
            return self._bridge_cache[key]

        # Optional helpers ---------------------------------------------
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

        def convert(self, ssd: SSD) -> ConversionResult:
            boundary = len(ssd.boundary_qubits or [])
            rank = ssd.top_s

            window = min(boundary, 4)
            dense = 1 << window
            chi_tilde = min(rank, 16)
            full = 1 << min(boundary, 16)

            cost_b2b = rank ** 3 + boundary * (rank ** 2) + rank ** 2
            cost_lw = 2.0 * dense
            cost_st = chi_tilde ** 3 + chi_tilde ** 2
            cost_full = 2.0 * full

            if rank <= 4 and boundary <= 6:
                primitive = Primitive.B2B
                cost = float(cost_b2b)
                fidelity = (rank / boundary) if boundary else 1.0
            elif boundary <= 10:
                primitive = Primitive.LW
                cost = float(cost_lw)
                fidelity = 1.0
            elif rank <= 16:
                primitive = Primitive.ST
                cost = float(cost_st)
                fidelity = (chi_tilde / rank) if rank else 1.0
            else:
                primitive = Primitive.Full
                cost = float(cost_full)
                fidelity = 1.0

            if fidelity > 1.0:
                fidelity = 1.0

            return ConversionResult(primitive=primitive, cost=cost, fidelity=float(fidelity))

        def convert_boundary_to_statevector(self, ssd: SSD) -> List[complex]:
            dim = 1 << len(ssd.boundary_qubits or [])
            state = [0j] * dim
            if dim:
                norm = 1.0 / math.sqrt(dim)
                phases = [1.0 + 0j] * len(ssd.boundary_qubits or [])
                vecs = ssd.vectors or []
                if vecs:
                    for i, val in enumerate(vecs[0][: len(phases)]):
                        phases[i] = (-1.0 + 0j) if val < 0 else (1.0 + 0j)
                for idx in range(dim):
                    amp = 1.0 + 0j
                    for bit in range(len(phases)):
                        if idx >> bit & 1:
                            amp *= phases[bit]
                    state[idx] = amp * norm
            return state

        def convert_boundary_to_stn(self, ssd: SSD) -> StnTensor:
            state = self.convert_boundary_to_statevector(ssd)
            tab = self.learn_stabilizer(state)
            return StnTensor(amplitudes=state, tableau=tab)

        def convert_boundary_to_tableau(self, ssd: SSD):
            class Tableau:
                def __init__(self, n: int):
                    self.num_qubits = n

            return Tableau(len(ssd.boundary_qubits or []))

        def convert_boundary_to_dd(self, ssd: SSD):
            return (len(ssd.boundary_qubits or []), 0)

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
        "StnTensor",
        "ConversionEngine",
    ]

