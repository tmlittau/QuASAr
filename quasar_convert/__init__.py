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
        CompressionStats,
        StnTensor,
        MPS,
        StimTableau,
        ConversionEngine as _CEngine,
    )

    class ConversionEngine:
        """Thin Python wrapper around the C++ implementation with caching."""

        def __init__(
            self,
            cache_limit: int | None = None,
            st_chi_cap: int = 16,
            *,
            truncation_tolerance: float = 0.0,
            truncation_max_terms: int | None = None,
            truncation_normalise: bool = True,
        ) -> None:
            self._cache_limit = cache_limit
            self._ssd_cache: OrderedDict[tuple, SSD] = OrderedDict()
            self._boundary_cache: OrderedDict[tuple, SSD] = OrderedDict()
            self._bridge_cache: OrderedDict[tuple, list] = OrderedDict()
            self.st_chi_cap = st_chi_cap
            self.truncation_tolerance = truncation_tolerance
            self.truncation_max_terms = truncation_max_terms
            self.truncation_normalise = truncation_normalise

        def _ensure_impl(self) -> None:
            if "_impl" not in self.__dict__:
                self.__dict__["_impl"] = _CEngine()
                self._impl.st_chi_cap = self.st_chi_cap
                self._impl.truncation_tolerance = float(self.truncation_tolerance)
                max_terms = self.truncation_max_terms
                self._impl.truncation_max_terms = int(max_terms) if max_terms else 0
                self._impl.truncation_normalise = bool(self.truncation_normalise)

        @property
        def truncation_tolerance(self) -> float:
            return self.__dict__.get("_truncation_tolerance", 0.0)

        @truncation_tolerance.setter
        def truncation_tolerance(self, value: float) -> None:
            self.__dict__["_truncation_tolerance"] = value
            if "_impl" in self.__dict__:
                self._impl.truncation_tolerance = float(value)

        @property
        def truncation_max_terms(self) -> int | None:
            return self.__dict__.get("_truncation_max_terms")

        @truncation_max_terms.setter
        def truncation_max_terms(self, value: int | None) -> None:
            self.__dict__["_truncation_max_terms"] = value
            if "_impl" in self.__dict__:
                self._impl.truncation_max_terms = int(value) if value else 0

        @property
        def truncation_normalise(self) -> bool:
            return bool(self.__dict__.get("_truncation_normalise", True))

        @truncation_normalise.setter
        def truncation_normalise(self, value: bool) -> None:
            self.__dict__["_truncation_normalise"] = bool(value)
            if "_impl" in self.__dict__:
                self._impl.truncation_normalise = bool(value)

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

        if hasattr(_CEngine, "extract_local_window_dd"):

            def extract_local_window_dd(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.extract_local_window_dd(*args, **kwargs)

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

        if hasattr(_CEngine, "last_compression_stats"):

            def last_compression_stats(self) -> CompressionStats:  # type: ignore[override]
                self._ensure_impl()
                return self._impl.last_compression_stats()

        if hasattr(_CEngine, "compressed_cardinality"):

            def compressed_cardinality(self) -> int:  # type: ignore[override]
                self._ensure_impl()
                return int(self._impl.compressed_cardinality())

        if hasattr(_CEngine, "convert_boundary_to_stn"):

            def convert_boundary_to_stn(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.convert_boundary_to_stn(*args, **kwargs)

        if hasattr(_CEngine, "mps_to_statevector"):

            def mps_to_statevector(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.mps_to_statevector(*args, **kwargs)

        if hasattr(_CEngine, "dense_statevector_queries"):

            def dense_statevector_queries(self) -> int:  # type: ignore[override]
                self._ensure_impl()
                return int(self._impl.dense_statevector_queries())

        if hasattr(_CEngine, "convert_boundary_to_tableau"):

            def convert_boundary_to_tableau(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.convert_boundary_to_tableau(*args, **kwargs)

        if hasattr(_CEngine, "dd_to_tableau"):

            def dd_to_tableau(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.dd_to_tableau(*args, **kwargs)

        if hasattr(_CEngine, "tableau_to_statevector"):

            def tableau_to_statevector(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.tableau_to_statevector(*args, **kwargs)

        if hasattr(_CEngine, "tableau_to_mps"):

            def tableau_to_mps(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.tableau_to_mps(*args, **kwargs)

        if hasattr(_CEngine, "tableau_to_dd"):

            def tableau_to_dd(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.tableau_to_dd(*args, **kwargs)

        if hasattr(_CEngine, "convert_boundary_to_dd"):

            def convert_boundary_to_dd(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.convert_boundary_to_dd(*args, **kwargs)

        if hasattr(_CEngine, "dd_to_statevector"):

            def dd_to_statevector(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.dd_to_statevector(*args, **kwargs)

        if hasattr(_CEngine, "dd_to_mps"):

            def dd_to_mps(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.dd_to_mps(*args, **kwargs)

        if hasattr(_CEngine, "learn_stabilizer"):

            def learn_stabilizer(self, *args, **kwargs):  # type: ignore[override]
                self._ensure_impl()
                return self._impl.learn_stabilizer(*args, **kwargs)

    __all__ = [
        "SSD",
        "Backend",
        "Primitive",
        "ConversionResult",
        "CompressionStats",
        "StnTensor",
        "MPS",
        "StimTableau",
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
    class CompressionStats:
        original_terms: int = 0
        retained_terms: int = 0
        fidelity: float = 1.0

    @dataclass
    class StnTensor:
        amplitudes: List[complex]
        tableau: object | None = None

    @dataclass
    class MPS:
        tensors: List[List[complex]] | None = None
        bond_dims: List[int] | None = None

    class ConversionEngine:
        def __init__(
            self,
            cache_limit: int | None = None,
            st_chi_cap: int = 16,
            *,
            truncation_tolerance: float = 0.0,
            truncation_max_terms: int | None = None,
            truncation_normalise: bool = True,
        ) -> None:
            self._cache_limit = cache_limit
            self._ssd_cache: OrderedDict[tuple, SSD] = OrderedDict()
            self._boundary_cache: OrderedDict[tuple, SSD] = OrderedDict()
            self._bridge_cache: OrderedDict[tuple, list] = OrderedDict()
            self.st_chi_cap = st_chi_cap
            self.truncation_tolerance = truncation_tolerance
            self.truncation_max_terms = truncation_max_terms
            self.truncation_normalise = truncation_normalise
            self._compression_stats = CompressionStats()

        def _apply_truncation(self, state: List[complex]) -> List[complex]:
            stats = CompressionStats(original_terms=len(state), retained_terms=len(state), fidelity=1.0)
            if not state:
                self._compression_stats = stats
                return state
            tol = float(self.truncation_tolerance or 0.0)
            max_terms = int(self.truncation_max_terms) if self.truncation_max_terms else 0
            if tol <= 0.0 and max_terms == 0:
                self._compression_stats = stats
                return state
            magnitudes = [abs(val) ** 2 for val in state]
            total_norm = sum(magnitudes)
            if total_norm <= 0.0:
                stats.retained_terms = 0
                stats.fidelity = 1.0
                self._compression_stats = stats
                return [0j] * len(state)
            threshold_sq = tol * tol if tol > 0.0 else 0.0
            keep = [idx for idx, mag in enumerate(magnitudes) if threshold_sq == 0.0 or mag >= threshold_sq]
            if not keep:
                keep = [max(range(len(magnitudes)), key=magnitudes.__getitem__)]
            if max_terms and len(keep) > max_terms:
                keep = sorted(keep, key=lambda idx: magnitudes[idx], reverse=True)[:max_terms]
                keep.sort()
            retained_norm = sum(magnitudes[idx] for idx in keep)
            if retained_norm <= 0.0:
                stats.retained_terms = 0
                stats.fidelity = 1.0
                self._compression_stats = stats
                return [0j] * len(state)
            result = [0j] * len(state)
            scale = (total_norm / retained_norm) ** 0.5 if self.truncation_normalise and retained_norm > 0.0 else 1.0
            for idx in keep:
                result[idx] = state[idx] * scale
            stats.retained_terms = len(keep)
            stats.fidelity = min(1.0, retained_norm / total_norm if total_norm else 1.0)
            self._compression_stats = stats
            return result

        def last_compression_stats(self) -> CompressionStats:
            return self._compression_stats

        def compressed_cardinality(self) -> int:
            stats = self._compression_stats
            return stats.retained_terms or stats.original_terms

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

        def mps_to_statevector(self, mps: MPS) -> List[complex]:
            tensors = mps.tensors or []
            if not tensors:
                return []
            bonds = mps.bond_dims or []
            n = len(tensors)
            if len(bonds) != n + 1:
                bonds = [1]
                for t in tensors:
                    right = len(t) // (bonds[-1] * 2)
                    bonds.append(right)
            first = bonds[1]
            current = [0j] * (2 * first)
            t0 = tensors[0]
            for p in range(2):
                for r in range(first):
                    current[p * first + r] = t0[p * first + r]
            left_dim = 2
            for q in range(1, n):
                chi = bonds[q]
                next_chi = bonds[q + 1]
                tensor = tensors[q]
                next_state = [0j] * (left_dim * 2 * next_chi)
                for i in range(left_dim):
                    for k in range(chi):
                        coeff = current[i * chi + k]
                        if coeff == 0j:
                            continue
                        for p in range(2):
                            for r in range(next_chi):
                                idx = (i * 2 + p) * next_chi + r
                                next_state[idx] += coeff * tensor[(k * 2 + p) * next_chi + r]
                current = next_state
                left_dim *= 2
            final = bonds[n]
            state = [current[i * final] for i in range(left_dim)]
            return self._apply_truncation(state)

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
            return self._apply_truncation(window)

        def convert(
            self,
            ssd: SSD,
            window_1q_gates: int = 0,
            window_2q_gates: int = 0,
            window: int | None = None,
        ) -> ConversionResult:
            boundary = len(ssd.boundary_qubits or [])
            rank = ssd.top_s

            if window is None:
                window_size = min(boundary, 4)
            else:
                window_size = min(boundary, max(window, 0))
            dense = 1 << window_size
            chi_tilde = min(rank, self.st_chi_cap)
            full = 1 << min(boundary, 16)

            svd_cost = min(boundary * (rank ** 2), rank * (boundary ** 2))
            cost_b2b = svd_cost + boundary * (rank ** 2) + rank ** 2
            gate_time = window_1q_gates + window_2q_gates
            cost_lw = (2.0 + gate_time) * dense
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
            return self._apply_truncation(state)

        def convert_boundary_to_stn(self, ssd: SSD) -> StnTensor:
            state = self.convert_boundary_to_statevector(ssd)
            tab = self.learn_stabilizer(state)
            return StnTensor(amplitudes=state, tableau=tab)

        def convert_boundary_to_tableau(self, ssd: SSD):
            class Tableau:
                def __init__(self, n: int):
                    self.num_qubits = n

            return Tableau(len(ssd.boundary_qubits or []))

        def dd_to_tableau(self, *args, **kwargs):
            return None

        def convert_boundary_to_dd(self, ssd: SSD):
            return (len(ssd.boundary_qubits or []), 0)

        def clone_dd_edge(self, num_qubits: int, edge: object, package: object):
            """Fallback clone helper using dense vectors when the extension is missing."""

            try:
                from mqt.core import dd as mqt_dd  # type: ignore
            except Exception:  # pragma: no cover - optional dependency
                mqt_dd = None
            if mqt_dd is not None and isinstance(edge, mqt_dd.VectorDD):
                try:
                    amps = edge.get_vector()
                    if hasattr(package, "from_vector"):
                        clone = package.from_vector(amps)
                        return (num_qubits, clone)
                except Exception:
                    pass
            return (num_qubits, edge)

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

        def dense_statevector_queries(self) -> int:
            return 0

    __all__ = [
        "SSD",
        "Backend",
        "Primitive",
        "ConversionResult",
        "CompressionStats",
        "StnTensor",
        "MPS",
        "ConversionEngine",
    ]

