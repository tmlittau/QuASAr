from __future__ import annotations

"""Dynamic programming planner for contiguous circuit partitions.

This module implements the algorithm described in the QuASAr draft for
optimally assigning simulation backends to contiguous circuit fragments.  The
planner evaluates all possible cut positions and backend choices using a
simple dynamic programming (DP) approach.  Each DP table entry stores the
cumulative cost up to a given gate index and acts as a backpointer to recover
an optimal execution plan.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Set, Tuple, Hashable

from .cost import Backend, Cost, CostEstimator
from .partitioner import CLIFFORD_GATES, Partitioner
from .ssd import ConversionLayer
from . import config

if True:  # pragma: no cover - used for type checking when available
    try:
        from .circuit import Circuit, Gate
    except Exception:  # pragma: no cover
        Circuit = Gate = None  # type: ignore


# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """Single contiguous fragment of the circuit.

    Attributes
    ----------
    start, end:
        Gate index range ``[start, end)`` represented by this step.
    backend:
        Simulation backend chosen for this fragment.
    parallel:
        Optional groups of qubits that are independent within the
        fragment and can therefore be executed in parallel.  Each entry
        is a tuple of qubit indices belonging to one independent group.
    """

    start: int
    end: int
    backend: Backend
    parallel: Tuple[Tuple[int, ...], ...] = ()


@dataclass
class DPEntry:
    """Entry in the dynamic programming table."""

    cost: Cost
    prev_index: int
    prev_backend: Optional[Backend]


@dataclass
class PlanResult:
    """Return value of :meth:`Planner.plan`."""

    table: List[Dict[Optional[Backend], DPEntry]]
    final_backend: Optional[Backend]
    gates: List['Gate']
    explicit_steps: Optional[List[PlanStep]] = None
    explicit_conversions: Optional[List['ConversionLayer']] = None

    # The ``steps`` property recovers the final plan lazily using the
    # backpointers contained in ``table``.  If ``explicit_steps`` is provided
    # (e.g., after refinement passes) the stored sequence is returned
    # directly without consulting the DP table.
    @property
    def steps(self) -> List[PlanStep]:
        if self.explicit_steps is not None:
            return self.explicit_steps
        return self.recover()

    @property
    def conversions(self) -> List['ConversionLayer']:
        """Conversion layers associated with the plan.

        When planning populates conversions explicitly (``explicit_conversions``)
        they are returned directly.  Otherwise an empty list is provided as the
        planner lacks sufficient context to recover conversion information from
        the DP table alone.
        """

        if self.explicit_conversions is not None:
            return self.explicit_conversions
        return []

    def recover(
        self,
        index: Optional[int] = None,
        backend: Optional[Backend] = None,
    ) -> List[PlanStep]:
        """Recover a plan by following backpointers.

        Parameters
        ----------
        index, backend:
            DP cell to start from.  If omitted, the routine follows the
            backpointers of the optimal solution (i.e., from the final table
            entry).
        """

        if self.explicit_steps is not None:
            return self.explicit_steps
        if not self.table:
            return []
        if index is None:
            index = len(self.table) - 1
        if backend is None:
            backend = self.final_backend
        steps: List[PlanStep] = []
        i = index
        b = backend
        part = Partitioner()
        while i > 0 and b is not None:
            entry = self.table[i][b]
            segment = self.gates[entry.prev_index:i]
            groups = part.parallel_groups(segment)
            qubits = tuple(g[0] for g in groups) if groups else ()
            steps.append(
                PlanStep(
                    start=entry.prev_index,
                    end=i,
                    backend=b,
                    parallel=qubits,
                )
            )
            i = entry.prev_index
            b = entry.prev_backend
        steps.reverse()
        return steps


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _add_cost(a: Cost, b: Cost) -> Cost:
    """Combine two cost estimates sequentially.

    Runtime costs add up whereas memory requirements are assumed to be
    dominated by the larger of the two contributions.
    """

    return Cost(
        time=a.time + b.time,
        memory=max(a.memory, b.memory),
        log_depth=max(a.log_depth, b.log_depth),
    )


def _better(a: Cost, b: Cost) -> bool:
    """Return ``True`` if cost ``a`` is preferable over ``b``."""

    return (a.time, a.memory) < (b.time, b.memory)


def _supported_backends(gates: Iterable[Gate]) -> List[Backend]:
    """Determine which backends can simulate a gate sequence.

    The function now exposes general-purpose backends even for Clifford-only
    circuits while still listing :class:`Backend.TABLEAU` first to indicate the
    preferred specialised simulator.
    """

    gates = list(gates)
    names = [g.gate.upper() for g in gates]
    num_gates = len(gates)
    qubits = {q for g in gates for q in g.qubits}
    num_qubits = len(qubits)

    candidates: List[Backend] = []

    clifford = names and all(name in CLIFFORD_GATES for name in names)
    if clifford:
        candidates.append(Backend.TABLEAU)

    if num_qubits < 20:
        candidates.append(Backend.STATEVECTOR)
        return candidates

    multi = [g for g in gates if len(g.qubits) > 1]
    local = multi and all(
        len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi
    )

    if num_gates <= 2 ** num_qubits and not local:
        candidates.append(Backend.DECISION_DIAGRAM)
    if local:
        candidates.append(Backend.MPS)
    candidates.append(Backend.STATEVECTOR)

    return candidates


def _simulation_cost(
    estimator: CostEstimator, backend: Backend, num_qubits: int, num_gates: int
) -> Cost:
    """Query the cost estimator for a simulation fragment."""

    if backend == Backend.TABLEAU:
        return estimator.tableau(num_qubits, num_gates)
    if backend == Backend.MPS:
        return estimator.mps(num_qubits, num_gates, chi=4)
    if backend == Backend.DECISION_DIAGRAM:
        return estimator.decision_diagram(num_gates=num_gates, frontier=num_qubits)
    return estimator.statevector(num_qubits, num_gates)


# ---------------------------------------------------------------------------
# Planner implementation
# ---------------------------------------------------------------------------


class Planner:
    """Plan optimal backend assignments using dynamic programming."""

    def __init__(
        self,
        estimator: CostEstimator | None = None,
        *,
        top_k: int = 4,
        batch_size: int = 1,
        max_memory: float | None = None,
        quick_max_qubits: int | None = config.DEFAULT.quick_max_qubits,
        quick_max_gates: int | None = config.DEFAULT.quick_max_gates,
        quick_max_depth: int | None = config.DEFAULT.quick_max_depth,
        backend_order: Optional[List[Backend]] = None,
        conversion_cost_multiplier: float = 1.0,
    ):
        """Create a new planner instance.

        Parameters
        ----------
        estimator:
            Optional cost estimator instance.
        top_k:
            Number of candidate backends to retain per DP cell.
        batch_size:
            Gate count granularity during the initial coarse planning pass.
        max_memory:
            Global memory limit in bytes for simulation estimates.
        quick_max_qubits:
            If not ``None`` and the circuit spans at most this many qubits,
            a direct single-backend estimate is used.  Defaults to
            ``config.DEFAULT.quick_max_qubits``.
        quick_max_gates:
            If not ``None`` and the circuit contains at most this many gates,
            a direct single-backend estimate is used.  Defaults to
            ``config.DEFAULT.quick_max_gates``.
        quick_max_depth:
            If not ``None`` and the circuit depth does not exceed this value,
            a direct single-backend estimate is used.  Defaults to
            ``config.DEFAULT.quick_max_depth``.
        backend_order:
            Optional ordering of :class:`Backend` values to prefer when costs
            are equal.  Defaults to the configuration's preferred order.
        conversion_cost_multiplier:
            Factor applied to conversion time estimates.  Values greater than
            one discourage backend switches while values below one encourage
            them.  Defaults to ``1.0``.
        """

        self.estimator = estimator or CostEstimator()
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.quick_max_qubits = quick_max_qubits
        self.quick_max_gates = quick_max_gates
        self.quick_max_depth = quick_max_depth
        self.backend_order = list(backend_order) if backend_order is not None else list(config.DEFAULT.preferred_backend_order)
        self.conversion_cost_multiplier = conversion_cost_multiplier
        # Cache mapping gate fingerprints to ``PlanResult`` objects.
        # The cache allows reusing planning results for repeated gate
        # sequences which can occur when subcircuits are analysed multiple
        # times during scheduling.
        self.cache: Dict[tuple[Hashable, Backend | None], PlanResult] = {}
        self.cache_hits = 0

    # ------------------------------------------------------------------
    def _backend_rank(self, backend: Backend) -> int:
        try:
            return self.backend_order.index(backend)
        except ValueError:
            return len(self.backend_order)

    def _order_backends(self, backends: List[Backend]) -> List[Backend]:
        return sorted(backends, key=lambda b: (b != Backend.TABLEAU, self._backend_rank(b)))

    # ------------------------------------------------------------------
    def _dp(
        self,
        gates: List["Gate"],
        *,
        initial_backend: Optional[Backend] = None,
        target_backend: Optional[Backend] = None,
        batch_size: int = 1,
        max_memory: float | None = None,
        forced_backend: Backend | None = None,
    ) -> PlanResult:
        """Internal DP routine supporting batching and pruning.

        When ``forced_backend`` is provided only that backend is considered
        during planning.  A ``ValueError`` is raised if the backend cannot
        simulate a segment of the circuit.
        """

        n = len(gates)
        if n == 0:
            init = initial_backend if initial_backend is not None else None
            table = [{init: DPEntry(cost=Cost(0.0, 0.0), prev_index=0, prev_backend=None)}]
            return PlanResult(table=table, final_backend=init, gates=gates)

        # Pre-compute prefix and future qubit sets to derive boundary sizes.
        prefix_qubits: List[Set[int]] = [set() for _ in range(n + 1)]
        running: Set[int] = set()
        for i, gate in enumerate(gates, start=1):
            running |= set(gate.qubits)
            prefix_qubits[i] = running.copy()

        future_qubits: List[Set[int]] = [set() for _ in range(n + 1)]
        running.clear()
        for idx in range(n - 1, -1, -1):
            running |= set(gates[idx].qubits)
            future_qubits[idx] = running.copy()

        boundaries = [prefix_qubits[i] & future_qubits[i] for i in range(n + 1)]

        table: List[Dict[Optional[Backend], DPEntry]] = [dict() for _ in range(n + 1)]
        start_backend = initial_backend if initial_backend is not None else None
        table[0][start_backend] = DPEntry(cost=Cost(0.0, 0.0), prev_index=0, prev_backend=None)

        indices = list(range(0, n, batch_size)) + [n]

        for idx_i in range(1, len(indices)):
            i = indices[idx_i]
            for idx_j in range(idx_i):
                j = indices[idx_j]
                segment = gates[j:i]
                qubits = {q for g in segment for q in g.qubits}
                num_qubits = len(qubits)
                num_gates = i - j
                backends = self._order_backends(_supported_backends(segment))
                if forced_backend is not None:
                    if forced_backend not in backends:
                        raise ValueError(
                            f"Backend {forced_backend} unsupported for given circuit segment"
                        )
                    backends = [forced_backend]
                candidates: List[Tuple[Backend, Cost]] = []
                for backend in backends:
                    cost = _simulation_cost(self.estimator, backend, num_qubits, num_gates)
                    if max_memory is not None and cost.memory > max_memory:
                        continue
                    candidates.append((backend, cost))
                if not candidates:
                    candidates = [
                        (backend, _simulation_cost(self.estimator, backend, num_qubits, num_gates))
                        for backend in backends
                    ]
                for backend, sim_cost in candidates:
                    for prev_backend, prev_entry in table[j].items():
                        conv_cost = Cost(0.0, 0.0)
                        if prev_backend is not None and prev_backend != backend:
                            boundary = boundaries[j]
                            if boundary:
                                rank = 2 ** len(boundary)
                                frontier = len(boundary)
                                conv_est = self.estimator.conversion(
                                    prev_backend,
                                    backend,
                                    num_qubits=len(boundary),
                                    rank=rank,
                                    frontier=frontier,
                                )
                                conv_cost = Cost(
                                    time=conv_est.cost.time * self.conversion_cost_multiplier,
                                    memory=conv_est.cost.memory,
                                )
                                if max_memory is not None and conv_cost.memory > max_memory:
                                    continue
                        total_cost = _add_cost(_add_cost(prev_entry.cost, conv_cost), sim_cost)
                        entry = table[i].get(backend)
                        if entry is None or _better(total_cost, entry.cost):
                            table[i][backend] = DPEntry(
                                cost=total_cost,
                                prev_index=j,
                                prev_backend=prev_backend,
                            )
            if self.top_k and len(table[i]) > self.top_k:
                best = sorted(table[i].items(), key=lambda kv: kv[1].cost.time)[: self.top_k]
                table[i] = dict(best)

        final_entries = table[n]
        backend: Optional[Backend] = None
        if target_backend is not None:
            if target_backend in final_entries:
                backend = target_backend
        elif final_entries:
            backend = min(final_entries.items(), key=lambda kv: kv[1].cost.time)[0]

        return PlanResult(table=table, final_backend=backend, gates=gates)

    # ------------------------------------------------------------------
    def _conversions_for_steps(
        self, gates: List["Gate"], steps: List[PlanStep]
    ) -> List[ConversionLayer]:
        """Derive conversion layers for a sequence of plan steps."""

        if len(steps) <= 1:
            return []

        n = len(gates)
        prefix_qubits: List[Set[int]] = [set() for _ in range(n + 1)]
        running: Set[int] = set()
        for i, gate in enumerate(gates, start=1):
            running |= set(gate.qubits)
            prefix_qubits[i] = running.copy()

        future_qubits: List[Set[int]] = [set() for _ in range(n + 1)]
        running.clear()
        for idx in range(n - 1, -1, -1):
            running |= set(gates[idx].qubits)
            future_qubits[idx] = running.copy()

        layers: List[ConversionLayer] = []
        for prev, step in zip(steps, steps[1:]):
            if prev.backend == step.backend:
                continue
            cut = step.start
            boundary = sorted(prefix_qubits[cut] & future_qubits[cut])
            if not boundary:
                continue
            rank = 2 ** len(boundary)
            frontier = len(boundary)
            conv_est = self.estimator.conversion(
                prev.backend,
                step.backend,
                num_qubits=len(boundary),
                rank=rank,
                frontier=frontier,
            )
            layers.append(
                ConversionLayer(
                    boundary=tuple(boundary),
                    source=prev.backend,
                    target=step.backend,
                    rank=rank,
                    frontier=frontier,
                    primitive=conv_est.primitive,
                    cost=conv_est.cost,
                )
            )
        return layers

    # ------------------------------------------------------------------
    def _fingerprint(self, gates: List["Gate"]) -> Hashable:
        """Create an immutable fingerprint for a sequence of gates."""

        fp: List[Hashable] = []
        for g in gates:
            params = tuple(sorted(g.params.items()))
            fp.append((g.gate, tuple(g.qubits), params))
        return tuple(fp)

    def cache_lookup(self, gates: List["Gate"], backend: Backend | None = None) -> Optional[PlanResult]:
        """Return a cached plan for ``gates`` if available.

        Parameters
        ----------
        gates:
            Sequence of gates to plan.
        backend:
            Optional backend restriction.  The cache key incorporates the
            backend so that plans for different backends do not collide.
        """

        key = (self._fingerprint(gates), backend)
        result = self.cache.get(key)
        if result is not None:
            self.cache_hits += 1
        return result

    def cache_insert(
        self, gates: List["Gate"], result: PlanResult, backend: Backend | None = None
    ) -> None:
        """Insert ``result`` into the planning cache."""

        key = (self._fingerprint(gates), backend)
        self.cache[key] = result

    def _single_backend(self, gates: List["Gate"], max_memory: float | None) -> Tuple[Backend, Cost]:
        """Return best single-backend estimate for the full gate list."""

        qubits = {q for g in gates for q in g.qubits}
        num_qubits = len(qubits)
        num_gates = len(gates)
        backends = self._order_backends(_supported_backends(gates))
        candidates: List[Tuple[Backend, Cost]] = []
        for backend in backends:
            cost = _simulation_cost(self.estimator, backend, num_qubits, num_gates)
            if max_memory is not None and cost.memory > max_memory:
                continue
            candidates.append((backend, cost))
        if not candidates:
            candidates = [
                (backend, _simulation_cost(self.estimator, backend, num_qubits, num_gates))
                for backend in backends
            ]
        return min(candidates, key=lambda kv: (kv[1].time, kv[1].memory))

    def plan(
        self,
        circuit: Circuit,
        *,
        use_cache: bool = True,
        max_memory: float | None = None,
        backend: Backend | None = None,
    ) -> PlanResult:
        """Compute the optimal contiguous partition plan using optional
        coarse and refinement passes.

        The planner stores previously computed plans in an in-memory cache
        keyed by the circuit's gate sequence.  Subsequent calls for an
        identical gate sequence will reuse the cached result.  When a
        ``backend`` is supplied the entire circuit is executed on that
        backend, provided it can simulate the gates (only the Tableau backend
        is restricted to Clifford circuits).
        """

        gates = circuit.gates

        threshold = max_memory if max_memory is not None else self.max_memory

        if use_cache:
            cached = self.cache_lookup(gates, backend)
            if cached is not None:
                circuit.ssd.conversions = list(cached.conversions)
                return cached

        num_qubits = circuit.num_qubits
        num_gates = circuit.num_gates
        depth = circuit.depth

        if backend is not None:
            # Allow explicitly requested backends as long as they can simulate the
            # circuit.  Only Tableau has restrictions (it requires a Clifford
            # circuit).
            if backend == Backend.TABLEAU:
                names = [g.gate.upper() for g in gates]
                if names and not all(name in CLIFFORD_GATES for name in names):
                    raise ValueError(f"Backend {backend} unsupported for given circuit")
            cost = _simulation_cost(self.estimator, backend, num_qubits, num_gates)
            if threshold is not None and cost.memory > threshold:
                raise ValueError("Requested backend exceeds memory threshold")
            part = Partitioner()
            groups = part.parallel_groups(gates)
            parallel = tuple(g[0] for g in groups) if groups else ()
            step = PlanStep(start=0, end=len(gates), backend=backend, parallel=parallel)
            conversions = self._conversions_for_steps(gates, [step])
            circuit.ssd.conversions = list(conversions)
            result = PlanResult(
                table=[],
                final_backend=backend,
                gates=gates,
                explicit_steps=[step],
                explicit_conversions=conversions,
            )
            if use_cache:
                self.cache_insert(gates, result, backend)
            return result

        quick = True
        if self.quick_max_qubits is not None and num_qubits > self.quick_max_qubits:
            quick = False
        if self.quick_max_gates is not None and num_gates > self.quick_max_gates:
            quick = False
        if self.quick_max_depth is not None and depth > self.quick_max_depth:
            quick = False
        if quick and any(
            t is not None
            for t in (self.quick_max_qubits, self.quick_max_gates, self.quick_max_depth)
        ):
            backend_choice, cost = self._single_backend(gates, threshold)
            if threshold is None or cost.memory <= threshold:
                part = Partitioner()
                groups = part.parallel_groups(gates)
                parallel = tuple(g[0] for g in groups) if groups else ()
                step = PlanStep(start=0, end=len(gates), backend=backend_choice, parallel=parallel)
                conversions = self._conversions_for_steps(gates, [step])
                circuit.ssd.conversions = list(conversions)
                result = PlanResult(
                    table=[],
                    final_backend=backend_choice,
                    gates=gates,
                    explicit_steps=[step],
                    explicit_conversions=conversions,
                )
                if use_cache:
                    self.cache_insert(gates, result, backend_choice)
                return result

        if threshold is not None and gates:
            backend_choice, cost = self._single_backend(gates, threshold)
            if cost.memory <= threshold:
                part = Partitioner()
                groups = part.parallel_groups(gates)
                parallel = tuple(g[0] for g in groups) if groups else ()
                step = PlanStep(start=0, end=len(gates), backend=backend_choice, parallel=parallel)
                conversions = self._conversions_for_steps(gates, [step])
                circuit.ssd.conversions = list(conversions)
                result = PlanResult(
                    table=[],
                    final_backend=backend_choice,
                    gates=gates,
                    explicit_steps=[step],
                    explicit_conversions=conversions,
                )
                if use_cache:
                    self.cache_insert(gates, result, backend_choice)
                return result

        # First perform a coarse plan using the configured batch size.
        coarse = self._dp(gates, batch_size=self.batch_size, max_memory=threshold)

        # If no batching was requested we are done.
        if self.batch_size == 1:
            steps = list(coarse.steps)
            conversions = self._conversions_for_steps(gates, steps)
            circuit.ssd.conversions = list(conversions)
            coarse.explicit_steps = steps
            coarse.explicit_conversions = conversions
            if use_cache:
                self.cache_insert(gates, coarse)
            return coarse

        # Refine each coarse segment individually.
        refined_steps: List[PlanStep] = []
        prev_backend: Optional[Backend] = None
        for step in coarse.steps:
            segment = gates[step.start:step.end]
            sub = self._dp(
                segment,
                initial_backend=prev_backend,
                target_backend=step.backend,
                batch_size=1,
                max_memory=threshold,
            )
            for sub_step in sub.steps:
                refined_steps.append(
                    PlanStep(
                        start=sub_step.start + step.start,
                        end=sub_step.end + step.start,
                        backend=sub_step.backend,
                        parallel=sub_step.parallel,
                    )
                )
            prev_backend = step.backend

        final_backend = refined_steps[-1].backend if refined_steps else None
        conversions = self._conversions_for_steps(gates, refined_steps)
        circuit.ssd.conversions = list(conversions)
        result = PlanResult(
            table=[],
            final_backend=final_backend,
            gates=gates,
            explicit_steps=refined_steps,
            explicit_conversions=conversions,
        )
        if use_cache:
            self.cache_insert(gates, result)
        return result


__all__ = ["Planner", "PlanResult", "PlanStep", "DPEntry"]
