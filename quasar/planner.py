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

    # The ``steps`` property recovers the final plan lazily using the
    # backpointers contained in ``table``.  If ``explicit_steps`` is provided
    # (e.g., after refinement passes) the stored sequence is returned
    # directly without consulting the DP table.
    @property
    def steps(self) -> List[PlanStep]:
        if self.explicit_steps is not None:
            return self.explicit_steps
        return self.recover()

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

    return Cost(time=a.time + b.time, memory=max(a.memory, b.memory))


def _better(a: Cost, b: Cost) -> bool:
    """Return ``True`` if cost ``a`` is preferable over ``b``."""

    return (a.time, a.memory) < (b.time, b.memory)


def _supported_backends(gates: Iterable[Gate]) -> List[Backend]:
    """Determine which backends can simulate a gate sequence."""

    gates = list(gates)
    names = [g.gate.upper() for g in gates]
    num_gates = len(gates)
    qubits = {q for g in gates for q in g.qubits}
    num_qubits = len(qubits)

    # --- Clifford only ---
    if names and all(name in CLIFFORD_GATES for name in names):
        return [Backend.TABLEAU]

    candidates: List[Backend] = []

    # --- Local multi-qubit gates for MPS ---
    multi = [g for g in gates if len(g.qubits) > 1]
    local = multi and all(
        len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi
    )
    if local:
        candidates.append(Backend.MPS)

    # --- Decision diagrams when gate count is small ---
    if num_gates <= 2 ** num_qubits:
        candidates.append(Backend.DECISION_DIAGRAM)

    # --- Statevector backend always available ---
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
    ):
        self.estimator = estimator or CostEstimator()
        self.top_k = top_k
        self.batch_size = batch_size
        # Cache mapping gate fingerprints to ``PlanResult`` objects.
        # The cache allows reusing planning results for repeated gate
        # sequences which can occur when subcircuits are analysed multiple
        # times during scheduling.
        self.cache: Dict[Hashable, PlanResult] = {}
        self.cache_hits = 0

    # ------------------------------------------------------------------
    def _dp(
        self,
        gates: List["Gate"],
        *,
        initial_backend: Optional[Backend] = None,
        target_backend: Optional[Backend] = None,
        batch_size: int = 1,
    ) -> PlanResult:
        """Internal DP routine supporting batching and pruning."""

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
                backends = _supported_backends(segment)
                for backend in backends:
                    sim_cost = _simulation_cost(self.estimator, backend, num_qubits, num_gates)
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
                                conv_cost = conv_est.cost
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
    def _fingerprint(self, gates: List["Gate"]) -> Hashable:
        """Create an immutable fingerprint for a sequence of gates."""

        fp: List[Hashable] = []
        for g in gates:
            params = tuple(sorted(g.params.items()))
            fp.append((g.gate, tuple(g.qubits), params))
        return tuple(fp)

    def cache_lookup(self, gates: List["Gate"]) -> Optional[PlanResult]:
        """Return a cached plan for ``gates`` if available."""

        key = self._fingerprint(gates)
        result = self.cache.get(key)
        if result is not None:
            self.cache_hits += 1
        return result

    def cache_insert(self, gates: List["Gate"], result: PlanResult) -> None:
        """Insert ``result`` into the planning cache."""

        key = self._fingerprint(gates)
        self.cache[key] = result

    def plan(self, circuit: Circuit, *, use_cache: bool = True) -> PlanResult:
        """Compute the optimal contiguous partition plan using optional
        coarse and refinement passes.

        The planner stores previously computed plans in an in-memory cache
        keyed by the circuit's gate sequence.  Subsequent calls for an
        identical gate sequence will reuse the cached result.
        """

        gates = circuit.gates

        if use_cache:
            cached = self.cache_lookup(gates)
            if cached is not None:
                return cached

        # First perform a coarse plan using the configured batch size.
        coarse = self._dp(gates, batch_size=self.batch_size)

        # If no batching was requested we are done.
        if self.batch_size == 1:
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
        result = PlanResult(table=[], final_backend=final_backend, gates=gates, explicit_steps=refined_steps)
        if use_cache:
            self.cache_insert(gates, result)
        return result


__all__ = ["Planner", "PlanResult", "PlanStep", "DPEntry"]
