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
from typing import Dict, List, Optional, Iterable, Set, Tuple

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
    """Return value of :meth:`Planner.plan`.

    Attributes
    ----------
    table:
        Full DP table.  ``table[i][b]`` contains the best known cost to
        simulate the first ``i`` gates ending with backend ``b``.  Each entry
        also stores a backpointer for plan reconstruction.
    final_backend:
        Backend used for the last fragment in the optimal plan.
    """

    table: List[Dict[Optional[Backend], DPEntry]]
    final_backend: Optional[Backend]
    gates: List['Gate']

    # The ``steps`` property recovers the final plan lazily using the
    # backpointers contained in ``table``.
    @property
    def steps(self) -> List[PlanStep]:
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

    def __init__(self, estimator: CostEstimator | None = None):
        self.estimator = estimator or CostEstimator()

    # ------------------------------------------------------------------
    def plan(self, circuit: Circuit) -> PlanResult:
        """Compute the optimal contiguous partition plan.

        Parameters
        ----------
        circuit:
            Input circuit to analyse.

        Returns
        -------
        :class:`PlanResult`
            Object containing the DP table and convenience methods to recover
            the chosen plan.
        """

        gates = circuit.gates
        n = len(gates)
        if n == 0:
            return PlanResult(
                table=[{None: DPEntry(cost=Cost(0, 0), prev_index=0, prev_backend=None)}],
                final_backend=None,
                gates=[],
            )

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

        # DP table initialisation.  The entry at position 0 represents an
        # empty prefix using ``None`` as a pseudo backend.
        table: List[Dict[Optional[Backend], DPEntry]] = [
            {None: DPEntry(cost=Cost(0.0, 0.0), prev_index=0, prev_backend=None)}
        ] + [dict() for _ in range(n)]

        # Fill DP table -------------------------------------------------
        for i in range(1, n + 1):
            for j in range(i):
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
                                rank = min(2 ** len(boundary), 2 ** 8)
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

        # Select best terminal backend ---------------------------------
        final_entries = table[n]
        backend: Optional[Backend] = None
        if final_entries:
            backend = min(final_entries.items(), key=lambda kv: kv[1].cost.time)[0]

        return PlanResult(table=table, final_backend=backend, gates=gates)


__all__ = ["Planner", "PlanResult", "PlanStep", "DPEntry"]
