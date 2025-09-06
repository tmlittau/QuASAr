from __future__ import annotations

"""Dynamic programming planner for contiguous circuit partitions.

This module implements the algorithm described in the QuASAr draft for
optimally assigning simulation backends to contiguous circuit fragments.  The
planner evaluates all possible cut positions and backend choices using a
simple dynamic programming (DP) approach.  Each DP table entry stores the
cumulative cost up to a given gate index and acts as a backpointer to recover
an optimal execution plan.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Set, Tuple, Hashable

from .cost import Backend, Cost, CostEstimator
from .partitioner import CLIFFORD_GATES, Partitioner
from .ssd import ConversionLayer, SSD
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
    gates: List["Gate"]
    explicit_steps: Optional[List[PlanStep]] = None
    explicit_conversions: Optional[List["ConversionLayer"]] = None
    step_costs: Optional[List[Cost]] = None
    replay_ssd: Dict[int, "SSD"] = field(default_factory=dict)

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
    def conversions(self) -> List["ConversionLayer"]:
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
            segment = self.gates[entry.prev_index : i]
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
        conversion=a.conversion + b.conversion,
        replay=a.replay + b.replay,
    )


def _better(a: Cost, b: Cost) -> bool:
    """Return ``True`` if cost ``a`` is preferable over ``b``."""

    return (a.time, a.memory) < (b.time, b.memory)


def _supported_backends(
    gates: Iterable[Gate],
    *,
    sparsity: float | None = None,
    circuit: "Circuit" | None = None,
    phase_rotation_diversity: int | None = None,
    amplitude_rotation_diversity: int | None = None,
    allow_tableau: bool = True,
    estimator: CostEstimator | None = None,
    max_memory: float | None = None,
) -> List[Backend]:
    """Determine which backends can simulate a gate sequence.

    Parameters
    ----------
    gates:
        Gate sequence under consideration.
    sparsity:
        Optional sparsity metric for the overall circuit.
    circuit:
        Circuit providing heuristic metrics.  Explicit ``sparsity`` and
        rotation-diversity arguments take precedence when supplied.
    allow_tableau:
        If ``True`` and the gate sequence is Clifford-only, include
        :class:`Backend.TABLEAU` as a candidate.  When ``False`` the tableau
        backend is never proposed even if the segment itself is Clifford.  This
        allows callers to disable specialised Clifford handling when the
        surrounding circuit contains non-Clifford operations.
    estimator:
        Optional cost estimator providing an MPS bond dimension via
        ``estimator.chi_max``.
    max_memory:
        Optional memory threshold used to exclude backends that exceed the
        limit for the estimated cost.
    """

    if circuit is not None:
        if sparsity is None:
            sparsity = getattr(circuit, "sparsity", None)
        if phase_rotation_diversity is None:
            phase_rotation_diversity = getattr(circuit, "phase_rotation_diversity", None)
        if amplitude_rotation_diversity is None:
            amplitude_rotation_diversity = getattr(circuit, "amplitude_rotation_diversity", None)

    gates = list(gates)
    names = [g.gate.upper() for g in gates]
    num_gates = len(gates)
    qubits = {q for g in gates for q in g.qubits}
    num_qubits = (max(qubits) + 1) if qubits else 0

    clifford = names and all(name in CLIFFORD_GATES for name in names)
    if allow_tableau and clifford:
        return [Backend.TABLEAU]

    candidates: List[Backend] = []

    sparse = sparsity if sparsity is not None else 0.0
    phase_rot = phase_rotation_diversity if phase_rotation_diversity is not None else 0
    amp_rot = amplitude_rotation_diversity if amplitude_rotation_diversity is not None else 0
    nnz = int((1 - sparse) * (2 ** num_qubits))
    multi = [g for g in gates if len(g.qubits) > 1]
    local = bool(multi) and all(
        len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi
    )
    from .sparsity import adaptive_dd_sparsity_threshold

    s_thresh = adaptive_dd_sparsity_threshold(num_qubits)
    passes = (
        sparse >= s_thresh
        and nnz <= config.DEFAULT.dd_nnz_threshold
        and phase_rot <= config.DEFAULT.dd_phase_rotation_diversity_threshold
        and amp_rot <= config.DEFAULT.dd_amplitude_rotation_diversity_threshold
    )
    dd_metric = False
    if passes:
        s_score = sparse / s_thresh if s_thresh > 0 else 0.0
        nnz_score = 1 - nnz / config.DEFAULT.dd_nnz_threshold
        phase_score = 1 - phase_rot / config.DEFAULT.dd_phase_rotation_diversity_threshold
        amp_score = 1 - amp_rot / config.DEFAULT.dd_amplitude_rotation_diversity_threshold
        weight_sum = (
            config.DEFAULT.dd_sparsity_weight
            + config.DEFAULT.dd_nnz_weight
            + config.DEFAULT.dd_phase_rotation_weight
            + config.DEFAULT.dd_amplitude_rotation_weight
        )
        weighted = (
            config.DEFAULT.dd_sparsity_weight * s_score
            + config.DEFAULT.dd_nnz_weight * nnz_score
            + config.DEFAULT.dd_phase_rotation_weight * phase_score
            + config.DEFAULT.dd_amplitude_rotation_weight * amp_score
        )
        metric = weighted / weight_sum if weight_sum else 0.0
        dd_metric = metric >= config.DEFAULT.dd_metric_threshold

    mps_metric = False
    if estimator is not None and all(len(g.qubits) <= 2 for g in gates):
        chi_cap = estimator.chi_max
        if chi_cap is not None and chi_cap >= 1:
            num_meas = sum(
                1 for g in gates if len(g.qubits) == 1 and g.gate.upper() in {"MEASURE", "RESET"}
            )
            num_1q = sum(
                1
                for g in gates
                if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
            )
            num_2q = num_gates - num_1q - num_meas
            cost = estimator.mps(num_qubits, num_1q + num_meas, num_2q, chi_cap)
            if max_memory is None or cost.memory <= max_memory:
                mps_metric = True

    if dd_metric:
        candidates.append(Backend.DECISION_DIAGRAM)
    if mps_metric:
        candidates.append(Backend.MPS)
    candidates.append(Backend.STATEVECTOR)

    order = list(config.DEFAULT.preferred_backend_order)

    def rank(b: Backend) -> int:
        try:
            idx = order.index(b)
        except ValueError:
            idx = len(order)
        if dd_metric and b == Backend.DECISION_DIAGRAM:
            return -1
        return idx

    ranking = sorted(candidates, key=lambda b: (b != Backend.TABLEAU, rank(b)))
    ranking_str = ">".join(b.name for b in ranking)

    if config.DEFAULT.verbose_selection:
        print(
            "[backend-selection] "
            f"sparsity={sparse:.6f} rotation_diversity={rot:.6f} nnz={nnz} "
            f"locality={local} candidates={ranking_str}"
        )

    if config.DEFAULT.backend_selection_log:
        try:
            with open(config.DEFAULT.backend_selection_log, "a", encoding="utf8") as f:
                f.write(
                    f"{sparse:.6f},{nnz},{rot:.6f},{int(local)},{ranking_str}\n"
                )
        except OSError:
            pass

    return candidates


def _circuit_depth(gates: Iterable["Gate"]) -> int:
    """Return the logical depth of ``gates``.

    This is a lightweight helper mirroring :meth:`Circuit._compute_depth`
    without requiring a full :class:`Circuit` instance.
    """

    qubit_levels: Dict[int, int] = {}
    depth = 0
    for gate in gates:
        start = max((qubit_levels.get(q, 0) for q in gate.qubits), default=0)
        level = start + 1
        for q in gate.qubits:
            qubit_levels[q] = level
        if level > depth:
            depth = level
    return depth


def _simulation_cost(
    estimator: CostEstimator,
    backend: Backend,
    num_qubits: int,
    num_1q_gates: int,
    num_2q_gates: int,
    num_meas: int,
) -> Cost:
    """Query the cost estimator for a simulation fragment."""

    num_gates = num_1q_gates + num_2q_gates + num_meas
    if backend == Backend.TABLEAU:
        return estimator.tableau(num_qubits, num_gates)
    if backend == Backend.MPS:
        return estimator.mps(
            num_qubits,
            num_1q_gates + num_meas,
            num_2q_gates,
            chi=4,
            svd=True,
        )
    if backend == Backend.DECISION_DIAGRAM:
        return estimator.decision_diagram(num_gates=num_gates, frontier=num_qubits)
    return estimator.statevector(num_qubits, num_1q_gates, num_2q_gates, num_meas)


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
        self.backend_order = (
            list(backend_order)
            if backend_order is not None
            else list(config.DEFAULT.preferred_backend_order)
        )
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

    def _order_backends(
        self, backends: List[Backend], *, dd_metric: bool = False
    ) -> List[Backend]:
        def rank(b: Backend) -> int:
            if dd_metric and b == Backend.DECISION_DIAGRAM:
                return -1
            return self._backend_rank(b)

        return sorted(backends, key=lambda b: (b != Backend.TABLEAU, rank(b)))

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
        allow_tableau: bool = True,
        symmetry: float | None = None,
        sparsity: float | None = None,
        phase_rotation_diversity: int | None = None,
        amplitude_rotation_diversity: int | None = None,
    ) -> PlanResult:
        """Internal DP routine supporting batching and pruning.

        When ``forced_backend`` is provided only that backend is considered
        during planning.  A ``ValueError`` is raised if the backend cannot
        simulate a segment of the circuit.  ``sparsity`` and rotation metrics
        are forwarded to :func:`_supported_backends`.
        """
        from .sparsity import adaptive_dd_sparsity_threshold
        nnz_estimate = None
        if sparsity is not None:
            nnz_estimate = int((1 - sparsity) * (2 ** len({q for g in gates for q in g.qubits})))
        s_thresh = adaptive_dd_sparsity_threshold(len({q for g in gates for q in g.qubits}))
        passes = (
            (sparsity is not None and sparsity >= s_thresh)
            and (
                nnz_estimate is not None and nnz_estimate <= config.DEFAULT.dd_nnz_threshold
            )
            and (
                (phase_rotation_diversity is None
                 or phase_rotation_diversity <= config.DEFAULT.dd_phase_rotation_diversity_threshold)
                and (
                    amplitude_rotation_diversity is None
                    or amplitude_rotation_diversity
                    <= config.DEFAULT.dd_amplitude_rotation_diversity_threshold
                )
            )
        )
        dd_metric = False
        if passes:
            s_score = sparsity / s_thresh if s_thresh > 0 else 0.0
            nnz_score = 1 - nnz_estimate / config.DEFAULT.dd_nnz_threshold
            phase_score = 1 - (
                (phase_rotation_diversity or 0)
                / config.DEFAULT.dd_phase_rotation_diversity_threshold
            )
            amp_score = 1 - (
                (amplitude_rotation_diversity or 0)
                / config.DEFAULT.dd_amplitude_rotation_diversity_threshold
            )
            weight_sum = (
                config.DEFAULT.dd_sparsity_weight
                + config.DEFAULT.dd_nnz_weight
                + config.DEFAULT.dd_phase_rotation_weight
                + config.DEFAULT.dd_amplitude_rotation_weight
            )
            weighted = (
                config.DEFAULT.dd_sparsity_weight * s_score
                + config.DEFAULT.dd_nnz_weight * nnz_score
                + config.DEFAULT.dd_phase_rotation_weight * phase_score
                + config.DEFAULT.dd_amplitude_rotation_weight * amp_score
            )
            metric = weighted / weight_sum if weight_sum else 0.0
            dd_metric = metric >= config.DEFAULT.dd_metric_threshold

        n = len(gates)
        if n == 0:
            init = initial_backend if initial_backend is not None else None
            table = [
                {init: DPEntry(cost=Cost(0.0, 0.0), prev_index=0, prev_backend=None)}
            ]
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
        table[0][start_backend] = DPEntry(
            cost=Cost(0.0, 0.0), prev_index=0, prev_backend=None
        )

        indices = list(range(0, n, batch_size)) + [n]

        for idx_i in range(1, len(indices)):
            i = indices[idx_i]
            for idx_j in range(idx_i):
                j = indices[idx_j]
                segment = gates[j:i]
                qubits = {q for g in segment for q in g.qubits}
                num_qubits = len(qubits)
                num_gates = i - j
                num_meas = sum(
                    1 for g in segment if g.gate.upper() in {"MEASURE", "RESET"}
                )
                num_1q = sum(
                    1
                    for g in segment
                    if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
                )
                num_2q = num_gates - num_1q - num_meas
                backends = self._order_backends(
                    _supported_backends(
                        segment,
                        sparsity=sparsity,
                        phase_rotation_diversity=phase_rotation_diversity,
                        amplitude_rotation_diversity=amplitude_rotation_diversity,
                        allow_tableau=allow_tableau,
                        estimator=self.estimator,
                        max_memory=max_memory,
                    ),
                    dd_metric=dd_metric,
                )
                if forced_backend is not None:
                    if forced_backend not in backends:
                        raise ValueError(
                            f"Backend {forced_backend} unsupported for given circuit segment"
                        )
                    backends = [forced_backend]
                candidates: List[Tuple[Backend, Cost]] = []
                for backend in backends:
                    cost = _simulation_cost(
                        self.estimator, backend, num_qubits, num_1q, num_2q, num_meas
                    )
                    if max_memory is not None and cost.memory > max_memory:
                        continue
                    candidates.append((backend, cost))
                if not candidates:
                    candidates = [
                        (
                            backend,
                            _simulation_cost(
                                self.estimator,
                                backend,
                                num_qubits,
                                num_1q,
                                num_2q,
                                num_meas,
                            ),
                        )
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
                                    time=conv_est.cost.time
                                    * self.conversion_cost_multiplier,
                                    memory=conv_est.cost.memory,
                                )
                                if (
                                    max_memory is not None
                                    and conv_cost.memory > max_memory
                                ):
                                    continue
                        total_cost = _add_cost(
                            _add_cost(prev_entry.cost, conv_cost), sim_cost
                        )
                        entry = table[i].get(backend)
                        if entry is None or _better(total_cost, entry.cost):
                            table[i][backend] = DPEntry(
                                cost=total_cost,
                                prev_index=j,
                                prev_backend=prev_backend,
                            )
            if self.top_k and len(table[i]) > self.top_k:
                best = sorted(table[i].items(), key=lambda kv: kv[1].cost.time)[
                    : self.top_k
                ]
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

    def cache_lookup(
        self, gates: List["Gate"], backend: Backend | None = None
    ) -> Optional[PlanResult]:
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

    def _single_backend(
        self,
        gates: List["Gate"],
        max_memory: float | None,
        *,
        sparsity: float | None = None,
        phase_rotation_diversity: int | None = None,
        amplitude_rotation_diversity: int | None = None,
        allow_tableau: bool = True,
    ) -> Tuple[Backend, Cost]:
        """Return best single-backend estimate for the full gate list.

        Parameters
        ----------
        gates:
            Gate sequence to estimate.
        max_memory:
            Optional memory threshold.
        symmetry, sparsity:
            Optional heuristic metrics for the overall circuit.
        phase_rotation_diversity, amplitude_rotation_diversity:
            Optional counts of distinct rotation angles used by the circuit.
        allow_tableau:
            Propagate the circuit-level Clifford check.  When ``False`` the
            tableau backend is never considered even if ``gates`` are
            individually Clifford.
        """

        qubits = {q for g in gates for q in g.qubits}
        num_qubits = len(qubits)
        num_gates = len(gates)
        num_meas = sum(1 for g in gates if g.gate.upper() in {"MEASURE", "RESET"})
        num_1q = sum(
            1
            for g in gates
            if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
        )
        num_2q = num_gates - num_1q - num_meas
        from .sparsity import adaptive_dd_sparsity_threshold
        nnz_estimate = int((1 - (sparsity or 0.0)) * (2 ** num_qubits))
        s_thresh = adaptive_dd_sparsity_threshold(num_qubits)
        passes = (
            (sparsity or 0.0) >= s_thresh
            and nnz_estimate <= config.DEFAULT.dd_nnz_threshold
            and (phase_rotation_diversity or 0)
            <= config.DEFAULT.dd_phase_rotation_diversity_threshold
            and (amplitude_rotation_diversity or 0)
            <= config.DEFAULT.dd_amplitude_rotation_diversity_threshold
        )
        dd_metric = False
        if passes:
            s_score = (sparsity or 0.0) / s_thresh if s_thresh > 0 else 0.0
            nnz_score = 1 - nnz_estimate / config.DEFAULT.dd_nnz_threshold
            phase_score = 1 - (
                (phase_rotation_diversity or 0)
                / config.DEFAULT.dd_phase_rotation_diversity_threshold
            )
            amp_score = 1 - (
                (amplitude_rotation_diversity or 0)
                / config.DEFAULT.dd_amplitude_rotation_diversity_threshold
            )
            weight_sum = (
                config.DEFAULT.dd_sparsity_weight
                + config.DEFAULT.dd_nnz_weight
                + config.DEFAULT.dd_phase_rotation_weight
                + config.DEFAULT.dd_amplitude_rotation_weight
            )
            weighted = (
                config.DEFAULT.dd_sparsity_weight * s_score
                + config.DEFAULT.dd_nnz_weight * nnz_score
                + config.DEFAULT.dd_phase_rotation_weight * phase_score
                + config.DEFAULT.dd_amplitude_rotation_weight * amp_score
            )
            metric = weighted / weight_sum if weight_sum else 0.0
            dd_metric = metric >= config.DEFAULT.dd_metric_threshold
        backends = self._order_backends(
            _supported_backends(
                gates,
                sparsity=sparsity,
                phase_rotation_diversity=phase_rotation_diversity,
                amplitude_rotation_diversity=amplitude_rotation_diversity,
                allow_tableau=allow_tableau,
                estimator=self.estimator,
                max_memory=max_memory,
            ),
            dd_metric=dd_metric,
        )
        candidates: List[Tuple[Backend, Cost]] = []
        for backend in backends:
            cost = _simulation_cost(
                self.estimator, backend, num_qubits, num_1q, num_2q, num_meas
            )
            if max_memory is not None and cost.memory > max_memory:
                continue
            candidates.append((backend, cost))
        if not candidates:
            candidates = [
                (
                    backend,
                    _simulation_cost(
                        self.estimator, backend, num_qubits, num_1q, num_2q, num_meas
                    ),
                )
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
        names = [g.gate.upper() for g in gates]
        clifford_circuit = bool(names) and all(name in CLIFFORD_GATES for name in names)
        allow_tableau = clifford_circuit

        threshold = max_memory if max_memory is not None else self.max_memory

        if use_cache:
            cached = self.cache_lookup(gates, backend)
            if cached is not None:
                circuit.ssd.conversions = list(cached.conversions)
                return cached

        num_qubits = circuit.num_qubits
        num_gates = circuit.num_gates
        depth = circuit.depth
        num_meas = sum(1 for g in gates if g.gate.upper() in {"MEASURE", "RESET"})
        num_1q = sum(
            1
            for g in gates
            if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
        )
        num_2q = num_gates - num_1q - num_meas

        if self.estimator is not None:
            fidelity = config.DEFAULT.mps_target_fidelity
            chi_cap = self.estimator.chi_for_constraints(
                num_qubits, gates, fidelity, threshold
            )
            self.estimator.chi_max = chi_cap if chi_cap > 0 else None

        if backend is not None:
            # Allow explicitly requested backends as long as they can simulate the
            # circuit.  Only Tableau has restrictions (it requires a Clifford
            # circuit).
            if backend == Backend.TABLEAU:
                if names and not all(name in CLIFFORD_GATES for name in names):
                    raise ValueError(f"Backend {backend} unsupported for given circuit")
            cost = _simulation_cost(
                self.estimator, backend, num_qubits, num_1q, num_2q, num_meas
            )
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
        # Pre-compute the cost of executing the full circuit on a single backend
        single_backend_choice, single_cost = self._single_backend(
            gates,
            threshold,
            sparsity=circuit.sparsity,
            phase_rotation_diversity=circuit.phase_rotation_diversity,
            amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
            allow_tableau=allow_tableau,
        )

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
            if threshold is None or single_cost.memory <= threshold:
                part = Partitioner()
                groups = part.parallel_groups(gates)
                parallel = tuple(g[0] for g in groups) if groups else ()
                step = PlanStep(
                    start=0,
                    end=len(gates),
                    backend=single_backend_choice,
                    parallel=parallel,
                )
                result = PlanResult(
                    table=[],
                    final_backend=single_backend_choice,
                    gates=gates,
                    explicit_steps=[step],
                    explicit_conversions=[],
                )
                circuit.ssd.conversions = []
                if use_cache:
                    self.cache_insert(gates, result, single_backend_choice)
                return result

        # Lightweight pre-pass comparing single backend to coarse partitioning
        pre_batch = max(1, len(gates) // 4)
        pre = self._dp(
            gates,
            batch_size=pre_batch,
            max_memory=threshold,
            allow_tableau=allow_tableau,
            sparsity=circuit.sparsity,
            phase_rotation_diversity=circuit.phase_rotation_diversity,
            amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
        )
        pre_cost = (
            pre.table[-1][pre.final_backend].cost if pre.table else Cost(0.0, 0.0)
        )
        overhead = Cost(time=len(gates) * 1e-6, memory=0.0)
        if _better(single_cost, _add_cost(pre_cost, overhead)) and (
            threshold is None or single_cost.memory <= threshold
        ):
            part = Partitioner()
            groups = part.parallel_groups(gates)
            parallel = tuple(g[0] for g in groups) if groups else ()
            step = PlanStep(
                start=0,
                end=len(gates),
                backend=single_backend_choice,
                parallel=parallel,
            )
            result = PlanResult(
                table=[],
                final_backend=single_backend_choice,
                gates=gates,
                explicit_steps=[step],
                explicit_conversions=[],
            )
            circuit.ssd.conversions = []
            if use_cache:
                self.cache_insert(gates, result, single_backend_choice)
            return result

        # Perform a coarse plan using the configured batch size for refinement
        coarse = self._dp(
            gates,
            batch_size=self.batch_size,
            max_memory=threshold,
            allow_tableau=allow_tableau,
            sparsity=circuit.sparsity,
            phase_rotation_diversity=circuit.phase_rotation_diversity,
            amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
        )

        dp_cost = (
            coarse.table[-1][coarse.final_backend].cost
            if coarse.table
            else Cost(0.0, 0.0)
        )

        if _better(single_cost, dp_cost) and (
            threshold is None or single_cost.memory <= threshold
        ):
            part = Partitioner()
            groups = part.parallel_groups(gates)
            parallel = tuple(g[0] for g in groups) if groups else ()
            step = PlanStep(
                start=0,
                end=len(gates),
                backend=single_backend_choice,
                parallel=parallel,
            )
            result = PlanResult(
                table=[],
                final_backend=single_backend_choice,
                gates=gates,
                explicit_steps=[step],
                explicit_conversions=[],
            )
            circuit.ssd.conversions = []
            if use_cache:
                self.cache_insert(gates, result, single_backend_choice)
            return result

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
            segment = gates[step.start : step.end]
            sub = self._dp(
                segment,
                initial_backend=prev_backend,
                target_backend=step.backend,
                batch_size=1,
                max_memory=threshold,
                allow_tableau=allow_tableau,
                sparsity=circuit.sparsity,
                phase_rotation_diversity=circuit.phase_rotation_diversity,
                amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
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
