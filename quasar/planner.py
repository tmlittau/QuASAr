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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Hashable, Iterator

from .cost import Backend, Cost, CostEstimator
from quasar_convert import ConversionEngine
from .partitioner import CLIFFORD_GATES, CLIFFORD_PLUS_T_GATES, Partitioner
from .ssd import ConversionLayer, SSD
from . import config
from .analyzer import AnalysisResult
from .method_selector import MethodSelector, NoFeasibleBackendError

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
    parallel: Tuple[Tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class GateSegmentView(Iterable["Gate"]):
    """Lightweight view over a contiguous slice of gates.

    The view stores only start and end indices into the original ``gates``
    sequence and therefore avoids creating temporary lists when iterated
    multiple times.
    """

    gates: List["Gate"]
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Invalid segment bounds")

    def __iter__(self) -> Iterator["Gate"]:
        for idx in range(self.start, self.end):
            yield self.gates[idx]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.end - self.start

    def qubits(self) -> Set[int]:
        qubits: Set[int] = set()
        for gate in self:
            qubits.update(gate.qubits)
        return qubits


@dataclass(frozen=True)
class SupportedBackendMetrics:
    """Pre-computed metrics describing a gate segment.

    The planner can forward these metrics to :func:`_supported_backends` to
    avoid recomputing basic statistics such as gate counts, locality and
    sparsity derived features.  All values mirror the information that the
    helper previously collected internally.
    """

    num_gates: int
    num_qubits: int
    num_meas: int
    num_1q: int
    num_2q: int
    num_t: int
    clifford: bool
    clifford_plus_t: bool
    has_multi_qubit: bool
    local_multi_qubit: bool
    two_qubit_only: bool
    sparsity: float | None = None
    phase_rotation_diversity: int | None = None
    amplitude_rotation_diversity: int | None = None

    @classmethod
    def from_gates(
        cls,
        gates: Iterable["Gate"],
        *,
        sparsity: float | None = None,
        phase_rotation_diversity: int | None = None,
        amplitude_rotation_diversity: int | None = None,
    ) -> "SupportedBackendMetrics":
        """Build metrics by inspecting ``gates`` directly."""

        gate_list = list(gates)
        names = [g.gate.upper() for g in gate_list]
        num_gates = len(gate_list)
        qubits = {q for g in gate_list for q in g.qubits}
        num_qubits = len(qubits)
        meas_ops = {"MEASURE", "RESET"}
        num_meas = sum(1 for name in names if name in meas_ops)
        num_1q = sum(
            1
            for gate, name in zip(gate_list, names)
            if len(gate.qubits) == 1 and name not in meas_ops
        )
        num_2q = num_gates - num_1q - num_meas
        num_t = sum(1 for name in names if name in {"T", "TDG"})
        multi = [g for g in gate_list if len(g.qubits) > 1]
        local = bool(multi) and all(
            len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi
        )
        two_qubit_only = all(len(g.qubits) <= 2 for g in gate_list)
        clifford = bool(names) and all(name in CLIFFORD_GATES for name in names)
        clifford_t = bool(names) and all(
            name in CLIFFORD_PLUS_T_GATES for name in names
        )
        return cls(
            num_gates=num_gates,
            num_qubits=num_qubits,
            num_meas=num_meas,
            num_1q=num_1q,
            num_2q=num_2q,
            num_t=num_t,
            clifford=clifford,
            clifford_plus_t=clifford_t,
            has_multi_qubit=bool(multi),
            local_multi_qubit=local,
            two_qubit_only=two_qubit_only,
            sparsity=sparsity,
            phase_rotation_diversity=phase_rotation_diversity,
            amplitude_rotation_diversity=amplitude_rotation_diversity,
        )

    @classmethod
    def from_gate(
        cls,
        gate: "Gate",
        *,
        sparsity: float | None = None,
        phase_rotation_diversity: int | None = None,
        amplitude_rotation_diversity: int | None = None,
    ) -> "SupportedBackendMetrics":
        """Return metrics for a single gate."""

        name = gate.gate.upper()
        qubits = tuple(gate.qubits)
        qubit_set = set(qubits)
        meas_ops = {"MEASURE", "RESET"}
        is_measure = name in meas_ops
        is_1q = len(qubits) == 1 and not is_measure
        num_meas = 1 if is_measure else 0
        num_1q = 1 if is_1q else 0
        num_2q = 1 - num_1q - num_meas
        has_multi = len(qubits) > 1 and not is_measure
        local_multi = (
            has_multi
            and len(qubits) == 2
            and abs(qubits[0] - qubits[1]) == 1
        )
        return cls(
            num_gates=1,
            num_qubits=len(qubit_set),
            num_meas=num_meas,
            num_1q=num_1q,
            num_2q=num_2q,
            num_t=1 if name in {"T", "TDG"} else 0,
            clifford=name in CLIFFORD_GATES,
            clifford_plus_t=name in CLIFFORD_PLUS_T_GATES,
            has_multi_qubit=has_multi,
            local_multi_qubit=local_multi,
            two_qubit_only=len(qubits) <= 2,
            sparsity=sparsity,
            phase_rotation_diversity=phase_rotation_diversity,
            amplitude_rotation_diversity=amplitude_rotation_diversity,
        )

@dataclass
class ConversionEstimate:
    """Diagnostic record of a conversion estimate between backends."""

    stage: str
    start: int
    end: int
    source: Optional[Backend]
    target: Backend
    boundary: Tuple[int, ...]
    cost: Cost
    primitive: str | None = None
    feasible: bool = True
    reason: str | None = None


@dataclass
class PlanDiagnostics:
    """Diagnostics collected while planning a circuit."""

    single_backend: Backend | None = None
    single_cost: Cost | None = None
    pre_cost: Cost | None = None
    pre_overhead: Cost | None = None
    dp_cost: Cost | None = None
    refined_cost: Cost | None = None
    strategy: str | None = None
    conversion_estimates: List[ConversionEstimate] = field(default_factory=list)
    backend_selection: Dict[str, dict[str, Any]] = field(default_factory=dict)

    def record_conversion(
        self,
        *,
        stage: str,
        start: int,
        end: int,
        source: Backend | None,
        target: Backend,
        boundary: Iterable[int],
        cost: Cost,
        primitive: str | None = None,
        feasible: bool = True,
        reason: str | None = None,
    ) -> None:
        self.conversion_estimates.append(
            ConversionEstimate(
                stage=stage,
                start=start,
                end=end,
                source=source,
                target=target,
                boundary=tuple(sorted(boundary)),
                cost=cost,
                primitive=primitive,
                feasible=feasible,
                reason=reason,
            )
        )


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
    analysis: AnalysisResult | None = None
    diagnostics: PlanDiagnostics | None = None

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
        part: Partitioner | None = None
        while i > 0 and b is not None:
            entry = self.table[i][b]
            parallel_groups: Tuple[Tuple[int, ...], ...] | None = getattr(
                entry, "parallel", None
            )
            if parallel_groups is None:
                if part is None:
                    part = Partitioner()
                segment = self.gates[entry.prev_index : i]
                groups = part.parallel_groups(segment)
                parallel_groups = tuple(g[0] for g in groups) if groups else ()
            steps.append(
                PlanStep(
                    start=entry.prev_index,
                    end=i,
                    backend=b,
                    parallel=parallel_groups,
                )
            )
            i = entry.prev_index
            b = entry.prev_backend
        steps.reverse()
        return steps


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _add_cost(a: Cost, b: Cost, *, parallel: bool = False) -> Cost:
    """Combine two cost estimates.

    When ``parallel`` is ``False`` the costs are composed sequentially:
    runtimes add up while memory requirements are dominated by the larger
    contribution.  For ``parallel=True`` the operands represent
    independent groups executed concurrently where runtime is governed by
    the slower branch and memory adds up across groups.
    """

    if parallel:
        return Cost(
            time=max(a.time, b.time),
            memory=a.memory + b.memory,
            log_depth=max(a.log_depth, b.log_depth),
            conversion=a.conversion + b.conversion,
            replay=a.replay + b.replay,
        )
    return Cost(
        time=a.time + b.time,
        memory=max(a.memory, b.memory),
        log_depth=max(a.log_depth, b.log_depth),
        conversion=a.conversion + b.conversion,
        replay=a.replay + b.replay,
    )


def _remap_gates_to_dense_indices(
    gates: Iterable["Gate"],
) -> tuple[list["Gate"], int]:
    """Return a copy of ``gates`` with qubit indices remapped to start at zero."""

    gate_list = list(gates)
    if not gate_list:
        return [], 0
    qubit_order = sorted({q for gate in gate_list for q in gate.qubits})
    index_map = {qubit: idx for idx, qubit in enumerate(qubit_order)}
    if all(index_map[q] == q for q in index_map):
        return list(gate_list), len(qubit_order)
    remapped = [
        Gate(gate.gate, [index_map[q] for q in gate.qubits], params=dict(gate.params))
        for gate in gate_list
    ]
    return remapped, len(qubit_order)


def _better(
    a: Cost,
    b: Cost,
    perf_prio: str = "memory",
    *,
    parallel: bool = False,
) -> bool:
    """Return ``True`` if cost ``a`` is preferable over ``b``.

    The comparison can prioritise either runtime or memory based on
    ``perf_prio``.  When ``parallel`` is ``True`` and both costs are
    identical a parallel plan is considered preferable to break ties.
    """

    if parallel and a.time == b.time and a.memory == b.memory:
        return True
    if perf_prio == "time":
        return (a.time, a.memory) < (b.time, b.memory)
    return (a.memory, a.time) < (b.memory, b.time)


def _dominates(a: Cost, b: Cost, epsilon: float) -> bool:
    """Return ``True`` if cost ``a`` dominates ``b`` within ``epsilon``.

    A cost ``a`` dominates ``b`` when both its runtime and memory are less
    than or equal to ``b`` up to a relative tolerance of ``epsilon``.  The
    helper is used for epsilon-dominance merging and branch-and-bound
    pruning of dynamic programming states.
    """

    return a.time <= b.time * (1 + epsilon) and a.memory <= b.memory * (1 + epsilon)


def _prune_epsilon(
    entries: Dict[Optional[Backend], DPEntry],
    *,
    epsilon: float,
    perf_prio: str,
) -> Dict[Optional[Backend], DPEntry]:
    """Merge near-identical states for the same backend within ``epsilon``.

    The planner tracks a single entry per backend.  The helper therefore only
    needs to resolve the unlikely case where multiple entries for the same
    backend arise.  Cross-backend pruning is intentionally avoided so that
    alternative backends remain available for later conversions.
    """

    pruned: Dict[Optional[Backend], DPEntry] = {}
    for backend, entry in entries.items():
        existing = pruned.get(backend)
        if existing is None:
            pruned[backend] = entry
            continue
        if _dominates(existing.cost, entry.cost, epsilon):
            continue
        if _dominates(entry.cost, existing.cost, epsilon) or _better(
            entry.cost, existing.cost, perf_prio
        ):
            pruned[backend] = entry
    return pruned


def _supported_backends(
    gates: Iterable[Gate],
    *,
    metrics: SupportedBackendMetrics | None = None,
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
    metrics:
        Optional pre-computed statistics describing ``gates``.  When omitted the
        helper derives the metrics on demand.
    sparsity:
        Optional sparsity metric for the overall circuit.  Overrides
        ``metrics.sparsity`` when provided.
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

    metrics_sparsity = metrics.sparsity if metrics is not None else None
    metrics_phase = metrics.phase_rotation_diversity if metrics is not None else None
    metrics_amp = (
        metrics.amplitude_rotation_diversity if metrics is not None else None
    )

    if circuit is not None:
        if sparsity is None and metrics_sparsity is None:
            metrics_sparsity = getattr(circuit, "sparsity", None)
        if phase_rotation_diversity is None and metrics_phase is None:
            metrics_phase = getattr(circuit, "phase_rotation_diversity", None)
        if amplitude_rotation_diversity is None and metrics_amp is None:
            metrics_amp = getattr(circuit, "amplitude_rotation_diversity", None)

    sparse_input = sparsity if sparsity is not None else metrics_sparsity
    phase_input = (
        phase_rotation_diversity
        if phase_rotation_diversity is not None
        else metrics_phase
    )
    amp_input = (
        amplitude_rotation_diversity
        if amplitude_rotation_diversity is not None
        else metrics_amp
    )

    metrics_obj = metrics
    if metrics_obj is None:
        metrics_obj = SupportedBackendMetrics.from_gates(
            gates,
            sparsity=sparse_input,
            phase_rotation_diversity=phase_input,
            amplitude_rotation_diversity=amp_input,
        )

    num_gates = metrics_obj.num_gates
    num_qubits = metrics_obj.num_qubits
    num_meas = metrics_obj.num_meas
    num_1q = metrics_obj.num_1q
    num_2q = metrics_obj.num_2q
    num_t = metrics_obj.num_t

    sparse_value = (
        sparse_input
        if sparse_input is not None
        else (metrics_obj.sparsity if metrics_obj.sparsity is not None else 0.0)
    )
    phase_value = (
        phase_input
        if phase_input is not None
        else (metrics_obj.phase_rotation_diversity or 0)
    )
    amp_value = (
        amp_input
        if amp_input is not None
        else (metrics_obj.amplitude_rotation_diversity or 0)
    )

    clifford = metrics_obj.clifford
    if allow_tableau and clifford:
        if estimator is not None:
            cost = estimator.tableau(num_qubits, num_gates)
            # Memory check uses calibrated coefficients.
            if max_memory is not None and cost.memory > max_memory:
                return []
        return [Backend.TABLEAU]

    clifford_t = metrics_obj.clifford_plus_t

    candidates: List[Backend] = []

    sparse = sparse_value if sparse_value is not None else 0.0
    phase_rot = phase_value
    amp_rot = amp_value
    rot = max(phase_rot or 0, amp_rot or 0)
    nnz = int((1 - sparse) * (2 ** num_qubits))
    local = metrics_obj.local_multi_qubit
    from .sparsity import adaptive_dd_sparsity_threshold

    s_thresh = adaptive_dd_sparsity_threshold(num_qubits)
    amp_thresh = config.adaptive_dd_amplitude_rotation_threshold(num_qubits, sparse)
    passes = (
        sparse >= s_thresh
        and nnz <= config.DEFAULT.dd_nnz_threshold
        and phase_rot <= config.DEFAULT.dd_phase_rotation_diversity_threshold
        and amp_rot <= amp_thresh
    )
    dd_metric = False
    if passes:
        s_score = sparse / s_thresh if s_thresh > 0 else 0.0
        nnz_score = 1 - nnz / config.DEFAULT.dd_nnz_threshold
        phase_score = 1 - phase_rot / config.DEFAULT.dd_phase_rotation_diversity_threshold
        amp_score = 1 - amp_rot / amp_thresh
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

    candidates: List[Backend] = []
    mps_metric = False
    num_clifford = max(0, num_gates - num_t - num_meas)

    if estimator is None:
        if dd_metric:
            candidates.append(Backend.DECISION_DIAGRAM)
        if mps_metric:
            candidates.append(Backend.MPS)
        if clifford_t:
            candidates.append(Backend.EXTENDED_STABILIZER)
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
                "[backend-selection] ",
                f"sparsity={sparse:.6f} rotation_diversity={rot:.6f} nnz={nnz} ",
                f"locality={local} candidates={ranking_str}",
            )

        if config.DEFAULT.backend_selection_log:
            try:
                with open(config.DEFAULT.backend_selection_log, "a", encoding="utf8") as f:
                    f.write(
                        f"{sparse:.6f},{nnz},{rot:.6f},{int(local)},{ranking_str}\n",
                    )
            except OSError:
                pass

        return ranking

    if metrics_obj.two_qubit_only:
        chi_cap = estimator.chi_max
        if chi_cap is not None and chi_cap > 1:
            cost = estimator.mps(num_qubits, num_1q + num_meas, num_2q, chi_cap)
            if max_memory is None or cost.memory <= max_memory:
                mps_metric = True

    costs: Dict[Backend, Cost] = {}
    # Estimates rely on calibrated coefficients for realistic costs.
    if dd_metric:
        dd_cost = estimator.decision_diagram(num_gates=num_gates, frontier=num_qubits)
        if max_memory is None or dd_cost.memory <= max_memory:
            costs[Backend.DECISION_DIAGRAM] = dd_cost
    if mps_metric:
        mps_cost = estimator.mps(num_qubits, num_1q + num_meas, num_2q, chi=4, svd=True)
        if max_memory is None or mps_cost.memory <= max_memory:
            costs[Backend.MPS] = mps_cost
    if clifford_t:
        ext_cost = estimator.extended_stabilizer(
            num_qubits,
            num_clifford,
            num_t,
            num_meas=num_meas,
            depth=num_gates,
        )
        if max_memory is None or ext_cost.memory <= max_memory:
            costs[Backend.EXTENDED_STABILIZER] = ext_cost
    sv_cost = estimator.statevector(num_qubits, num_1q, num_2q, num_meas)
    if max_memory is None or sv_cost.memory <= max_memory:
        costs[Backend.STATEVECTOR] = sv_cost
    candidates = list(costs.keys())
    # Select backends by estimated memory then runtime to respect calibration.
    ranking = sorted(candidates, key=lambda b: (costs[b].memory, costs[b].time))
    ranking_str = ">".join(b.name for b in ranking)

    if config.DEFAULT.verbose_selection:
        print(
            "[backend-selection] ",
            f"sparsity={sparse:.6f} rotation_diversity={rot:.6f} nnz={nnz} ",
            f"locality={local} candidates={ranking_str}",
        )

    if config.DEFAULT.backend_selection_log:
        try:
            with open(config.DEFAULT.backend_selection_log, "a", encoding="utf8") as f:
                f.write(
                    f"{sparse:.6f},{nnz},{rot:.6f},{int(local)},{ranking_str}\n",
                )
        except OSError:
            pass

    return ranking


def _circuit_depth(gates: Iterable["Gate"]) -> int:
    """Return the logical depth of ``gates``.

    This is a lightweight helper mirroring :meth:`Circuit._compute_depth`
    without requiring a full :class:`Circuit` instance.
    """

    gate_list = list(gates)
    if not gate_list:
        return 0
    gate_set = {id(g): g for g in gate_list}
    indegree: Dict[int, int] = {
        id(g): sum(1 for p in g.predecessors if id(p) in gate_set) for g in gate_list
    }
    ready = [g for g in gate_list if indegree[id(g)] == 0]
    depth = 0
    while ready:
        depth += 1
        next_ready: List[Gate] = []
        for gate in ready:
            for succ in gate.successors:
                key = id(succ)
                if key in indegree:
                    indegree[key] -= 1
                    if indegree[key] == 0:
                        next_ready.append(succ)
        ready = next_ready
    return depth


def _simulation_cost(
    estimator: CostEstimator,
    backend: Backend,
    num_qubits: int,
    num_1q_gates: int,
    num_2q_gates: int,
    num_meas: int,
    *,
    num_t_gates: int = 0,
    depth: int | None = None,
) -> Cost:
    """Query the cost estimator for a simulation fragment."""

    num_gates = num_1q_gates + num_2q_gates + num_meas
    if backend == Backend.TABLEAU:
        return estimator.tableau(num_qubits, num_gates)
    if backend == Backend.MPS:
        chi = getattr(estimator, "chi_max", None) or 4
        return estimator.mps(
            num_qubits,
            num_1q_gates + num_meas,
            num_2q_gates,
            chi=chi,
            svd=True,
        )
    if backend == Backend.DECISION_DIAGRAM:
        return estimator.decision_diagram(num_gates=num_gates, frontier=num_qubits)
    if backend == Backend.EXTENDED_STABILIZER:
        num_clifford = max(0, num_gates - num_t_gates - num_meas)
        dep = depth if depth is not None else num_gates
        return estimator.extended_stabilizer(
            num_qubits,
            num_clifford,
            num_t_gates,
            num_meas=num_meas,
            depth=dep,
        )
    return estimator.statevector(num_qubits, num_1q_gates, num_2q_gates, num_meas)


def _parallel_simulation_cost(
    estimator: CostEstimator,
    backend: Backend,
    groups: List[Tuple[Tuple[int, ...], List["Gate"]]],
) -> Cost:
    """Estimate cost for executing independent groups in parallel."""

    if not groups:
        return Cost(0.0, 0.0)
    costs: List[Cost] = []
    for qubits, gates in groups:
        m = len(gates)
        meas = sum(1 for g in gates if g.gate.upper() in {"MEASURE", "RESET"})
        one = sum(
            1
            for g in gates
            if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
        )
        two = m - one - meas
        t_count = sum(1 for g in gates if g.gate.upper() in {"T", "TDG"})
        depth = _circuit_depth(gates)
        costs.append(
            _simulation_cost(
                estimator,
                backend,
                len(qubits),
                one,
                two,
                meas,
                num_t_gates=t_count,
                depth=depth,
            )
        )
    total = costs[0]
    for cost in costs[1:]:
        total = _add_cost(total, cost, parallel=True)
    return Cost(
        time=total.time + estimator.parallel_time_overhead(len(groups)),
        memory=total.memory + estimator.parallel_memory_overhead(len(groups)),
        log_depth=total.log_depth,
        conversion=total.conversion,
        replay=total.replay,
    )


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
        perf_prio: str = "memory",
        horizon: int | None = None,
        epsilon: float = 0.01,
        selector: MethodSelector | None = None,
        conversion_engine: ConversionEngine | None = None,
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
        perf_prio:
            Performance priority used when comparing candidate costs.  Set to
            ``"time"`` to favour runtime over memory or ``"memory"`` to
            prioritise lower memory consumption.  Defaults to ``"memory"``.
        horizon:
            Optional sliding window limiting the dynamic programming look-back
            range in number of gates.  ``None`` disables the limit.
        epsilon:
            Relative tolerance for epsilon-dominance comparisons during state
            pruning.  Larger values merge more nearly equivalent states.
        conversion_engine:
            Optional conversion engine supplying refined cost estimates for
            backend switches.
        """

        self.estimator = estimator or CostEstimator()
        self.selector = selector or MethodSelector(self.estimator)
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
        self.perf_prio = perf_prio
        self.horizon = horizon
        self.epsilon = epsilon
        self.conversion_engine = conversion_engine
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

    def _print_selection_diagnostics(
        self, selection: dict[str, Any], *, stage: str | None = None
    ) -> None:
        """Emit backend selection diagnostics when verbose logging is enabled."""

        if not config.DEFAULT.verbose_selection:
            return
        metrics = selection.get("metrics", {})
        prefix = "[backend-selection]"
        stage_part = f" stage={stage}" if stage else ""
        metric_fields = [
            ("num_qubits", "qubits", lambda v: str(int(v))),
            ("num_gates", "gates", lambda v: str(int(v))),
            ("sparsity", "sparsity", lambda v: f"{float(v):.6f}"),
            ("phase_rotation_diversity", "phase_div", lambda v: str(int(v))),
            ("amplitude_rotation_diversity", "amp_div", lambda v: str(int(v))),
            ("nnz", "nnz", lambda v: str(int(v))),
            ("local", "local", lambda v: "yes" if v else "no"),
            (
                "mps_long_range_fraction",
                "mps_long",
                lambda v: f"{float(v):.3f}",
            ),
            (
                "mps_long_range_extent",
                "mps_span",
                lambda v: f"{float(v):.3f}",
            ),
            (
                "mps_max_interaction_distance",
                "mps_maxd",
                lambda v: str(int(v)),
            ),
        ]
        metric_parts: List[str] = []
        for key, label, fmt in metric_fields:
            if key in metrics and metrics[key] is not None:
                metric_parts.append(f"{label}={fmt(metrics[key])}")
        if "decision_diagram_metric" in metrics:
            metric_parts.append(
                f"dd_metric={float(metrics['decision_diagram_metric']):.6f}"
            )
        if "dd_metric_threshold" in metrics:
            metric_parts.append(
                f"dd_thr={float(metrics['dd_metric_threshold']):.6f}"
            )
        if metric_parts:
            print(f"{prefix}{stage_part} metrics: {', '.join(metric_parts)}")
        else:
            print(f"{prefix}{stage_part}")

        backends = selection.get("backends", {})
        order = [
            Backend.TABLEAU,
            Backend.DECISION_DIAGRAM,
            Backend.MPS,
            Backend.STATEVECTOR,
        ]
        for backend in order:
            entry = backends.get(backend)
            if entry is None:
                continue
            status = "feasible" if entry.get("feasible") else "rejected"
            if entry.get("selected"):
                status += " (selected)"
            details: List[str] = []
            cost = entry.get("cost")
            if isinstance(cost, Cost):
                details.append(f"time={cost.time:.6f}s")
                details.append(f"memory={cost.memory:.3e}B")
            if backend == Backend.MPS and "chi" in entry:
                details.append(f"chi={entry['chi']}")
                if entry.get("chi_limit") is not None:
                    details.append(f"chi_cap={entry['chi_limit']}")
            if backend == Backend.DECISION_DIAGRAM and "metric" in entry:
                details.append(f"metric={float(entry['metric']):.6f}")
            detail_str = f" {' '.join(details)}" if details else ""
            reasons = entry.get("reasons") or []
            reason_str = ""
            if reasons:
                reason_str = " [" + "; ".join(str(r) for r in reasons) + "]"
            print(f"{prefix}   {backend.name}: {status}{detail_str}{reason_str}")

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
        horizon: int | None = None,
        epsilon: float | None = None,
        upper_bound: Cost | None = None,
        stage: str | None = None,
        diagnostics: PlanDiagnostics | None = None,
    ) -> PlanResult:
        """Internal DP routine supporting batching and pruning.

        When ``forced_backend`` is provided only that backend is considered
        during planning.  A ``ValueError`` is raised if the backend cannot
        simulate a segment of the circuit.  ``sparsity`` and rotation metrics
        are forwarded to :func:`_supported_backends`.
        """
        from .sparsity import adaptive_dd_sparsity_threshold
        eps = self.epsilon if epsilon is None else epsilon
        window = self.horizon if horizon is None else horizon
        bound = upper_bound
        width = len({q for g in gates for q in g.qubits})
        nnz_estimate = None
        if sparsity is not None:
            nnz_estimate = int((1 - sparsity) * (2 ** width))
        s_thresh = adaptive_dd_sparsity_threshold(width)
        amp_thresh = config.adaptive_dd_amplitude_rotation_threshold(width, sparsity)
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
                    or amplitude_rotation_diversity <= amp_thresh
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
                (amplitude_rotation_diversity or 0) / amp_thresh
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

        @dataclass
        class _SegmentTracker:
            """Incrementally maintain parallel groups for a fixed start index."""

            gates: List["Gate"]
            start: int
            end: int = field(default=0)
            q_to_idx: Dict[int, int] = field(default_factory=dict)
            parent: List[int] = field(default_factory=list)
            size: List[int] = field(default_factory=list)
            component_qubits: Dict[int, set[int]] = field(default_factory=dict)
            component_records: Dict[int, List[Tuple[int, "Gate"]]] = field(
                default_factory=dict
            )
            groups_cache: Optional[List[Tuple[Tuple[int, ...], List["Gate"]]]] = None

            def __post_init__(self) -> None:
                self.end = self.start

            def _add_qubit(self, qubit: int) -> int:
                idx = len(self.q_to_idx)
                self.q_to_idx[qubit] = idx
                self.parent.append(idx)
                self.size.append(1)
                self.component_qubits[idx] = {qubit}
                self.component_records[idx] = []
                return idx

            def _find(self, idx: int) -> int:
                while self.parent[idx] != idx:
                    self.parent[idx] = self.parent[self.parent[idx]]
                    idx = self.parent[idx]
                return idx

            def _merge_records(self, dst: int, src: int) -> None:
                if dst == src:
                    return
                records_dst = self.component_records.get(dst, [])
                records_src = self.component_records.pop(src, [])
                if not records_dst:
                    self.component_records[dst] = records_src
                    return
                if not records_src:
                    self.component_records[dst] = records_dst
                    return
                merged: List[Tuple[int, "Gate"]] = []
                ia = ib = 0
                while ia < len(records_dst) and ib < len(records_src):
                    if records_dst[ia][0] <= records_src[ib][0]:
                        merged.append(records_dst[ia])
                        ia += 1
                    else:
                        merged.append(records_src[ib])
                        ib += 1
                if ia < len(records_dst):
                    merged.extend(records_dst[ia:])
                if ib < len(records_src):
                    merged.extend(records_src[ib:])
                self.component_records[dst] = merged

            def _union(self, a: int, b: int) -> int:
                ra, rb = self._find(a), self._find(b)
                if ra == rb:
                    return ra
                if self.size[ra] < self.size[rb]:
                    ra, rb = rb, ra
                self.parent[rb] = ra
                self.size[ra] += self.size[rb]
                self.component_qubits.setdefault(ra, set()).update(
                    self.component_qubits.pop(rb, set())
                )
                self._merge_records(ra, rb)
                return ra

            def extend(self, end: int) -> None:
                if end <= self.end:
                    return
                for offset in range(self.end, end):
                    gate = self.gates[offset]
                    if not gate.qubits:
                        continue
                    indices: List[int] = []
                    for qubit in gate.qubits:
                        idx = self.q_to_idx.get(qubit)
                        if idx is None:
                            idx = self._add_qubit(qubit)
                        indices.append(idx)
                    root = indices[0]
                    for idx in indices[1:]:
                        root = self._union(root, idx)
                    root = self._find(root)
                    self.component_records.setdefault(root, []).append((offset, gate))
                    self.component_qubits.setdefault(root, set()).update(
                        gate.qubits
                    )
                self.end = end
                self.groups_cache = None

            def groups(self) -> List[Tuple[Tuple[int, ...], List["Gate"]]]:
                if self.groups_cache is None:
                    groups: List[Tuple[Tuple[int, ...], List["Gate"]]] = []
                    for root, qubits in self.component_qubits.items():
                        records = self.component_records.get(root, [])
                        if not records:
                            continue
                        ordered = [gate for _, gate in records]
                        groups.append((tuple(sorted(qubits)), ordered))
                    groups.sort(key=lambda item: item[0])
                    self.groups_cache = groups
                return self.groups_cache

        class _ParallelGroupCache:
            def __init__(self, gate_list: List["Gate"]):
                self._gates = gate_list
                self._trackers: Dict[int, _SegmentTracker] = {}

            def get(
                self, start: int, end: int
            ) -> List[Tuple[Tuple[int, ...], List["Gate"]]]:
                tracker = self._trackers.get(start)
                if tracker is None:
                    tracker = _SegmentTracker(self._gates, start)
                    self._trackers[start] = tracker
                tracker.extend(end)
                return tracker.groups()

            def discard_before(self, threshold: int) -> None:
                stale = [idx for idx in self._trackers if idx < threshold]
                for idx in stale:
                    del self._trackers[idx]

        parallel_cache = _ParallelGroupCache(gates)

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

        # Prefix aggregates used when evaluating candidate segments.  These
        # values allow the DP loops to avoid repeatedly slicing ``gates`` just
        # to count gate types.
        meas_ops = {"MEASURE", "RESET"}
        t_ops = {"T", "TDG"}
        prefix_1q = [0] * (n + 1)
        prefix_2q = [0] * (n + 1)
        prefix_meas = [0] * (n + 1)
        prefix_t = [0] * (n + 1)
        prefix_non_clifford = [0] * (n + 1)
        prefix_non_clifford_t = [0] * (n + 1)
        prefix_non_local = [0] * (n + 1)
        prefix_gt_two = [0] * (n + 1)
        for idx, gate in enumerate(gates, start=1):
            name = gate.gate.upper()
            is_measure = name in meas_ops
            is_1q = len(gate.qubits) == 1 and not is_measure
            is_multi = len(gate.qubits) > 1 and not is_measure
            prefix_1q[idx] = prefix_1q[idx - 1] + (1 if is_1q else 0)
            prefix_2q[idx] = prefix_2q[idx - 1] + (1 if is_multi else 0)
            prefix_meas[idx] = prefix_meas[idx - 1] + (1 if is_measure else 0)
            prefix_t[idx] = prefix_t[idx - 1] + (1 if name in t_ops else 0)
            prefix_non_clifford[idx] = prefix_non_clifford[idx - 1] + (
                1 if name not in CLIFFORD_GATES else 0
            )
            prefix_non_clifford_t[idx] = prefix_non_clifford_t[idx - 1] + (
                1 if name not in CLIFFORD_PLUS_T_GATES else 0
            )
            non_local = 0
            if is_multi:
                if len(gate.qubits) != 2 or abs(gate.qubits[0] - gate.qubits[1]) != 1:
                    non_local = 1
            prefix_non_local[idx] = prefix_non_local[idx - 1] + non_local
            prefix_gt_two[idx] = prefix_gt_two[idx - 1] + (
                1 if len(gate.qubits) > 2 else 0
            )

        table: List[Dict[Optional[Backend], DPEntry]] = [dict() for _ in range(n + 1)]
        infeasible_segments: List[Tuple[int, int, List[Tuple[Backend, Cost]]]] = []
        start_backend = initial_backend if initial_backend is not None else None
        table[0][start_backend] = DPEntry(
            cost=Cost(0.0, 0.0), prev_index=0, prev_backend=None
        )

        indices = list(range(0, n, batch_size)) + [n]

        for idx_i in range(1, len(indices)):
            i = indices[idx_i]
            if window is not None:
                parallel_cache.discard_before(max(0, i - window))
            for idx_j in range(idx_i):
                j = indices[idx_j]
                if window is not None and i - j > window:
                    continue
                num_gates = i - j
                num_meas = prefix_meas[i] - prefix_meas[j]
                num_1q = prefix_1q[i] - prefix_1q[j]
                num_2q = prefix_2q[i] - prefix_2q[j]
                num_t_segment = prefix_t[i] - prefix_t[j]

                segment_view = GateSegmentView(gates, j, i)
                segment_qubits: Set[int] | None = None
                segment_groups: List[
                    Tuple[Tuple[int, ...], List["Gate"]]
                ] | None = None
                segment_depth: int | None = None

                def ensure_qubits() -> Set[int]:
                    nonlocal segment_qubits
                    if segment_qubits is None:
                        segment_qubits = segment_view.qubits()
                    return segment_qubits

                def ensure_groups() -> List[Tuple[Tuple[int, ...], List["Gate"]]]:
                    nonlocal segment_groups
                    if segment_groups is None:
                        if len(ensure_qubits()) > 1:
                            segment_groups = parallel_cache.get(j, i)
                        else:
                            segment_groups = []
                    return segment_groups

                def ensure_depth() -> int:
                    nonlocal segment_depth
                    if segment_depth is None:
                        segment_depth = _circuit_depth(segment_view)
                    return segment_depth

                segment_qubits = ensure_qubits()
                num_qubits = len(segment_qubits)
                multi_count = prefix_2q[i] - prefix_2q[j]
                non_local_count = prefix_non_local[i] - prefix_non_local[j]
                gt_two_count = prefix_gt_two[i] - prefix_gt_two[j]
                non_clifford_count = prefix_non_clifford[i] - prefix_non_clifford[j]
                non_clifford_t_count = (
                    prefix_non_clifford_t[i] - prefix_non_clifford_t[j]
                )
                segment_metrics = SupportedBackendMetrics(
                    num_gates=num_gates,
                    num_qubits=num_qubits,
                    num_meas=num_meas,
                    num_1q=num_1q,
                    num_2q=num_2q,
                    num_t=num_t_segment,
                    clifford=(num_gates > 0 and non_clifford_count == 0),
                    clifford_plus_t=(num_gates > 0 and non_clifford_t_count == 0),
                    has_multi_qubit=multi_count > 0,
                    local_multi_qubit=(multi_count > 0 and non_local_count == 0),
                    two_qubit_only=gt_two_count == 0,
                    sparsity=sparsity,
                    phase_rotation_diversity=phase_rotation_diversity,
                    amplitude_rotation_diversity=amplitude_rotation_diversity,
                )

                backends = self._order_backends(
                    _supported_backends(
                        segment_view,
                        metrics=segment_metrics,
                        sparsity=sparsity,
                        phase_rotation_diversity=phase_rotation_diversity,
                        amplitude_rotation_diversity=amplitude_rotation_diversity,
                        allow_tableau=allow_tableau,
                        estimator=self.estimator,
                        max_memory=max_memory,
                    ),
                    dd_metric=dd_metric,
                )
                requires_true_depth = num_qubits > 1 and num_2q > 0
                if forced_backend is not None:
                    if forced_backend not in backends:
                        raise ValueError(
                            f"Backend {forced_backend} unsupported for given circuit segment"
                        )
                    backends = [forced_backend]
                candidates: List[Tuple[Backend, Cost]] = []
                violations: List[Tuple[Backend, Cost]] = []
                for backend in backends:
                    depth_hint: int | None = None
                    if backend == Backend.EXTENDED_STABILIZER:
                        if requires_true_depth:
                            depth_hint = ensure_depth()
                        else:
                            depth_hint = num_gates
                    cost = _simulation_cost(
                        self.estimator,
                        backend,
                        num_qubits,
                        num_1q,
                        num_2q,
                        num_meas,
                        num_t_gates=num_t_segment,
                        depth=depth_hint,
                    )
                    groups = ensure_groups()
                    if len(groups) > 1:
                        par_cost = _parallel_simulation_cost(
                            self.estimator, backend, groups
                        )
                        if _better(par_cost, cost, self.perf_prio, parallel=True):
                            cost = par_cost
                    if max_memory is not None and cost.memory > max_memory:
                        violations.append((backend, cost))
                        continue
                    candidates.append((backend, cost))
                if not candidates:
                    if max_memory is not None:
                        if not backends:
                            retry_backends = _supported_backends(
                                segment_view,
                                metrics=segment_metrics,
                                sparsity=sparsity,
                                phase_rotation_diversity=phase_rotation_diversity,
                                amplitude_rotation_diversity=amplitude_rotation_diversity,
                                allow_tableau=allow_tableau,
                                estimator=self.estimator,
                                max_memory=None,
                            )
                            for backend in retry_backends:
                                depth_hint = None
                                if backend == Backend.EXTENDED_STABILIZER:
                                    if requires_true_depth:
                                        depth_hint = ensure_depth()
                                    else:
                                        depth_hint = num_gates
                                retry_cost = _simulation_cost(
                                    self.estimator,
                                    backend,
                                    num_qubits,
                                    num_1q,
                                    num_2q,
                                    num_meas,
                                    num_t_gates=num_t_segment,
                                    depth=depth_hint,
                                )
                                groups = ensure_groups()
                                if len(groups) > 1:
                                    par_retry = _parallel_simulation_cost(
                                        self.estimator, backend, groups
                                    )
                                    if _better(
                                        par_retry, retry_cost, self.perf_prio, parallel=True
                                    ):
                                        retry_cost = par_retry
                                violations.append((backend, retry_cost))
                        infeasible_segments.append((j, i, violations.copy()))
                    continue
                for backend, sim_cost in candidates:
                    for prev_backend, prev_entry in table[j].items():
                        if bound is not None and _dominates(bound, prev_entry.cost, eps):
                            continue
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
                                est_time = conv_est.cost.time
                                est_mem = conv_est.cost.memory
                                primitive = getattr(conv_est, "primitive", None)
                                if self.conversion_engine is not None:
                                    try:
                                        ce_time, ce_mem = self.conversion_engine.estimate_cost(
                                            len(boundary), backend
                                        )
                                        est_time = max(est_time, ce_time)
                                        est_mem = max(est_mem, ce_mem)
                                    except Exception:
                                        pass
                                conv_cost = Cost(
                                    time=est_time * self.conversion_cost_multiplier,
                                    memory=est_mem,
                                )
                                if max_memory is not None and conv_cost.memory > max_memory:
                                    if diagnostics is not None and stage is not None:
                                        diagnostics.record_conversion(
                                            stage=stage,
                                            start=j,
                                            end=i,
                                            source=prev_backend,
                                            target=backend,
                                            boundary=boundary,
                                            cost=conv_cost,
                                            primitive=primitive,
                                            feasible=False,
                                            reason="memory",
                                        )
                                    continue
                                if diagnostics is not None and stage is not None:
                                    diagnostics.record_conversion(
                                        stage=stage,
                                        start=j,
                                        end=i,
                                        source=prev_backend,
                                        target=backend,
                                        boundary=boundary,
                                        cost=conv_cost,
                                        primitive=primitive,
                                    )
                        total_cost = _add_cost(
                            _add_cost(prev_entry.cost, conv_cost), sim_cost
                        )
                        if bound is not None and _dominates(bound, total_cost, eps):
                            continue
                        entry = table[i].get(backend)
                        if entry is None or _better(total_cost, entry.cost, self.perf_prio):
                            cached_groups = ensure_groups()
                            parallel_qubits = (
                                tuple(group[0] for group in cached_groups)
                                if cached_groups
                                else ()
                            )
                            table[i][backend] = DPEntry(
                                cost=total_cost,
                                prev_index=j,
                                prev_backend=prev_backend,
                                parallel=parallel_qubits,
                            )
                            if i == n:
                                if bound is None or _better(total_cost, bound, self.perf_prio):
                                    bound = total_cost
            if table[i]:
                table[i] = _prune_epsilon(
                    table[i], epsilon=eps, perf_prio=self.perf_prio
                )
            if self.top_k and len(table[i]) > self.top_k:
                def cost_key(cost: Cost) -> tuple[float, float]:
                    if self.perf_prio == "time":
                        return (cost.time, cost.memory)
                    return (cost.memory, cost.time)

                best = sorted(
                    table[i].items(), key=lambda kv: cost_key(kv[1].cost)
                )[: self.top_k]
                table[i] = dict(best)

        final_entries = table[n]
        if not final_entries:
            if max_memory is not None:
                detail = ""
                best: Tuple[int, int, Backend, Cost] | None = None
                for start, end, options in infeasible_segments:
                    for backend, cost in options:
                        if best is None or cost.memory < best[3].memory:
                            best = (start, end, backend, cost)
                if best is not None:
                    start, end, backend, cost = best
                    detail = (
                        f" Smallest estimated requirement is {cost.memory:.3e}B "
                        f"with {backend.name} for segment [{start}, {end})."
                    )
                raise NoFeasibleBackendError(
                    "No backend combination satisfies the memory limit of "
                    f"{max_memory:.3e}B." + detail
                )
            raise NoFeasibleBackendError(
                "No feasible backend combination found for the provided circuit"
            )
        backend: Optional[Backend] = None
        if target_backend is not None:
            if target_backend in final_entries:
                backend = target_backend
        else:
            def cost_key(cost: Cost) -> tuple[float, float]:
                if self.perf_prio == "time":
                    return (cost.time, cost.memory)
                return (cost.memory, cost.time)

            backend = min(
                final_entries.items(), key=lambda kv: cost_key(kv[1].cost)
            )[0]

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
        target_accuracy: float | None = None,
        max_time: float | None = None,
        selection_diagnostics: dict[str, Any] | None = None,
    ) -> Tuple[Backend, Cost]:
        """Return best single-backend estimate for the full gate list.

        Raises
        ------
        NoFeasibleBackendError
            If no backend satisfies the provided resource constraints.
        """

        remapped_gates, num_qubits = _remap_gates_to_dense_indices(gates)
        backend, cost = self.selector.select(
            remapped_gates,
            num_qubits,
            sparsity=sparsity,
            phase_rotation_diversity=phase_rotation_diversity,
            amplitude_rotation_diversity=amplitude_rotation_diversity,
            allow_tableau=allow_tableau,
            max_memory=max_memory,
            max_time=max_time,
            target_accuracy=target_accuracy,
            diagnostics=selection_diagnostics,
        )
        return backend, cost

    def plan(
        self,
        circuit: Circuit,
        *,
        analysis: AnalysisResult | None = None,
        use_cache: bool = True,
        max_memory: float | None = None,
        backend: Backend | None = None,
        target_accuracy: float | None = None,
        max_time: float | None = None,
        optimization_level: int | None = None,
        explain: bool = False,
    ) -> PlanResult:
        """Compute the optimal contiguous partition plan using optional
        coarse and refinement passes.

        The planner stores previously computed plans in an in-memory cache
        keyed by the circuit's gate sequence.  Subsequent calls for an
        identical gate sequence will reuse the cached result.  When a
        ``backend`` is supplied the entire circuit is executed on that
        backend, provided it can simulate the gates (only the Tableau backend
        is restricted to Clifford circuits).

        Parameters
        ----------
        circuit:
            Circuit to plan.
        use_cache:
            Enable cache lookup for repeated gate sequences.
        max_memory:
            Optional memory ceiling in bytes.
        backend:
            Optional backend hint forcing execution on a single simulator.
        target_accuracy:
            Desired lower bound on simulation fidelity. Forwarded to the
            :class:`MethodSelector` when evaluating candidate methods.
        max_time:
            Maximum allowed execution time in seconds according to the cost
            model. Propagated to the :class:`MethodSelector` during backend
            selection.
        optimization_level:
            Heuristic tuning knob influencing cost comparisons.
        explain:
            When ``True`` collect diagnostic information about cost comparisons
            and conversion estimates.  The diagnostics are attached to the
            returned :class:`PlanResult` via ``plan.diagnostics``.

        Raises
        ------
        NoFeasibleBackendError
            If no backend satisfies the resource constraints implied by the
            provided limits.
        """

        gates = circuit.ensure_simplified()
        diagnostics = PlanDiagnostics() if explain else None
        names = [g.gate.upper() for g in gates]
        clifford_circuit = bool(names) and all(
            name in CLIFFORD_GATES for name in names
        )
        allow_tableau = clifford_circuit

        threshold = max_memory if max_memory is not None else self.max_memory

        def finalize(res: PlanResult) -> PlanResult:
            circuit.ssd.build_metadata()
            for part in circuit.ssd.partitions:
                if part.backend not in part.compatible_methods:
                    raise ValueError("Assigned backend incompatible with partition")
            return res

        cached: PlanResult | None = None
        if use_cache and not explain:
            cached = self.cache_lookup(gates, backend)
        if cached is not None:
            circuit.ssd.conversions = list(cached.conversions)
            if analysis is not None:
                cached.analysis = analysis
            return finalize(cached)

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
        num_t_total = sum(1 for g in gates if g.gate.upper() in {"T", "TDG"})

        if self.estimator is not None:
            fidelity = (
                target_accuracy
                if target_accuracy is not None
                else config.DEFAULT.mps_target_fidelity
            )
            remapped_gates, estimator_qubits = _remap_gates_to_dense_indices(gates)
            chi_cap = self.estimator.chi_for_constraints(
                estimator_qubits, remapped_gates, fidelity, threshold
            )
            self.estimator.chi_max = chi_cap if chi_cap > 0 else None

        perf_prio = "time" if optimization_level and optimization_level > 1 else self.perf_prio

        if backend is not None:
            # Allow explicitly requested backends as long as they can simulate the
            # circuit.  Only Tableau has restrictions (it requires a Clifford
            # circuit).
            if backend == Backend.TABLEAU:
                if names and not all(name in CLIFFORD_GATES for name in names):
                    raise ValueError(f"Backend {backend} unsupported for given circuit")
            # Invoke the selector for parity with automatic planning so that
            # benchmarking treats both paths equally.
            selection_trace: dict[str, Any] | None = None
            if diagnostics is not None or config.DEFAULT.verbose_selection:
                selection_trace = {}
            selector_gates, selector_qubits = _remap_gates_to_dense_indices(gates)
            self.selector.select(
                selector_gates,
                selector_qubits,
                sparsity=circuit.sparsity,
                phase_rotation_diversity=circuit.phase_rotation_diversity,
                amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
                allow_tableau=allow_tableau,
                max_memory=threshold,
                max_time=max_time,
                target_accuracy=target_accuracy,
                diagnostics=selection_trace,
            )
            if selection_trace is not None:
                if diagnostics is not None:
                    diagnostics.backend_selection["forced"] = selection_trace
                self._print_selection_diagnostics(selection_trace, stage="forced")
            cost = _simulation_cost(
                self.estimator,
                backend,
                num_qubits,
                num_1q,
                num_2q,
                num_meas,
                num_t_gates=num_t_total,
                depth=depth,
            )
            if threshold is not None and cost.memory > threshold:
                raise ValueError("Requested backend exceeds memory threshold")
            if max_time is not None and cost.time > max_time:
                raise ValueError("Requested backend exceeds time threshold")
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
                analysis=analysis,
            )
            if diagnostics is not None:
                diagnostics.single_backend = backend
                diagnostics.single_cost = cost
                diagnostics.dp_cost = cost
                diagnostics.strategy = "forced"
                result.diagnostics = diagnostics
            if use_cache:
                self.cache_insert(gates, result, backend)
            return finalize(result)
        # Pre-compute the cost of executing the full circuit on a single backend
        single_selection: dict[str, Any] | None = None
        if diagnostics is not None or config.DEFAULT.verbose_selection:
            single_selection = {}
        single_backend_choice: Backend | None = None
        single_cost = Cost(float("inf"), float("inf"))
        try:
            single_backend_choice, single_cost = self._single_backend(
                gates,
                threshold,
                sparsity=circuit.sparsity,
                phase_rotation_diversity=circuit.phase_rotation_diversity,
                amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
                allow_tableau=allow_tableau,
                target_accuracy=target_accuracy,
                max_time=max_time,
                selection_diagnostics=single_selection,
            )
        except NoFeasibleBackendError as exc:
            if single_selection is not None:
                single_selection["error"] = str(exc)
        if single_selection is not None:
            if diagnostics is not None:
                diagnostics.backend_selection["single"] = single_selection
            self._print_selection_diagnostics(single_selection, stage="single")

        part = Partitioner()
        groups = part.parallel_groups(gates) if num_qubits > 1 else []
        if single_backend_choice is not None and len(groups) > 1:
            par_cost = _parallel_simulation_cost(
                self.estimator, single_backend_choice, groups
            )
            if _better(par_cost, single_cost, perf_prio, parallel=True):
                single_cost = par_cost
        if diagnostics is not None:
            diagnostics.single_backend = single_backend_choice
            diagnostics.single_cost = single_cost

        quick = single_backend_choice is not None
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
            if (
                (threshold is None or single_cost.memory <= threshold)
                and (max_time is None or single_cost.time <= max_time)
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
                    analysis=analysis,
                )
                circuit.ssd.conversions = []
                if diagnostics is not None:
                    diagnostics.strategy = "quick"
                    diagnostics.dp_cost = single_cost
                    result.diagnostics = diagnostics
                if use_cache:
                    self.cache_insert(gates, result, single_backend_choice)
                return finalize(result)

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
            horizon=self.horizon,
            stage="pre",
            diagnostics=diagnostics,
        )
        pre_cost = (
            pre.table[-1][pre.final_backend].cost
            if pre.table and pre.final_backend in pre.table[-1]
            else Cost(float("inf"), float("inf"))
        )
        overhead = Cost(time=len(gates) * 1e-6, memory=0.0)
        if diagnostics is not None:
            diagnostics.pre_cost = pre_cost
            diagnostics.pre_overhead = overhead
        if (
            single_backend_choice is not None
            and _better(single_cost, _add_cost(pre_cost, overhead), perf_prio)
            and (threshold is None or single_cost.memory <= threshold)
            and (max_time is None or single_cost.time <= max_time)
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
                analysis=analysis,
            )
            circuit.ssd.conversions = []
            if diagnostics is not None:
                diagnostics.strategy = "single"
                diagnostics.dp_cost = pre_cost
                result.diagnostics = diagnostics
            if use_cache:
                self.cache_insert(gates, result, single_backend_choice)
            return finalize(result)

        # Perform a coarse plan using the configured batch size for refinement
        try:
            coarse = self._dp(
                gates,
                batch_size=self.batch_size,
                max_memory=threshold,
                allow_tableau=allow_tableau,
                sparsity=circuit.sparsity,
                phase_rotation_diversity=circuit.phase_rotation_diversity,
                amplitude_rotation_diversity=circuit.amplitude_rotation_diversity,
                horizon=self.horizon,
                stage="coarse",
                diagnostics=diagnostics,
            )
        except NoFeasibleBackendError as exc:
            if threshold is not None:
                raise NoFeasibleBackendError(
                    "Unable to plan circuit under memory limit "
                    f"{threshold:.3e}B: {exc}"
                ) from exc
            raise

        dp_cost = (
            coarse.table[-1][coarse.final_backend].cost
            if coarse.table and coarse.final_backend in coarse.table[-1]
            else Cost(float("inf"), float("inf"))
        )
        if diagnostics is not None:
            diagnostics.dp_cost = dp_cost

        if (
            single_backend_choice is not None
            and _better(single_cost, dp_cost, perf_prio)
            and (threshold is None or single_cost.memory <= threshold)
            and (max_time is None or single_cost.time <= max_time)
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
                analysis=analysis,
            )
            circuit.ssd.conversions = []
            if diagnostics is not None:
                diagnostics.strategy = "single"
                result.diagnostics = diagnostics
            if use_cache:
                self.cache_insert(gates, result, single_backend_choice)
            return finalize(result)

        # If no batching was requested we are done.
        if self.batch_size == 1:
            if max_time is not None and dp_cost.time > max_time:
                raise ValueError("Estimated plan runtime exceeds max_time")
            steps = list(coarse.steps)
            conversions = self._conversions_for_steps(gates, steps)
            circuit.ssd.conversions = list(conversions)
            coarse.explicit_steps = steps
            coarse.explicit_conversions = conversions
            if diagnostics is not None:
                diagnostics.strategy = "dp"
                coarse.diagnostics = diagnostics
            if use_cache:
                self.cache_insert(gates, coarse)
            if analysis is not None:
                coarse.analysis = analysis
            return finalize(coarse)

        # Refine each coarse segment individually.
        refined_steps: List[PlanStep] = []
        prev_backend: Optional[Backend] = None
        total_cost = Cost(time=0.0, memory=0.0)
        part = Partitioner()
        for step in coarse.steps:
            segment = gates[step.start : step.end]
            try:
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
                    horizon=self.horizon,
                    stage="refine",
                    diagnostics=diagnostics,
                )
            except NoFeasibleBackendError as exc:
                if threshold is not None:
                    raise NoFeasibleBackendError(
                        "Unable to refine segment "
                        f"[{step.start}, {step.end}) under memory limit "
                        f"{threshold:.3e}B: {exc}"
                    ) from exc
                raise
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
            for sub_step in sub.steps:
                seg = gates[sub_step.start + step.start : sub_step.end + step.start]
                n = len({q for g in seg for q in g.qubits})
                m = len(seg)
                meas = sum(
                    1 for g in seg if g.gate.upper() in {"MEASURE", "RESET"}
                )
                one = sum(
                    1
                    for g in seg
                    if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
                )
                two = m - one - meas
                groups = part.parallel_groups(seg) if n > 1 else []
                t_sub = sum(1 for g in seg if g.gate.upper() in {"T", "TDG"})
                seg_depth = _circuit_depth(seg)
                cost = _simulation_cost(
                    self.estimator,
                    sub_step.backend,
                    n,
                    one,
                    two,
                    meas,
                    num_t_gates=t_sub,
                    depth=seg_depth,
                )
                if len(groups) > 1:
                    par_cost = _parallel_simulation_cost(
                        self.estimator, sub_step.backend, groups
                    )
                    if _better(par_cost, cost, self.perf_prio, parallel=True):
                        cost = par_cost
                total_cost = _add_cost(total_cost, cost)

        final_backend = refined_steps[-1].backend if refined_steps else None
        conversions = self._conversions_for_steps(gates, refined_steps)
        circuit.ssd.conversions = list(conversions)
        if diagnostics is not None:
            diagnostics.refined_cost = total_cost if refined_steps else dp_cost
            diagnostics.strategy = "refined"
        result = PlanResult(
            table=[],
            final_backend=final_backend,
            gates=gates,
            explicit_steps=refined_steps,
            explicit_conversions=conversions,
            analysis=analysis,
        )
        if diagnostics is not None:
            result.diagnostics = diagnostics
        if use_cache:
            self.cache_insert(gates, result)
        if max_time is not None:
            total = total_cost if refined_steps else dp_cost
            if total.time > max_time:
                raise ValueError("Estimated plan runtime exceeds max_time")
        return finalize(result)


__all__ = [
    "Planner",
    "PlanResult",
    "PlanStep",
    "DPEntry",
    "ConversionEstimate",
    "PlanDiagnostics",
]
