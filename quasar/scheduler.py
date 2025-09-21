from __future__ import annotations

"""Execution scheduler for QuASAr."""

from dataclasses import dataclass, field, fields
from typing import Callable, Dict, List, Sequence
from concurrent.futures import ThreadPoolExecutor
import time
import tracemalloc
import numpy as np
import stim

try:  # Optional decision diagram backend dependency
    from mqt.core import dd as mqt_dd
except ImportError:  # pragma: no cover - backend optional
    mqt_dd = None

from .planner import Planner, PlanStep, PlanResult, PlanDiagnostics, _add_cost
from .method_selector import NoFeasibleBackendError
from .analyzer import AnalysisResult
from .partitioner import CLIFFORD_GATES
from .cost import Backend, Cost, CostEstimator
from . import config
from .circuit import Circuit, Gate
from .ssd import SSD, ConversionLayer, SSDPartition, SSDCache
from .backends import (
    AerStatevectorBackend,
    AerMPSBackend,
    StimBackend,
    DecisionDiagramBackend,
)
from .backends.mps import tensor_product as mps_tensor_product
from .backends.stim_backend import direct_sum
from quasar_convert import ConversionEngine, SSD as CESD

# Type alias for cost monitoring hook
CostHook = Callable[[PlanStep, Cost, Cost], bool]


def _clone_backend_instance(template: object) -> object:
    """Return a fresh backend instance mirroring ``template`` initial fields."""

    cls = template.__class__
    try:
        field_defs = fields(cls)
    except TypeError:  # pragma: no cover - non-dataclass fallback
        return cls()
    init_kwargs = {
        f.name: getattr(template, f.name)
        for f in field_defs
        if f.init and hasattr(template, f.name)
    }
    return cls(**init_kwargs)


def _tensor_statevectors(
    left_state: np.ndarray,
    left_qubits: Sequence[int],
    right_state: np.ndarray,
    right_qubits: Sequence[int],
    merged_qubits: Sequence[int],
) -> np.ndarray:
    """Return the tensor product of two local statevectors.

    The resulting vector follows the little-endian ordering dictated by
    ``merged_qubits``.
    """

    num_qubits = len(merged_qubits)
    result = np.zeros(1 << num_qubits, dtype=complex)
    left_positions = [merged_qubits.index(q) for q in left_qubits]
    right_positions = [merged_qubits.index(q) for q in right_qubits]

    left_state = np.asarray(left_state, dtype=complex)
    right_state = np.asarray(right_state, dtype=complex)

    for basis in range(1 << num_qubits):
        bits = [(basis >> i) & 1 for i in range(num_qubits)]
        left_index = 0
        for offset, pos in enumerate(left_positions):
            left_index |= bits[pos] << offset
        right_index = 0
        for offset, pos in enumerate(right_positions):
            right_index |= bits[pos] << offset
        amplitude_left = left_state[left_index] if left_positions else 1.0
        amplitude_right = right_state[right_index] if right_positions else 1.0
        result[basis] = amplitude_left * amplitude_right

    return result


def merge_subsystems(
    left: object,
    right: object,
    gate: Gate,
    *,
    left_qubits: Sequence[int],
    right_qubits: Sequence[int],
) -> tuple[object, tuple[int, ...]]:
    """Merge two backend instances into a single combined subsystem."""

    _ = gate  # gate information may influence ordering in future enhancements
    backend_kind = getattr(left, "backend", None)
    if backend_kind != getattr(right, "backend", None):
        raise TypeError("Cannot merge simulators from different backends")
    merged_qubits: tuple[int, ...] = tuple(left_qubits) + tuple(
        q for q in right_qubits if q not in left_qubits
    )
    merged = _clone_backend_instance(left)
    if backend_kind == Backend.STATEVECTOR:
        merged.load(len(merged_qubits))
        left_state = np.asarray(left.statevector(), dtype=complex)
        right_state = np.asarray(right.statevector(), dtype=complex)
        combined = _tensor_statevectors(
            left_state, left_qubits, right_state, right_qubits, merged_qubits
        )
        merged.ingest(combined, num_qubits=len(merged_qubits))
    elif backend_kind == Backend.MPS:
        merged.load(len(merged_qubits))
        left_ssd = left.extract_ssd()
        right_ssd = right.extract_ssd()
        left_state = left_ssd.partitions[0].state if left_ssd.partitions else None
        right_state = right_ssd.partitions[0].state if right_ssd.partitions else None
        combined = mps_tensor_product(left_state, right_state)
        merged.ingest(combined, num_qubits=len(merged_qubits))
    elif backend_kind == Backend.TABLEAU:
        merged.load(len(merged_qubits))
        left_ssd = left.extract_ssd()
        right_ssd = right.extract_ssd()
        left_tab = left_ssd.partitions[0].state if left_ssd.partitions else None
        right_tab = right_ssd.partitions[0].state if right_ssd.partitions else None
        if left_tab is None or right_tab is None:
            combined = _tensor_statevectors(
                np.asarray(left.statevector(), dtype=complex),
                left_qubits,
                np.asarray(right.statevector(), dtype=complex),
                right_qubits,
                merged_qubits,
            )
            merged.ingest(combined, num_qubits=len(merged_qubits))
        else:
            combined_tab = direct_sum(left_tab, right_tab)
            merged.ingest(combined_tab, num_qubits=len(merged_qubits))
    elif backend_kind == Backend.DECISION_DIAGRAM:
        merged.load(len(merged_qubits))
        package = merged.package
        if package is None:
            raise RuntimeError("Decision diagram backend failed to initialise package")
        left_vec = package.from_vector(np.asarray(left.statevector(), dtype=complex))
        right_vec = package.from_vector(np.asarray(right.statevector(), dtype=complex))
        combined_vec = package.vector_kronecker(
            left_vec, right_vec, len(right_qubits)
        )
        package.inc_ref_vec(combined_vec)
        if merged.state is not None:
            package.dec_ref_vec(merged.state)
        merged.state = combined_vec
    else:  # pragma: no cover - unsupported backend
        raise NotImplementedError(f"Unsupported backend for merge: {backend_kind}")

    merged.history = list(getattr(left, "history", [])) + list(
        getattr(right, "history", [])
    )
    return merged, merged_qubits


@dataclass
class RunMetrics:
    """Aggregated execution metrics produced by :meth:`Scheduler.run`.

    Attributes
    ----------
    cost:
        Wall-clock and memory cost of gate execution and conversions.
    backend_switches:
        Number of times execution changed between backends.
    conversion_durations:
        List of wall-clock durations for each conversion.
    plan_cache_hits:
        Number of reused plans retrieved from the planner cache.
    fidelity:
        Optional state fidelity against a reference state.
    """

    cost: Cost = field(default_factory=lambda: Cost(time=0.0, memory=0.0))
    backend_switches: int = 0
    conversion_durations: List[float] = field(default_factory=list)
    plan_cache_hits: int = 0
    fidelity: float | None = None


@dataclass
class Scheduler:
    """Coordinate execution of planned circuit partitions."""

    planner: Planner | None = None
    conversion_engine: ConversionEngine | None = None
    backends: Dict[Backend, object] | None = None
    quick_max_qubits: int | None = config.DEFAULT.quick_max_qubits
    quick_max_gates: int | None = config.DEFAULT.quick_max_gates
    quick_max_depth: int | None = config.DEFAULT.quick_max_depth
    backend_order: List[Backend] = field(
        default_factory=lambda: list(config.DEFAULT.preferred_backend_order)
    )
    parallel_backends: List[Backend] = field(
        default_factory=lambda: list(config.DEFAULT.parallel_backends)
    )
    backend_selection_log: str | None = config.DEFAULT.backend_selection_log
    verbose_selection: bool = config.DEFAULT.verbose_selection
    coeff_ema_decay: float = config.DEFAULT.coeff_ema_decay
    # Fractional tolerance before triggering a replan due to cost mismatch
    replan_tolerance: float = 0.05
    ssd_cache: SSDCache = field(default_factory=SSDCache)

    def _clone_decision_diagram_state(
        self, ssd: SSD | None, target_backend: object | None
    ) -> tuple[int, object] | None:
        """Return a decision diagram clone tied to ``target_backend``'s package."""

        if (
            self.conversion_engine is None
            or not hasattr(self.conversion_engine, "clone_dd_edge")
            or mqt_dd is None
            or ssd is None
            or target_backend is None
        ):
            return None

        package = getattr(target_backend, "package", None)
        if package is None:
            return None
        partitions = getattr(ssd, "partitions", None) or []
        for part in partitions:
            state = getattr(part, "state", None)
            if (
                isinstance(state, tuple)
                and len(state) == 2
                and isinstance(state[1], mqt_dd.VectorDD)
            ):
                num_qubits = int(state[0])
                try:
                    return self.conversion_engine.clone_dd_edge(num_qubits, state[1], package)
                except Exception:
                    return None
        return None

    def __post_init__(self) -> None:
        if self.backends is None:
            # Instantiate default simulation backends.  The dense
            # representations use Qiskit Aer implementations; callers may
            # supply instances with custom ``method`` arguments via the
            # ``backends`` parameter.  The other optional backends fall back
            # to stub classes when their dependencies are missing.
            self.backends = {
                Backend.STATEVECTOR: AerStatevectorBackend(),
                Backend.MPS: AerMPSBackend(),
                Backend.TABLEAU: StimBackend(),
                Backend.DECISION_DIAGRAM: DecisionDiagramBackend(),
            }

    def should_use_quick_path(
        self,
        circuit: Circuit,
        *,
        backend: Backend | None = None,
        max_time: float | None = None,
        optimization_level: int | None = None,
        force: bool = False,
    ) -> bool:
        """Return ``True`` if ``circuit`` can bypass planning.

        The decision is based on the quick-path heuristics configured for the
        scheduler.  Explicitly setting ``force`` to ``True`` bypasses these
        heuristics and unconditionally enables quick-path execution.  When a
        backend is specified the heuristics are still evaluated unless
        ``force`` is also ``True``.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        backend:
            Optional override selecting a specific backend.
        max_time:
            When provided, quick-path execution is disabled to ensure the plan
            can be checked against the runtime constraint.
        optimization_level:
            Heuristic tuning knob.  ``0`` forces full planning regardless of
            circuit size.
        force:
            When ``True`` skip heuristic checks and force quick-path execution.

        Returns
        -------
        bool
            ``True`` when the circuit is small enough to execute directly
            without invoking the planner or when ``force`` is ``True``.
        """

        if force:
            return True
        if max_time is not None:
            return False
        if optimization_level is not None and optimization_level <= 0:
            return False

        quick = True
        num_qubits = circuit.num_qubits
        num_gates = len(circuit.topological())
        depth = circuit.depth
        if self.quick_max_qubits is not None and num_qubits > self.quick_max_qubits:
            quick = False
        if self.quick_max_gates is not None and num_gates > self.quick_max_gates:
            quick = False
        if self.quick_max_depth is not None and depth > self.quick_max_depth:
            quick = False

        return quick and any(
            t is not None
            for t in (self.quick_max_qubits, self.quick_max_gates, self.quick_max_depth)
        )

    # ------------------------------------------------------------------
    def select_backend(
        self,
        circuit: Circuit,
        *,
        backend: Backend | None = None,
        max_time: float | None = None,
        optimization_level: int | None = None,
    ) -> Backend | None:
        """Return the backend ``run`` would use for ``circuit``.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        backend:
            Optional override selecting a specific backend.  When provided the
            returned value will always be this backend.
        max_time:
            Runtime constraint in seconds.  When set, quick-path execution is
            disabled to ensure the constraint can be validated.
        optimization_level:
            Heuristic tuning knob influencing quick-path checks.

        Returns
        -------
        Backend | None
            The backend chosen for direct execution without planning or
            ``None`` if the scheduler would perform full planning.
        """

        if backend is not None:
            return backend

        if not self.should_use_quick_path(
            circuit,
            max_time=max_time,
            optimization_level=optimization_level,
        ):
            return None

        names = [g.gate.upper() for g in circuit.topological()]
        num_qubits = circuit.num_qubits

        sparsity = getattr(circuit, "sparsity", None)
        phase_rot = getattr(circuit, "phase_rotation_diversity", None)
        amp_rot = getattr(circuit, "amplitude_rotation_diversity", None)
        from .sparsity import sparsity_estimate, adaptive_dd_sparsity_threshold
        from .symmetry import phase_rotation_diversity, amplitude_rotation_diversity

        if sparsity is None:
            sparsity = sparsity_estimate(circuit)
        if phase_rot is None:
            phase_rot = phase_rotation_diversity(circuit)
        if amp_rot is None:
            amp_rot = amplitude_rotation_diversity(circuit)

        nnz_estimate = int((1 - sparsity) * (2**num_qubits))
        s_thresh = adaptive_dd_sparsity_threshold(num_qubits)
        amp_thresh = config.adaptive_dd_amplitude_rotation_threshold(
            num_qubits, sparsity
        )
        s_score = sparsity / s_thresh if s_thresh > 0 else 0.0
        nnz_score = 1 - nnz_estimate / config.DEFAULT.dd_nnz_threshold
        phase_score = (
            1 - phase_rot / config.DEFAULT.dd_phase_rotation_diversity_threshold
        )
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
        passes = (
            sparsity >= s_thresh
            and nnz_estimate <= config.DEFAULT.dd_nnz_threshold
            and phase_rot <= config.DEFAULT.dd_phase_rotation_diversity_threshold
            and amp_rot <= amp_thresh
        )
        dd_metric = passes and metric >= config.DEFAULT.dd_metric_threshold

        multi = [g for g in circuit.topological() if len(g.qubits) > 1]
        local = bool(multi) and all(
            len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi
        )

        candidates: List[Backend]
        if names and all(name in CLIFFORD_GATES for name in names):
            candidates = [Backend.TABLEAU]
        else:
            candidates = []
            if dd_metric:
                candidates.append(Backend.DECISION_DIAGRAM)
            if local:
                candidates.append(Backend.MPS)
            candidates.append(Backend.STATEVECTOR)

        def backend_rank(b: Backend) -> int:
            try:
                return self.backend_order.index(b)
            except ValueError:
                return len(self.backend_order)

        def order_backends(backends: List[Backend]) -> List[Backend]:
            def rank(b: Backend) -> int:
                if dd_metric and b == Backend.DECISION_DIAGRAM:
                    return -1
                return backend_rank(b)

            return sorted(backends, key=lambda b: (b != Backend.TABLEAU, rank(b)))

        ranking = order_backends(candidates)
        backend_choice = ranking[0]

        rotation = max(phase_rot, amp_rot)
        ranking_str = ">".join(b.name for b in ranking)
        if self.verbose_selection:
            print(
                "[backend-selection] "
                f"sparsity={sparsity:.6f} rotation_diversity={rotation:.6f} "
                f"nnz={nnz_estimate} locality={local} candidates={ranking_str} "
                f"selected={backend_choice.name}"
            )

        if self.backend_selection_log:
            try:
                with open(self.backend_selection_log, "a", encoding="utf8") as f:
                    f.write(
                        f"{sparsity:.6f},{nnz_estimate},{phase_rot:.6f},{amp_rot:.6f},{int(local)},{backend_choice.name},{metric:.6f},{ranking_str}\n"
                    )
            except OSError:
                pass

        return backend_choice

    # ------------------------------------------------------------------
    def prepare_run(
        self,
        circuit: Circuit,
        plan: PlanResult | None = None,
        *,
        analysis: AnalysisResult | None = None,
        backend: Backend | None = None,
        target_accuracy: float | None = None,
        max_time: float | None = None,
        optimization_level: int | None = None,
        max_memory: float | None = None,
        explain: bool = False,
    ) -> PlanResult:
        """Prepare an execution plan for ``circuit``.

        This step performs cache lookups and cost estimation but does not
        execute any gates.  The returned :class:`PlanResult` contains
        precomputed cost estimates for each step which allows
        :meth:`run` to execute without invoking the planner again.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        plan:
            Optional pre-computed plan to reuse.
        backend:
            Optional backend hint for planning.
        target_accuracy:
            Desired lower bound on simulation fidelity.
        max_time:
            Upper bound on estimated runtime in seconds.
        max_memory:
            Optional memory ceiling in bytes. Forwarded to the planner when a
            cached plan is not available so that backend selection respects the
            configured limit.
        optimization_level:
            Heuristic tuning knob influencing planner behaviour.
        explain:
            Enable planner diagnostics on the returned :class:`PlanResult`.
        """

        gates = circuit.simplify_classical_controls()

        backend_choice = self.select_backend(
            circuit,
            backend=backend,
            max_time=max_time,
            optimization_level=optimization_level,
        )
        if plan is None and backend_choice is not None:
            if self.planner is not None:
                quick_cost = self._estimate_cost(backend_choice, gates)
            else:
                estimator = CostEstimator()
                quick_cost = self._cost_from_estimator(
                    estimator, backend_choice, gates
                )
            if max_memory is not None and quick_cost.memory > max_memory:
                backend_choice = None
            else:
                # Quick path – execute the entire circuit on a single backend
                plan = PlanResult(
                    table=[],
                    final_backend=backend_choice,
                    gates=gates,
                    explicit_steps=[PlanStep(0, len(gates), backend_choice)],
                    analysis=analysis,
                )
                plan.explicit_conversions = []
                plan.step_costs = [quick_cost]
                if explain:
                    plan.diagnostics = PlanDiagnostics(
                        single_backend=backend_choice,
                        single_cost=quick_cost,
                        strategy="quick",
                    )
                circuit.ssd.conversions = []
                qubits = tuple(range(circuit.num_qubits))
                history = tuple(g.gate for g in gates)
                circuit.ssd.partitions = [
                    SSDPartition(
                        subsystems=(qubits,),
                        history=history,
                        backend=backend_choice,
                        cost=quick_cost,
                    )
                ]
                if max_time is not None and quick_cost.time > max_time:
                    raise ValueError("Estimated runtime exceeds max_time")
                return plan

        if self.conversion_engine is None:
            self.conversion_engine = ConversionEngine()
        if self.planner is None:
            self.planner = Planner(
                quick_max_qubits=self.quick_max_qubits,
                quick_max_gates=self.quick_max_gates,
                quick_max_depth=self.quick_max_depth,
                backend_order=self.backend_order,
                conversion_engine=self.conversion_engine,
            )
        elif getattr(self.planner, "conversion_engine", None) is None:
            self.planner.conversion_engine = self.conversion_engine

        if plan is None:
            cached_plan: PlanResult | None = None
            if not explain:
                cached_plan = self.planner.cache_lookup(gates, backend)
            if cached_plan is None:
                try:
                    plan = self.planner.plan(
                        circuit,
                        analysis=analysis,
                        backend=backend,
                        target_accuracy=target_accuracy,
                        max_time=max_time,
                        max_memory=max_memory,
                        optimization_level=optimization_level,
                        explain=explain,
                    )
                except NoFeasibleBackendError as exc:
                    if max_memory is not None:
                        raise NoFeasibleBackendError(
                            "Planner could not satisfy memory limit "
                            f"{max_memory:.3e}B: {exc}"
                        ) from exc
                    raise
            else:
                plan = cached_plan

        conversions = list(getattr(plan, "conversions", []))
        circuit.ssd.conversions = conversions
        plan.explicit_conversions = conversions

        steps = plan.steps
        step_costs: List[Cost] = []
        for step in steps:
            segment = gates[step.start : step.end]
            step_costs.append(self._estimate_cost(step.backend, segment))
        plan.step_costs = step_costs
        parts: List[SSDPartition] = []
        total_cost = Cost(time=0.0, memory=0.0)
        for step, cost in zip(steps, step_costs):
            segment = gates[step.start : step.end]
            qubits = tuple(sorted({q for g in segment for q in g.qubits}))
            history = tuple(g.gate for g in segment)
            subsystems = (
                tuple(tuple(sorted(grp)) for grp in step.parallel)
                if step.parallel
                else (qubits,)
            )
            parts.append(
                SSDPartition(
                    subsystems=subsystems,
                    history=history,
                    backend=step.backend,
                    cost=cost,
                )
            )
            total_cost = _add_cost(total_cost, cost)

        merged_steps: List[PlanStep] = []
        merged_costs: List[Cost] = []
        merged_parts: List[SSDPartition] = []
        for step, cost, part in zip(steps, step_costs, parts):
            if merged_steps and step.backend == merged_steps[-1].backend:
                prev_step = merged_steps[-1]
                merged_steps[-1] = PlanStep(
                    start=prev_step.start,
                    end=step.end,
                    backend=step.backend,
                    parallel=(),
                )
                merged_costs[-1] = _add_cost(merged_costs[-1], cost)
                prev_part = merged_parts[-1]
                # Merge qubit groups preserving any independent subsystems
                groups = [set(g) for g in prev_part.subsystems]
                for grp in part.subsystems:
                    s = set(grp)
                    for existing in groups:
                        if existing & s:
                            existing |= s
                            break
                    else:
                        groups.append(s)
                merged_state = None
                prev_state = prev_part.state
                part_state = part.state
                if prev_state is None:
                    merged_state = part_state
                elif part_state is None:
                    merged_state = prev_state
                elif (
                    step.backend == Backend.TABLEAU
                    and isinstance(prev_state, stim.Tableau)
                    and isinstance(part_state, stim.Tableau)
                ):
                    try:
                        merged_state = direct_sum(prev_state, part_state)
                    except Exception:  # pragma: no cover - fallback when stim fails
                        merged_state = None

                merged_parts[-1] = SSDPartition(
                    subsystems=tuple(tuple(sorted(g)) for g in groups),
                    history=prev_part.history + part.history,
                    backend=prev_part.backend,
                    cost=_add_cost(prev_part.cost, part.cost),
                    state=merged_state,
                )
            else:
                merged_steps.append(step)
                merged_costs.append(cost)
                merged_parts.append(part)

        plan.explicit_steps = merged_steps
        try:
            plan.steps = merged_steps  # type: ignore[assignment]
        except Exception:
            pass
        plan.step_costs = merged_costs
        parts = merged_parts
        circuit.ssd.partitions = parts
        if not circuit.ssd.conversions and len(plan.explicit_steps) == 1:
            circuit.ssd.partitions = [
                SSDPartition(
                    subsystems=(tuple(range(circuit.num_qubits)),),
                    history=tuple(g.gate for g in gates),
                    backend=plan.explicit_steps[0].backend,
                    cost=plan.step_costs[0],
                )
            ]
            circuit.ssd.conversions = []
            plan.explicit_conversions = []
        if not hasattr(plan, "replay_ssd"):
            plan.replay_ssd = {}
        sims: Dict[tuple[frozenset[int], Backend], tuple[object, tuple[int, ...]]] = {}
        for idx, step in enumerate(plan.steps):
            segment = gates[step.start : step.end]
            target = step.backend
            qubit_set = {q for g in segment for q in g.qubits}
            key = (frozenset(qubit_set), target)
            sim_obj: object | None = None
            sim_qubits: tuple[int, ...] | None = None
            if len(segment) == 1 and len(segment[0].qubits) == 2:
                gate = segment[0]
                left_info = next(
                    (
                        (k, s)
                        for k, s in sims.items()
                        if gate.qubits[0] in k[0]
                    ),
                    None,
                )
                right_info = next(
                    (
                        (k, s)
                        for k, s in sims.items()
                        if gate.qubits[1] in k[0]
                    ),
                    None,
                )
                if (
                    left_info
                    and right_info
                    and left_info[1][0] is not right_info[1][0]
                ):
                    left_sim, left_qubits = left_info[1]
                    right_sim, right_qubits = right_info[1]
                    merged_sim, merged_qubits = merge_subsystems(
                        left_sim,
                        right_sim,
                        gate,
                        left_qubits=left_qubits,
                        right_qubits=right_qubits,
                    )
                    layer = ConversionLayer(
                        boundary=tuple(gate.qubits),
                        source=left_info[0][1],
                        target=right_info[0][1],
                        rank=2,
                        frontier=len(gate.qubits),
                        primitive="BRIDGE",
                        cost=Cost(time=0.0, memory=0.0),
                    )
                    plan.explicit_conversions.append(layer)
                    circuit.ssd.conversions.append(layer)
                    for victim in (left_info[0], right_info[0]):
                        sims.pop(victim, None)
                    key = (frozenset(merged_qubits), target)
                    sim_obj = merged_sim
                    sim_qubits = merged_qubits
                    qubit_set = set(merged_qubits)
                    sims[key] = (sim_obj, sim_qubits)
                    try:
                        plan.replay_ssd[idx] = np.array(
                            merged_sim.statevector(), copy=True
                        )
                    except Exception:
                        plan.replay_ssd[idx] = merged_sim.extract_ssd()
            if sim_obj is None or sim_qubits is None:
                if key not in sims:
                    sim_qubits = tuple(sorted(qubit_set))
                    template = self.backends[target]
                    sim_obj = _clone_backend_instance(template)
                    sim_obj.load(len(sim_qubits))
                    sims[key] = (sim_obj, sim_qubits)
                else:
                    sim_obj, sim_qubits = sims[key]
            for existing_key in list(sims.keys()):
                if existing_key == key:
                    continue
                if existing_key[0] & set(sim_qubits) and existing_key[1] != target:
                    sims.pop(existing_key)
            mapping = {q: idx for idx, q in enumerate(sim_qubits)}
            for gate in segment:
                local_qubits = [mapping[q] for q in gate.qubits]
                sim_obj.apply_gate(gate.gate, local_qubits, gate.params)
            sims[key] = (sim_obj, sim_qubits)

        if max_time is not None and total_cost.time > max_time:
            raise ValueError("Estimated runtime exceeds max_time")
        return plan

    # ------------------------------------------------------------------
    def run(
        self,
        circuit: Circuit,
        plan: PlanResult | None = None,
        monitor: CostHook | None = None,
        *,
        analysis: AnalysisResult | None = None,
        instrument: bool = False,
        backend: Backend | None = None,
        target_accuracy: float | None = None,
        max_time: float | None = None,
        optimization_level: int | None = None,
        reference_state: List[complex] | np.ndarray | None = None,
    ) -> SSD | tuple[SSD, RunMetrics]:
        """Execute ``circuit`` according to a plan.

        When ``plan`` is ``None`` the method performs planning internally using
        :meth:`prepare_run`.  Providing an explicit :class:`PlanResult` allows
        callers to precompute a plan (e.g. for separate timing of planning and
        execution) and ensures that this method skips any additional calls to
        :meth:`Planner.plan`.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        plan:
            Optional precomputed execution plan.
        monitor:
            Optional callback receiving ``(step, observed, estimated)``.  The
            callback is invoked for each step but its return value is ignored.
        instrument:
            Enable timing and memory instrumentation using ``time.perf_counter``
            and :mod:`tracemalloc`.  When ``False`` (the default) these metrics
            are skipped and any supplied ``monitor`` callback is not invoked.
        backend:
            Optional backend hint used when planning is performed internally.
        target_accuracy:
            Desired lower bound on simulation fidelity.
        max_time:
            Upper bound on estimated runtime in seconds.
        optimization_level:
            Heuristic tuning knob influencing planner behaviour.

        Returns
        -------
        SSD
            Descriptor of the simulated state after all gates have been
            executed.  When ``instrument`` is ``True`` a tuple of
            ``(ssd, metrics)`` is returned where ``metrics`` records the
            aggregated wall-clock and memory cost of gate execution, the
            number of backend switches, individual conversion durations and
            any plan cache hits.  ``metrics.fidelity`` contains the state
            fidelity when ``reference_state`` is provided.
        """
        backend_switches = 0
        prev_backend: Backend | None = None
        conversion_durations: List[float] = []
        cache_hits_before = (
            self.planner.cache_hits if self.planner is not None else 0
        )

        if plan is None or plan.step_costs is None:
            plan = self.prepare_run(
                circuit,
                plan,
                analysis=analysis,
                backend=backend,
                target_accuracy=target_accuracy,
                max_time=max_time,
                optimization_level=optimization_level,
            )

        gates = circuit.simplify_classical_controls()

        steps: List[PlanStep] = list(plan.steps)
        conv_layers = list(getattr(plan, "conversions", []))
        conv_idx = 0
        est_costs = plan.step_costs or [Cost(time=0.0, memory=0.0)] * len(steps)
        if max_time is not None:
            total_est = Cost(time=0.0, memory=0.0)
            for c in est_costs:
                total_est = _add_cost(total_est, c)
            if total_est.time > max_time:
                raise ValueError("Estimated runtime exceeds max_time")

        if len(steps) == 1 and not conv_layers and not getattr(
            plan, "explicit_conversions", None
        ):
            step = steps[0]
            target = step.backend
            segment = gates[step.start : step.end]
            est_cost = est_costs[0]
            sim_obj = type(self.backends[target])()
            sim_obj.load(circuit.num_qubits)
            prev_backend = target
            if instrument:
                tracemalloc.start()
                tracemalloc.reset_peak()
                start_time = time.perf_counter()
            for gate in segment:
                sim_obj.apply_gate(gate.gate, gate.qubits, gate.params)
            if instrument:
                elapsed = time.perf_counter() - start_time
                _, peak = tracemalloc.get_traced_memory()
                total_gate_time = Cost(time=elapsed, memory=float(peak))
                observed = Cost(time=elapsed, memory=float(peak))
                coeff = {
                    Backend.STATEVECTOR: (
                        ["sv_gate_1q", "sv_gate_2q", "sv_meas"],
                        "sv_bytes_per_amp",
                    ),
                    Backend.MPS: (
                        ["mps_gate_1q", "mps_gate_2q", "mps_trunc"],
                        "mps_mem",
                    ),
                    Backend.TABLEAU: (["tab_gate"], "tab_mem"),
                    Backend.DECISION_DIAGRAM: (["dd_gate"], "dd_mem"),
                }[target]
                est = self.planner.estimator if self.planner is not None else None
                if est is not None:
                    updates: Dict[str, float] = {}
                    gate_keys, mem_key = coeff
                    if est_cost.time > 0:
                        ratio = observed.time / est_cost.time
                        for gk in gate_keys:
                            updates[gk] = est.coeff[gk] * ratio
                    if est_cost.memory > 0 and observed.memory > 0:
                        updates[mem_key] = (
                            est.coeff[mem_key]
                            * observed.memory
                            / est_cost.memory
                        )
                    if updates:
                        est.update_coefficients(updates, decay=self.coeff_ema_decay)
                if monitor:
                    monitor(step, observed, est_cost)
                tracemalloc.reset_peak()
                start_time = time.perf_counter()
            if self.conversion_engine is not None:
                try:
                    self.conversion_engine.extract_ssd([], 0)
                except Exception:
                    pass
            ssd = sim_obj.extract_ssd()
            if instrument:
                elapsed = time.perf_counter() - start_time
                _, peak = tracemalloc.get_traced_memory()
                conversion_time = elapsed
                conversion_durations.append(elapsed)
                total_gate_time.memory = max(total_gate_time.memory, float(peak))
                run_cost = Cost(
                    time=total_gate_time.time,
                    memory=total_gate_time.memory,
                    conversion=conversion_time,
                    replay=0.0,
                )
                tracemalloc.stop()
                fidelity = None
                if reference_state is not None:
                    final_sv = None
                    try:
                        if self.conversion_engine is not None:
                            target_ssd = ssd if ssd is not None else circuit.ssd
                            final_sv = self.ssd_cache.convert(
                                target_ssd,
                                "sv",
                                lambda: self.conversion_engine.convert_boundary_to_statevector(
                                    target_ssd
                                ),
                            )
                    except Exception:
                        pass
                    if final_sv is None and ssd is not None:
                        part_state = ssd.extract_state(0)
                        if part_state is not None:
                            if self.conversion_engine is not None:
                                try:
                                    final_sv = self.conversion_engine.tableau_to_statevector(part_state)
                                except Exception:
                                    final_sv = part_state
                            else:
                                final_sv = part_state
                    if isinstance(final_sv, (list, tuple, np.ndarray)):
                        ref = np.asarray(reference_state, dtype=complex)
                        state = np.asarray(final_sv, dtype=complex)
                        fidelity = float(abs(np.vdot(ref, state)) ** 2)
                metrics = RunMetrics(
                    cost=run_cost,
                    backend_switches=backend_switches,
                    conversion_durations=conversion_durations,
                    plan_cache_hits=(
                        (self.planner.cache_hits - cache_hits_before)
                        if self.planner is not None
                        else 0
                    ),
                    fidelity=fidelity,
                )
                return ssd if ssd is not None else circuit.ssd, metrics
            return ssd if ssd is not None else circuit.ssd

        sims: Dict[tuple, object] = {}
        total_gate_time = Cost(time=0.0, memory=0.0)
        conversion_time = 0.0
        replay_time = 0.0
        current_backend = None
        current_sim = None
        if instrument:
            tracemalloc.start()
        i = 0
        while i < len(steps):
            step = steps[i]
            target = step.backend
            if target != prev_backend:
                if prev_backend is not None:
                    backend_switches += 1
                prev_backend = target
            segment = gates[step.start : step.end]

            if (
                step.parallel
                and len(step.parallel) > 1
                and target in self.parallel_backends
            ):
                groups: List[List] = [[] for _ in step.parallel]
                mapping = {q: idx for idx, grp in enumerate(step.parallel) for q in grp}
                for gate in segment:
                    grp = mapping[gate.qubits[0]]
                    groups[grp].append(gate)
                jobs: List[tuple[object, List]] = []
                for grp, glist in zip(step.parallel, groups):
                    qset = frozenset(grp)
                    key_p = (qset, target)
                    if key_p not in sims:
                        sim_p = type(self.backends[target])()
                        sim_p.load(circuit.num_qubits)
                        sims[key_p] = sim_p
                    jobs.append((sims[key_p], glist))

                est_cost = est_costs[i]
                if instrument:
                    tracemalloc.reset_peak()
                    start_time = time.perf_counter()

                def run_group(job):
                    sim, glist = job
                    for g in glist:
                        sim.apply_gate(g.gate, g.qubits, g.params)

                with ThreadPoolExecutor() as executor:
                    executor.map(run_group, jobs)

                if instrument:
                    elapsed = time.perf_counter() - start_time
                    total_gate_time.time += elapsed
                    _, peak = tracemalloc.get_traced_memory()
                    total_gate_time.memory = max(total_gate_time.memory, float(peak))
                    observed = Cost(time=elapsed, memory=float(peak))

                    coeff = {
                        Backend.STATEVECTOR: (
                            ["sv_gate_1q", "sv_gate_2q", "sv_meas"],
                            "sv_bytes_per_amp",
                        ),
                        Backend.MPS: (
                            ["mps_gate_1q", "mps_gate_2q", "mps_trunc"],
                            "mps_mem",
                        ),
                        Backend.TABLEAU: (["tab_gate"], "tab_mem"),
                        Backend.DECISION_DIAGRAM: (["dd_gate"], "dd_mem"),
                    }[target]
                    est = self.planner.estimator if self.planner is not None else None
                    if est is not None:
                        updates: Dict[str, float] = {}
                        gate_keys, mem_key = coeff
                        if est_cost.time > 0:
                            ratio = observed.time / est_cost.time
                            ratio = max(0.1, min(ratio, 10.0))
                            for gk in gate_keys:
                                updates[gk] = est.coeff[gk] * ratio
                        if est_cost.memory > 0 and observed.memory > 0:
                            mem_ratio = observed.memory / est_cost.memory
                            mem_ratio = max(0.1, min(mem_ratio, 10.0))
                            updates[mem_key] = est.coeff[mem_key] * mem_ratio
                        if updates:
                            est.update_coefficients(updates, decay=self.coeff_ema_decay)

                    if monitor:
                        monitor(step, observed, est_cost)
                current_sim = None
                current_backend = None
                i += 1
                continue

            qubits = frozenset(q for g in segment for q in g.qubits)
            key = (qubits, target)

            # Detect gates that span qubits across different existing backends
            if len(segment) == 1 and len(segment[0].qubits) == 2:
                gate = segment[0]
                left_info = next(
                    ((k, s) for k, s in sims.items() if gate.qubits[0] in k[0]),
                    None,
                )
                right_info = next(
                    ((k, s) for k, s in sims.items() if gate.qubits[1] in k[0]),
                    None,
                )
                if left_info and right_info and left_info[1] is not right_info[1]:
                    prepared = plan.replay_ssd.get(i)
                    sim_obj = type(self.backends[target])()
                    sim_obj.load(circuit.num_qubits)
                    if instrument:
                        tracemalloc.reset_peak()
                        start_time = time.perf_counter()
                    try:
                        sim_obj.ingest(prepared, num_qubits=circuit.num_qubits)
                    except Exception:
                        if isinstance(prepared, SSD):
                            if target == Backend.TABLEAU:
                                rep = self.ssd_cache.convert(
                                    prepared,
                                    "tab",
                                    lambda: self.conversion_engine.convert_boundary_to_tableau(
                                        prepared
                                    ),
                                )
                            elif target == Backend.DECISION_DIAGRAM:
                                rep = self._clone_decision_diagram_state(prepared, sim_obj)
                                if rep is None:
                                    rep = self.ssd_cache.convert(
                                        prepared,
                                        "dd",
                                        lambda: self.conversion_engine.convert_boundary_to_dd(
                                            prepared
                                        ),
                                    )
                            else:
                                rep = self.ssd_cache.convert(
                                    prepared,
                                    "sv",
                                    lambda: self.conversion_engine.convert_boundary_to_statevector(
                                        prepared
                                    ),
                                )
                            sim_obj.ingest(rep, num_qubits=circuit.num_qubits)
                        else:
                            if (
                                target == Backend.TABLEAU
                                and self.conversion_engine is not None
                            ):
                                try:
                                    rep = self.conversion_engine.try_build_tableau(
                                        prepared
                                    )
                                    if rep is not None:
                                        sim_obj.ingest(
                                            rep, num_qubits=circuit.num_qubits
                                        )
                                    else:
                                        sim_obj.load(circuit.num_qubits)
                                except Exception:
                                    sim_obj.load(circuit.num_qubits)
                            else:
                                sim_obj.load(circuit.num_qubits)
                    if instrument:
                        elapsed = time.perf_counter() - start_time
                        _, peak = tracemalloc.get_traced_memory()
                        conversion_time += elapsed
                        conversion_durations.append(elapsed)
                        total_gate_time.memory = max(
                            total_gate_time.memory, float(peak)
                        )
                    sims.clear()
                    sims[(frozenset(range(circuit.num_qubits)), target)] = sim_obj
                    current_sim = sim_obj
                    current_backend = target
                    for g in segment:
                        current_sim.apply_gate(g.gate, g.qubits, g.params)
                    conv_idx += 1
                    i += 1
                    continue

            if key not in sims:
                sim_obj = type(self.backends[target])()
                sim_obj.load(circuit.num_qubits)
                sims[key] = sim_obj
            sim_obj = sims[key]

            if sim_obj is not current_sim:
                if (
                    getattr(sim_obj, "backend", None) == Backend.TABLEAU
                    and getattr(sim_obj, "num_qubits", circuit.num_qubits)
                    != circuit.num_qubits
                ):
                    sim_obj.load(circuit.num_qubits)
                if current_sim is not None:
                    if instrument:
                        tracemalloc.reset_peak()
                        start_time = time.perf_counter()
                        current_ssd = current_sim.extract_ssd()
                        elapsed = time.perf_counter() - start_time
                        _, peak = tracemalloc.get_traced_memory()
                        conversion_time += elapsed
                        conversion_durations.append(elapsed)
                        total_gate_time.memory = max(
                            total_gate_time.memory, float(peak)
                        )
                    else:
                        current_ssd = current_sim.extract_ssd()
                    layer = None
                    if conv_idx < len(conv_layers):
                        cand = conv_layers[conv_idx]
                        if cand.source == current_backend and cand.target == target:
                            layer = cand
                            conv_idx += 1
                    if layer is not None:
                        boundary = list(layer.boundary)
                        rank = layer.rank
                        primitive = layer.primitive
                    else:  # Fallback path; should not trigger in normal operation
                        if current_ssd is not None and getattr(
                            current_ssd, "partitions", None
                        ):
                            boundary = list(
                                set(current_ssd.partitions[0].qubits) & set(qubits)
                            )
                        else:
                            boundary = list(qubits)
                        rank = 2 ** len(boundary)
                        primitive = "Full"
                    conv_ssd = CESD(boundary_qubits=list(boundary), top_s=rank)
                    conv_ssd.fingerprint = (
                        tuple(conv_ssd.boundary_qubits or []),
                        conv_ssd.top_s,
                    )
                    if instrument:
                        tracemalloc.reset_peak()
                        start_time = time.perf_counter()
                    try:
                        if primitive == "B2B":
                            try:
                                sim_obj.ingest(
                                    current_ssd, num_qubits=circuit.num_qubits
                                )
                            except Exception:
                                if target == Backend.TABLEAU:
                                    rep = self.ssd_cache.convert(
                                        conv_ssd,
                                        "tab",
                                        lambda: self.conversion_engine.convert_boundary_to_tableau(
                                            conv_ssd
                                        ),
                                    )
                                elif target == Backend.DECISION_DIAGRAM:
                                    rep = self._clone_decision_diagram_state(current_ssd, sim_obj)
                                    if rep is None:
                                        rep = self.ssd_cache.convert(
                                            conv_ssd,
                                            "dd",
                                            lambda: self.conversion_engine.convert_boundary_to_dd(
                                                conv_ssd
                                            ),
                                        )
                                else:
                                    rep = self.ssd_cache.convert(
                                        conv_ssd,
                                        "sv",
                                        lambda: self.conversion_engine.convert_boundary_to_statevector(
                                            conv_ssd
                                        ),
                                    )
                                sim_obj.ingest(
                                    rep,
                                    num_qubits=circuit.num_qubits,
                                    mapping=boundary,
                                )
                        elif primitive == "LW":
                            rep = None
                            if (
                                current_backend == Backend.DECISION_DIAGRAM
                                and self.conversion_engine is not None
                                and hasattr(
                                    self.conversion_engine, "extract_local_window_dd"
                                )
                                and mqt_dd is not None
                            ):
                                partitions = getattr(current_ssd, "partitions", None) or []
                                for part in partitions:
                                    state_obj = getattr(part, "state", None)
                                    if (
                                        isinstance(state_obj, tuple)
                                        and len(state_obj) == 2
                                        and isinstance(state_obj[1], mqt_dd.VectorDD)
                                    ):
                                        try:
                                            rep = self.conversion_engine.extract_local_window_dd(
                                                state_obj[1], boundary
                                            )
                                        except Exception:
                                            rep = None
                                        break
                            if rep is None:
                                state = current_sim.statevector()
                                rep = self.conversion_engine.extract_local_window(
                                    state, boundary
                                )
                            sim_obj.ingest(
                                rep,
                                num_qubits=circuit.num_qubits,
                                mapping=boundary,
                            )
                        elif primitive == "ST":
                            rep = self.ssd_cache.bridge_tensor(
                                conv_ssd,
                                conv_ssd,
                                lambda: self.conversion_engine.build_bridge_tensor(
                                    conv_ssd, conv_ssd
                                ),
                            )
                            sim_obj.ingest(
                                rep,
                                num_qubits=circuit.num_qubits,
                                mapping=boundary,
                            )
                        else:
                            if target == Backend.TABLEAU:
                                rep = self.ssd_cache.convert(
                                    conv_ssd,
                                    "tab",
                                    lambda: self.conversion_engine.convert_boundary_to_tableau(
                                        conv_ssd
                                    ),
                                )
                            elif target == Backend.DECISION_DIAGRAM:
                                rep = self._clone_decision_diagram_state(current_ssd, sim_obj)
                                if rep is None:
                                    rep = self.ssd_cache.convert(
                                        conv_ssd,
                                        "dd",
                                        lambda: self.conversion_engine.convert_boundary_to_dd(
                                            conv_ssd
                                        ),
                                    )
                            else:
                                rep = self.ssd_cache.convert(
                                    conv_ssd,
                                    "sv",
                                    lambda: self.conversion_engine.convert_boundary_to_statevector(
                                        conv_ssd
                                    ),
                                )
                            sim_obj.ingest(
                                rep,
                                num_qubits=circuit.num_qubits,
                                mapping=boundary,
                            )
                    except Exception:
                        if (
                            target == Backend.TABLEAU
                            and self.conversion_engine is not None
                        ):
                            try:
                                state = None
                                if current_sim is not None:
                                    try:
                                        state = current_sim.statevector()
                                    except Exception:
                                        state = None
                                if state is not None:
                                    rep = self.conversion_engine.try_build_tableau(
                                        state
                                    )
                                    if rep is not None:
                                        sim_obj.ingest(
                                            rep,
                                            num_qubits=circuit.num_qubits,
                                            mapping=boundary,
                                        )
                                    else:
                                        sim_obj.load(circuit.num_qubits)
                                else:
                                    sim_obj.load(circuit.num_qubits)
                            except Exception:
                                sim_obj.load(circuit.num_qubits)
                        else:
                            sim_obj.load(circuit.num_qubits)
                    finally:
                        if instrument:
                            elapsed = time.perf_counter() - start_time
                            _, peak = tracemalloc.get_traced_memory()
                            conversion_time += elapsed
                            conversion_durations.append(elapsed)
                            total_gate_time.memory = max(
                                total_gate_time.memory, float(peak)
                            )
                current_sim = sim_obj
                current_backend = target
                for k in list(sims.keys()):
                    if k[0] == qubits and k != key:
                        sims.pop(k)

            est_cost = est_costs[i]

            if instrument:
                tracemalloc.reset_peak()
                start_time = time.perf_counter()

            for gate in segment:
                current_sim.apply_gate(gate.gate, gate.qubits, gate.params)

            if instrument:
                elapsed = time.perf_counter() - start_time
                total_gate_time.time += elapsed
                _, peak = tracemalloc.get_traced_memory()
                total_gate_time.memory = max(total_gate_time.memory, float(peak))
                observed = Cost(time=elapsed, memory=float(peak))

                # Update cost model based on observation
                coeff = {
                    Backend.STATEVECTOR: (
                        ["sv_gate_1q", "sv_gate_2q", "sv_meas"],
                        "sv_bytes_per_amp",
                    ),
                    Backend.MPS: (
                        ["mps_gate_1q", "mps_gate_2q", "mps_trunc"],
                        "mps_mem",
                    ),
                    Backend.TABLEAU: (["tab_gate"], "tab_mem"),
                    Backend.DECISION_DIAGRAM: (["dd_gate"], "dd_mem"),
                }[target]
                est = self.planner.estimator if self.planner is not None else None
                if est is not None:
                    updates: Dict[str, float] = {}
                    gate_keys, mem_key = coeff
                    if est_cost.time > 0:
                        ratio = observed.time / est_cost.time
                        for gk in gate_keys:
                            updates[gk] = est.coeff[gk] * ratio
                    if est_cost.memory > 0 and observed.memory > 0:
                        updates[mem_key] = (
                            est.coeff[mem_key] * observed.memory / est_cost.memory
                        )
                    if updates:
                        est.update_coefficients(updates, decay=self.coeff_ema_decay)

                if monitor:
                    monitor(step, observed, est_cost)
            i += 1

        if sims:
            parts: List[SSDPartition] = []
            used_qubits = set()
            for sim in sims.values():
                if instrument:
                    tracemalloc.reset_peak()
                    start_time = time.perf_counter()
                    ssd = sim.extract_ssd()
                    elapsed = time.perf_counter() - start_time
                    _, peak = tracemalloc.get_traced_memory()
                    conversion_time += elapsed
                    conversion_durations.append(elapsed)
                    total_gate_time.memory = max(total_gate_time.memory, float(peak))
                else:
                    ssd = sim.extract_ssd()
                if ssd is None:
                    continue
                parts.extend(ssd.partitions)
                used_qubits.update(q for p in ssd.partitions for q in p.qubits)
                circuit.ssd.conversions.extend(ssd.conversions)
            for part in circuit.ssd.partitions:
                if all(q not in used_qubits for q in part.qubits):
                    parts.append(part)
            ssd_res = SSD(parts, circuit.ssd.conversions)
            ssd_res.build_metadata()
            for part in ssd_res.partitions:
                if part.backend not in part.compatible_methods:
                    raise ValueError("Assigned backend incompatible with partition")
            if instrument:
                run_cost = Cost(
                    time=total_gate_time.time,
                    memory=total_gate_time.memory,
                    conversion=conversion_time,
                    replay=replay_time,
                )
                tracemalloc.stop()
                fidelity = None
                if reference_state is not None:
                    final_sv = None
                    try:
                        if self.conversion_engine is not None:
                            final_sv = self.ssd_cache.convert(
                                ssd_res,
                                "sv",
                                lambda: self.conversion_engine.convert_boundary_to_statevector(
                                    ssd_res
                                ),
                            )
                    except Exception:
                        pass
                    if final_sv is None and ssd_res is not None:
                        try:
                            part_state = ssd_res.extract_state(0)
                        except Exception:
                            part_state = None
                        if part_state is not None:
                            if self.conversion_engine is not None:
                                try:
                                    final_sv = self.conversion_engine.tableau_to_statevector(part_state)
                                except Exception:
                                    final_sv = part_state
                            else:
                                final_sv = part_state
                    if isinstance(final_sv, (list, tuple, np.ndarray)):
                        ref = np.asarray(reference_state, dtype=complex)
                        state = np.asarray(final_sv, dtype=complex)
                        fidelity = float(abs(np.vdot(ref, state)) ** 2)
                plan_cache_hits = (
                    self.planner.cache_hits - cache_hits_before
                    if self.planner is not None
                    else 0
                )
                metrics = RunMetrics(
                    cost=run_cost,
                    backend_switches=backend_switches,
                    conversion_durations=conversion_durations,
                    plan_cache_hits=plan_cache_hits,
                    fidelity=fidelity,
                )
                return ssd_res, metrics
            return ssd_res
        ssd_res = circuit.ssd
        ssd_res.build_metadata()
        if instrument:
            run_cost = Cost(
                time=total_gate_time.time,
                memory=total_gate_time.memory,
                conversion=conversion_time,
                replay=replay_time,
            )
            tracemalloc.stop()
            fidelity = None
            if reference_state is not None:
                final_sv = None
                try:
                    if self.conversion_engine is not None:
                        final_sv = self.ssd_cache.convert(
                            ssd_res,
                            "sv",
                            lambda: self.conversion_engine.convert_boundary_to_statevector(
                                ssd_res
                            ),
                        )
                except Exception:
                    pass
                if final_sv is None:
                    try:
                        part_state = ssd_res.extract_state(0)
                    except Exception:
                        part_state = None
                    if part_state is not None:
                        if self.conversion_engine is not None:
                            try:
                                final_sv = self.conversion_engine.tableau_to_statevector(part_state)
                            except Exception:
                                final_sv = part_state
                        else:
                            final_sv = part_state
                if isinstance(final_sv, (list, tuple, np.ndarray)):
                    ref = np.asarray(reference_state, dtype=complex)
                    state = np.asarray(final_sv, dtype=complex)
                    fidelity = float(abs(np.vdot(ref, state)) ** 2)
            plan_cache_hits = (
                self.planner.cache_hits - cache_hits_before
                if self.planner is not None
                else 0
            )
            metrics = RunMetrics(
                cost=run_cost,
                backend_switches=backend_switches,
                conversion_durations=conversion_durations,
                plan_cache_hits=plan_cache_hits,
                fidelity=fidelity,
            )
            return ssd_res, metrics
        return ssd_res

    # ------------------------------------------------------------------
    def _cost_from_estimator(
        self, estimator: CostEstimator, backend: Backend, gates: List[Gate]
    ) -> Cost:
        n = len({q for g in gates for q in g.qubits})
        m = len(gates)
        num_meas = sum(1 for g in gates if g.gate.upper() in {"MEASURE", "RESET"})
        num_1q = sum(
            1
            for g in gates
            if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
        )
        num_2q = m - num_1q - num_meas
        if backend == Backend.TABLEAU:
            return estimator.tableau(n, m)
        if backend == Backend.MPS:
            chi = getattr(estimator, "chi_max", None) or 4
            return estimator.mps(
                n,
                num_1q + num_meas,
                num_2q,
                chi=chi,
                svd=True,
            )
        if backend == Backend.DECISION_DIAGRAM:
            return estimator.decision_diagram(num_gates=m, frontier=n)
        return estimator.statevector(n, num_1q, num_2q, num_meas)

    def _estimate_cost(self, backend: Backend, gates: List[Gate]) -> Cost:
        return self._cost_from_estimator(self.planner.estimator, backend, gates)
