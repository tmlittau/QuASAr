from __future__ import annotations

"""Execution scheduler for QuASAr."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor
import time
import tracemalloc

from .planner import Planner, PlanStep, PlanResult
from .partitioner import CLIFFORD_GATES
from .cost import Backend, Cost
from . import config
from .circuit import Circuit, Gate
from .ssd import SSD, ConversionLayer, SSDPartition
from .backends import (
    AerStatevectorBackend,
    AerMPSBackend,
    StimBackend,
    DecisionDiagramBackend,
)
from quasar_convert import ConversionEngine, SSD as CESD

# Type alias for cost monitoring hook
CostHook = Callable[[PlanStep, Cost, Cost], bool]


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
    # Fractional tolerance before triggering a replan due to cost mismatch
    replan_tolerance: float = 0.05

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
        self, circuit: Circuit, *, backend: Backend | None = None
    ) -> bool:
        """Return ``True`` if ``circuit`` can bypass planning.

        The decision is based on the quick‑path heuristics configured for the
        scheduler.  When ``backend`` is explicitly specified the caller is
        requesting direct execution on a backend and therefore planning is not
        required.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        backend:
            Optional override selecting a specific backend.

        Returns
        -------
        bool
            ``True`` when the circuit is small enough to execute directly
            without invoking the planner.
        """

        if backend is not None:
            return True

        quick = True
        num_qubits = circuit.num_qubits
        num_gates = len(circuit.gates)
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
        self, circuit: Circuit, *, backend: Backend | None = None
    ) -> Backend | None:
        """Return the backend ``run`` would use for ``circuit``.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        backend:
            Optional override selecting a specific backend.  When provided the
            returned value will always be this backend.

        Returns
        -------
        Backend | None
            The backend chosen for direct execution without planning or
            ``None`` if the scheduler would perform full planning.
        """

        if backend is not None:
            return backend

        if not self.should_use_quick_path(circuit):
            return None

        names = [g.gate.upper() for g in circuit.gates]
        num_qubits = circuit.num_qubits
        num_gates = len(circuit.gates)

        if names and all(name in CLIFFORD_GATES for name in names):
            return Backend.TABLEAU
        if num_qubits < 20:
            return Backend.STATEVECTOR

        multi = [g for g in circuit.gates if len(g.qubits) > 1]
        local = multi and all(
            len(g.qubits) == 2 and abs(g.qubits[0] - g.qubits[1]) == 1 for g in multi
        )

        if num_gates <= 2 ** num_qubits and not local:
            return Backend.DECISION_DIAGRAM
        if local:
            return Backend.MPS
        return Backend.STATEVECTOR

    # ------------------------------------------------------------------
    def prepare_run(
        self,
        circuit: Circuit,
        plan: PlanResult | None = None,
        *,
        backend: Backend | None = None,
    ) -> PlanResult:
        """Prepare an execution plan for ``circuit``.

        This step performs cache lookups and cost estimation but does not
        execute any gates.  The returned :class:`PlanResult` contains
        precomputed cost estimates for each step which allows
        :meth:`run` to execute without invoking the planner again.
        """

        backend_choice = self.select_backend(circuit, backend=backend)
        if plan is None and backend_choice is not None:
            # Quick path – execute the entire circuit on a single backend
            plan = PlanResult(
                table=[],
                final_backend=backend_choice,
                gates=circuit.gates,
                explicit_steps=[PlanStep(0, len(circuit.gates), backend_choice)],
            )
            plan.explicit_conversions = []
            if self.planner is not None:
                plan.step_costs = [
                    self._estimate_cost(backend_choice, circuit.gates)
                ]
            else:
                plan.step_costs = [Cost(time=0.0, memory=0.0)]
            circuit.ssd.conversions = []
            return plan

        if self.planner is None:
            self.planner = Planner(
                quick_max_qubits=self.quick_max_qubits,
                quick_max_gates=self.quick_max_gates,
                quick_max_depth=self.quick_max_depth,
                backend_order=self.backend_order,
            )
        if self.conversion_engine is None:
            self.conversion_engine = ConversionEngine()

        if plan is None:
            plan = self.planner.cache_lookup(circuit.gates, backend)
            if plan is None:
                plan = self.planner.plan(circuit, backend=backend)

        conversions = list(getattr(plan, "conversions", []))
        circuit.ssd.conversions = conversions
        plan.explicit_conversions = conversions

        step_costs: List[Cost] = []
        for step in plan.steps:
            segment = circuit.gates[step.start : step.end]
            step_costs.append(self._estimate_cost(step.backend, segment))
        plan.step_costs = step_costs
        return plan

    # ------------------------------------------------------------------
    def run(
        self,
        circuit: Circuit,
        plan: PlanResult,
        monitor: CostHook | None = None,
    ) -> SSD:
        """Execute ``circuit`` according to ``plan``.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        plan:
            Prepared execution plan returned by :meth:`prepare_run`.
        monitor:
            Optional callback receiving ``(step, observed, estimated)``.  The
            callback is invoked for each step but its return value is ignored.
        Returns
        -------
        SSD
            Descriptor of the simulated state after all gates have been
            executed.
        """

        steps: List[PlanStep] = list(plan.steps)
        conv_layers = list(getattr(plan, "conversions", []))
        conv_idx = 0
        est_costs = plan.step_costs or [Cost(time=0.0, memory=0.0)] * len(steps)

        sims: Dict[tuple, object] = {}
        current_backend = None
        current_sim = None
        i = 0
        while i < len(steps):
            step = steps[i]
            target = step.backend
            segment = circuit.gates[step.start : step.end]

            if step.parallel and len(step.parallel) > 1 and target in self.parallel_backends:
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
                tracemalloc.start()
                start_time = time.perf_counter()

                def run_group(job):
                    sim, glist = job
                    for g in glist:
                        sim.apply_gate(g.gate, g.qubits, g.params)

                with ThreadPoolExecutor() as executor:
                    executor.map(run_group, jobs)

                elapsed = time.perf_counter() - start_time
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                observed = Cost(time=elapsed, memory=float(peak))

                coeff = {
                    Backend.STATEVECTOR: (["sv_gate_1q", "sv_gate_2q", "sv_meas"], "sv_mem"),
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
                        updates[mem_key] = est.coeff[mem_key] * observed.memory / est_cost.memory
                    if updates:
                        est.update_coefficients(updates)

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
                if (
                    left_info
                    and right_info
                    and left_info[1] is not right_info[1]
                ):
                    l_ssd = CESD(boundary_qubits=[gate.qubits[0]], top_s=2)
                    r_ssd = CESD(boundary_qubits=[gate.qubits[1]], top_s=2)
                    self.conversion_engine.build_bridge_tensor(l_ssd, r_ssd)
                    circuit.ssd.conversions.append(
                        ConversionLayer(
                            boundary=tuple(gate.qubits),
                            source=left_info[0][1],
                            target=right_info[0][1],
                            rank=2,
                            frontier=len(gate.qubits),
                            primitive="BRIDGE",
                            cost=Cost(time=0.0, memory=0.0),
                        )
                    )
                    sim_obj = type(self.backends[target])()
                    sim_obj.load(circuit.num_qubits)
                    for g in circuit.gates[: step.end]:
                        sim_obj.apply_gate(g.gate, g.qubits, g.params)
                    sims.clear()
                    sims[(frozenset(range(circuit.num_qubits)), target)] = sim_obj
                    current_sim = sim_obj
                    current_backend = target
                    i += 1
                    continue

            if key not in sims:
                sim_obj = type(self.backends[target])()
                sim_obj.load(circuit.num_qubits)
                sims[key] = sim_obj
            sim_obj = sims[key]

            if sim_obj is not current_sim:
                if current_sim is not None:
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
                        if current_ssd is not None and getattr(current_ssd, "partitions", None):
                            boundary = list(set(current_ssd.partitions[0].qubits) & set(qubits))
                        else:
                            boundary = list(qubits)
                        rank = 2 ** len(boundary)
                        primitive = "Full"
                    conv_ssd = CESD(boundary_qubits=list(boundary), top_s=rank)
                    try:
                        if primitive == "B2B":
                            try:
                                sim_obj.ingest(current_ssd, num_qubits=circuit.num_qubits)
                            except Exception:
                                if target == Backend.TABLEAU:
                                    rep = self.conversion_engine.convert_boundary_to_tableau(conv_ssd)
                                elif target == Backend.DECISION_DIAGRAM:
                                    rep = self.conversion_engine.convert_boundary_to_dd(conv_ssd)
                                else:
                                    rep = self.conversion_engine.convert_boundary_to_statevector(conv_ssd)
                                sim_obj.ingest(
                                    rep,
                                    num_qubits=circuit.num_qubits,
                                    mapping=boundary,
                                )
                        elif primitive == "LW":
                            state = current_sim.statevector()
                            rep = self.conversion_engine.extract_local_window(state, boundary)
                            sim_obj.ingest(
                                rep,
                                num_qubits=circuit.num_qubits,
                                mapping=boundary,
                            )
                        elif primitive == "ST":
                            rep = self.conversion_engine.build_bridge_tensor(conv_ssd, conv_ssd)
                            sim_obj.ingest(
                                rep,
                                num_qubits=circuit.num_qubits,
                                mapping=boundary,
                            )
                        else:
                            if target == Backend.TABLEAU:
                                rep = self.conversion_engine.convert_boundary_to_tableau(conv_ssd)
                            elif target == Backend.DECISION_DIAGRAM:
                                rep = self.conversion_engine.convert_boundary_to_dd(conv_ssd)
                            else:
                                rep = self.conversion_engine.convert_boundary_to_statevector(conv_ssd)
                            sim_obj.ingest(
                                rep,
                                num_qubits=circuit.num_qubits,
                                mapping=boundary,
                            )
                    except Exception:
                        sim_obj.load(circuit.num_qubits)
                current_sim = sim_obj
                current_backend = target
                for k in list(sims.keys()):
                    if k[0] == qubits and k != key:
                        sims.pop(k)

            est_cost = est_costs[i]

            tracemalloc.start()
            start_time = time.perf_counter()

            for gate in segment:
                current_sim.apply_gate(gate.gate, gate.qubits, gate.params)

            elapsed = time.perf_counter() - start_time
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            observed = Cost(time=elapsed, memory=float(peak))

            # Update cost model based on observation
            coeff = {
                Backend.STATEVECTOR: (["sv_gate_1q", "sv_gate_2q", "sv_meas"], "sv_mem"),
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
                    updates[mem_key] = est.coeff[mem_key] * observed.memory / est_cost.memory
                if updates:
                    est.update_coefficients(updates)

            if monitor:
                monitor(step, observed, est_cost)
            i += 1

        if sims:
            parts: List[SSDPartition] = []
            used_qubits = set()
            for sim in sims.values():
                ssd = sim.extract_ssd()
                if ssd is None:
                    continue
                parts.extend(ssd.partitions)
                used_qubits.update(q for p in ssd.partitions for q in p.qubits)
                circuit.ssd.conversions.extend(ssd.conversions)
            for part in circuit.ssd.partitions:
                if all(q not in used_qubits for q in part.qubits):
                    parts.append(part)
            return SSD(parts, circuit.ssd.conversions)
        return circuit.ssd

    # ------------------------------------------------------------------
    def _estimate_cost(self, backend: Backend, gates: List[Gate]) -> Cost:
        est = self.planner.estimator
        n = len({q for g in gates for q in g.qubits})
        m = len(gates)
        num_meas = sum(1 for g in gates if g.gate.upper() in {"MEASURE", "RESET"})
        num_1q = sum(
            1 for g in gates if len(g.qubits) == 1 and g.gate.upper() not in {"MEASURE", "RESET"}
        )
        num_2q = m - num_1q - num_meas
        if backend == Backend.TABLEAU:
            return est.tableau(n, m)
        if backend == Backend.MPS:
            return est.mps(
                n,
                num_1q + num_meas,
                num_2q,
                chi=4,
                svd=True,
            )
        if backend == Backend.DECISION_DIAGRAM:
            return est.decision_diagram(num_gates=m, frontier=n)
        return est.statevector(n, num_1q, num_2q, num_meas)
