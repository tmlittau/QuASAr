from __future__ import annotations

"""Execution scheduler for QuASAr."""

from dataclasses import dataclass
from typing import Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor
import time
import tracemalloc

from .planner import Planner, PlanStep, _supported_backends, _simulation_cost
from .cost import Backend, Cost, CostEstimator
from .circuit import Circuit
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
    quick_max_qubits: int | None = 25
    quick_max_gates: int | None = 200
    quick_max_depth: int | None = 50

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

    # ------------------------------------------------------------------
    def run(
        self,
        circuit: Circuit,
        monitor: CostHook | None = None,
        *,
        backend: Backend | None = None,
    ) -> SSD:
        """Execute ``circuit`` according to a planner-derived schedule.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        monitor:
            Optional callback receiving ``(step, observed, estimated)``.  If the
            callback returns ``True`` the scheduler re-plans the remaining gates
            starting from ``step.end``.
        Returns
        -------
        SSD
            Descriptor of the simulated state after all gates have been
            executed.
        """

        if backend is not None:
            sim = type(self.backends[backend])()
            sim.load(circuit.num_qubits)
            for gate in circuit.gates:
                sim.apply_gate(gate.gate, gate.qubits, gate.params)
            ssd = sim.extract_ssd()
            return ssd if ssd is not None else circuit.ssd

        # Quick path: directly pick best single backend without planner
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
        if quick and any(
            t is not None
            for t in (self.quick_max_qubits, self.quick_max_gates, self.quick_max_depth)
        ):
            estimator = CostEstimator()
            candidates: List[tuple[Backend, Cost]] = []
            for b in _supported_backends(circuit.gates):
                cost = _simulation_cost(estimator, b, num_qubits, num_gates)
                candidates.append((b, cost))
            backend_choice = min(
                candidates, key=lambda kv: (kv[1].time, kv[1].memory)
            )[0]
            sim = type(self.backends[backend_choice])()
            sim.load(num_qubits)
            for gate in circuit.gates:
                sim.apply_gate(gate.gate, gate.qubits, gate.params)
            ssd = sim.extract_ssd()
            return ssd if ssd is not None else circuit.ssd

        if self.planner is None:
            self.planner = Planner(
                quick_max_qubits=self.quick_max_qubits,
                quick_max_gates=self.quick_max_gates,
                quick_max_depth=self.quick_max_depth,
            )
        if self.conversion_engine is None:
            self.conversion_engine = ConversionEngine()

        plan = self.planner.cache_lookup(circuit.gates, backend)
        if plan is None:
            plan = self.planner.plan(circuit, backend=backend)
        steps: List[PlanStep] = list(plan.steps)

        sims: Dict[tuple, object] = {}
        current_backend = None
        current_sim = None
        i = 0
        while i < len(steps):
            step = steps[i]
            target = step.backend

            segment = circuit.gates[step.start : step.end]
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
                    current_sim = None
                    current_backend = None
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
                    layer = next(
                        (
                            c
                            for c in circuit.ssd.conversions
                            if c.source == current_backend and c.target == target
                        ),
                        None,
                    )
                    if layer:
                        boundary = list(layer.boundary)
                        rank = layer.rank
                    else:
                        if current_ssd is not None and getattr(current_ssd, "partitions", None):
                            boundary = list(set(current_ssd.partitions[0].qubits) & set(qubits))
                        else:
                            boundary = list(qubits)
                        rank = 2 ** len(boundary)
                    conv_ssd = CESD(boundary_qubits=list(boundary), top_s=rank)
                    primitive = None
                    cost_val = 0.0
                    try:
                        res = self.conversion_engine.convert(conv_ssd)
                        primitive = (
                            res.primitive.name
                            if hasattr(res.primitive, "name")
                            else str(res.primitive)
                        )
                        cost_val = getattr(res, "cost", 0.0)
                    except Exception:
                        primitive = None
                    if layer and primitive not in {"LW", "ST"}:
                        primitive = layer.primitive
                    primitive = primitive or "Full"
                    try:
                        if primitive == "B2B":
                            try:
                                sim_obj.ingest(current_ssd)
                            except Exception:
                                if target == Backend.TABLEAU:
                                    rep = self.conversion_engine.convert_boundary_to_tableau(conv_ssd)
                                elif target == Backend.DECISION_DIAGRAM:
                                    rep = self.conversion_engine.convert_boundary_to_dd(conv_ssd)
                                else:
                                    rep = self.conversion_engine.convert_boundary_to_statevector(conv_ssd)
                                sim_obj.ingest(rep)
                        elif primitive == "LW":
                            state = current_sim.statevector()
                            rep = self.conversion_engine.extract_local_window(state, boundary)
                            sim_obj.ingest(rep)
                        elif primitive == "ST":
                            rep = self.conversion_engine.build_bridge_tensor(conv_ssd, conv_ssd)
                            sim_obj.ingest(rep)
                        else:
                            if target == Backend.TABLEAU:
                                rep = self.conversion_engine.convert_boundary_to_tableau(conv_ssd)
                            elif target == Backend.DECISION_DIAGRAM:
                                rep = self.conversion_engine.convert_boundary_to_dd(conv_ssd)
                            else:
                                rep = self.conversion_engine.convert_boundary_to_statevector(conv_ssd)
                            sim_obj.ingest(rep)
                    except Exception:
                        sim_obj.load(circuit.num_qubits)
                    circuit.ssd.conversions.append(
                        ConversionLayer(
                            boundary=tuple(boundary),
                            source=current_backend,
                            target=target,
                            rank=rank,
                            frontier=len(boundary),
                            primitive=primitive,
                            cost=Cost(time=cost_val, memory=0.0),
                        )
                    )
                current_sim = sim_obj
                current_backend = target
                for k in list(sims.keys()):
                    if k[0] == qubits and k != key:
                        sims.pop(k)

            est_cost = self._estimate_cost(target, len(qubits), len(segment))

            tracemalloc.start()
            start_time = time.perf_counter()

            if step.parallel and len(step.parallel) > 1:
                groups: List[List] = [[] for _ in step.parallel]
                mapping = {q: idx for idx, grp in enumerate(step.parallel) for q in grp}

                for gate in segment:
                    grp = mapping[gate.qubits[0]]
                    groups[grp].append(gate)

                def run_group(glist):
                    for g in glist:
                        current_sim.apply_gate(g.gate, g.qubits, g.params)

                with ThreadPoolExecutor() as executor:
                    executor.map(run_group, groups)
            else:
                for gate in segment:
                    current_sim.apply_gate(gate.gate, gate.qubits, gate.params)

            elapsed = time.perf_counter() - start_time
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            observed = Cost(time=elapsed, memory=float(peak))

            # Update cost model based on observation
            coeff = {
                Backend.STATEVECTOR: ("sv_gate", "sv_mem"),
                Backend.MPS: ("mps_gate", "mps_mem"),
                Backend.TABLEAU: ("tab_gate", "tab_mem"),
                Backend.DECISION_DIAGRAM: ("dd_gate", "dd_mem"),
            }[target]
            updates: Dict[str, float] = {}
            est = self.planner.estimator
            gate_key, mem_key = coeff
            if est_cost.time > 0:
                updates[gate_key] = est.coeff[gate_key] * observed.time / est_cost.time
            if est_cost.memory > 0 and observed.memory > 0:
                updates[mem_key] = est.coeff[mem_key] * observed.memory / est_cost.memory
            if updates:
                est.update_coefficients(updates)

            trigger_replan = False
            if observed.time > est_cost.time:
                trigger_replan = True
            if monitor and monitor(step, observed, est_cost):
                trigger_replan = True
            if trigger_replan:
                remaining = Circuit(circuit.gates[step.end :])
                replanned = self.planner.cache_lookup(remaining.gates, backend)
                if replanned is None:
                    replanned = self.planner.plan(remaining, backend=backend)
                offset = step.end
                new_steps = [
                    PlanStep(s.start + offset, s.end + offset, s.backend, parallel=s.parallel)
                    for s in replanned.steps
                ]
                steps = steps[: i + 1] + new_steps
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
    def _estimate_cost(self, backend: Backend, n: int, m: int) -> Cost:
        est = self.planner.estimator
        if backend == Backend.TABLEAU:
            return est.tableau(n, m)
        if backend == Backend.MPS:
            return est.mps(n, m, chi=4)
        if backend == Backend.DECISION_DIAGRAM:
            return est.decision_diagram(num_gates=m, frontier=n)
        return est.statevector(n, m)
