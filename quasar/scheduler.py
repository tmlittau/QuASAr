from __future__ import annotations

"""Execution scheduler for QuASAr."""

from dataclasses import dataclass
from typing import Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor

from .planner import Planner, PlanStep
from .cost import Backend, Cost
from .circuit import Circuit
from .ssd import SSD, ConversionLayer, SSDPartition
from .backends import (
    StatevectorBackend,
    MPSBackend,
    StimBackend,
    DecisionDiagramBackend,
)
from quasar_convert import ConversionEngine, SSD as CESD

# Type alias for cost monitoring hook
CostHook = Callable[[PlanStep, float], bool]


@dataclass
class Scheduler:
    """Coordinate execution of planned circuit partitions."""

    planner: Planner | None = None
    conversion_engine: ConversionEngine | None = None
    backends: Dict[Backend, object] | None = None

    def __post_init__(self) -> None:
        self.planner = self.planner or Planner()
        self.conversion_engine = self.conversion_engine or ConversionEngine()
        if self.backends is None:
            self.backends = {
                Backend.STATEVECTOR: StatevectorBackend(),
                Backend.MPS: MPSBackend(),
                Backend.TABLEAU: StimBackend(),
                Backend.DECISION_DIAGRAM: DecisionDiagramBackend(),
            }

    # ------------------------------------------------------------------
    def run(self, circuit: Circuit, monitor: CostHook | None = None) -> SSD:
        """Execute ``circuit`` according to a planner-derived schedule.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        monitor:
            Optional callback receiving ``(step, cost)``.  If the callback
            returns ``True`` the scheduler re-plans the remaining gates starting
            from ``step.end``.
        Returns
        -------
        SSD
            Descriptor of the simulated state after all gates have been
            executed.
        """

        plan = self.planner.cache_lookup(circuit.gates)
        if plan is None:
            plan = self.planner.plan(circuit)
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
                backend = type(self.backends[target])()
                backend.load(circuit.num_qubits)
                sims[key] = backend
            backend = sims[key]

            if backend is not current_sim:
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
                                backend.ingest(current_ssd)
                            except Exception:
                                if target == Backend.TABLEAU:
                                    rep = self.conversion_engine.convert_boundary_to_tableau(conv_ssd)
                                elif target == Backend.DECISION_DIAGRAM:
                                    rep = self.conversion_engine.convert_boundary_to_dd(conv_ssd)
                                else:
                                    rep = self.conversion_engine.convert_boundary_to_statevector(conv_ssd)
                                backend.ingest(rep)
                        elif primitive == "LW":
                            state = current_sim.statevector()
                            rep = self.conversion_engine.extract_local_window(state, boundary)
                            backend.ingest(rep)
                        elif primitive == "ST":
                            rep = self.conversion_engine.build_bridge_tensor(conv_ssd, conv_ssd)
                            backend.ingest(rep)
                        else:
                            if target == Backend.TABLEAU:
                                rep = self.conversion_engine.convert_boundary_to_tableau(conv_ssd)
                            elif target == Backend.DECISION_DIAGRAM:
                                rep = self.conversion_engine.convert_boundary_to_dd(conv_ssd)
                            else:
                                rep = self.conversion_engine.convert_boundary_to_statevector(conv_ssd)
                            backend.ingest(rep)
                    except Exception:
                        backend.load(circuit.num_qubits)
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
                current_sim = backend
                current_backend = target
                for k in list(sims.keys()):
                    if k[0] == qubits and k != key:
                        sims.pop(k)

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

            if monitor:
                frag = circuit.gates[step.start : step.end]
                qubits = {q for g in frag for q in g.qubits}
                cost = self._estimate_cost(target, len(qubits), len(frag))
                if monitor(step, cost):
                    remaining = Circuit(circuit.gates[step.end :])
                    replanned = self.planner.cache_lookup(remaining.gates)
                    if replanned is None:
                        replanned = self.planner.plan(remaining)
                    offset = step.end
                    new_steps = [
                        PlanStep(s.start + offset, s.end + offset, s.backend)
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
    def _estimate_cost(self, backend: Backend, n: int, m: int) -> float:
        est = self.planner.estimator
        if backend == Backend.TABLEAU:
            return est.tableau(n, m).time
        if backend == Backend.MPS:
            return est.mps(n, m, chi=4).time
        if backend == Backend.DECISION_DIAGRAM:
            return est.decision_diagram(num_gates=m, frontier=n).time
        return est.statevector(n, m).time
