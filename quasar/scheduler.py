from __future__ import annotations

"""Execution scheduler for QuASAr."""

from dataclasses import dataclass
from typing import Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor

from .planner import Planner, PlanStep
from .cost import Backend
from .circuit import Circuit
from .ssd import SSD
from .backends import (
    StatevectorBackend,
    MPSBackend,
    StimBackend,
    DecisionDiagramBackend,
)
from quasar_convert import ConversionEngine

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

        current_backend = None
        current_sim = None
        i = 0
        while i < len(steps):
            step = steps[i]
            target = step.backend
            backend = self.backends[target]

            # Prepare backend and perform conversions when switching
            if current_sim is None:
                backend.load(circuit.num_qubits)
            elif backend is not current_sim:
                ssd = current_sim.extract_ssd()
                layer = next(
                    (
                        c
                        for c in circuit.ssd.conversions
                        if c.source == current_backend and c.target == target
                    ),
                    None,
                )
                boundary = list(layer.boundary) if layer else []
                rank = layer.rank if layer else 0
                primitive = layer.primitive if layer else None
                if primitive is None:
                    try:
                        res = self.conversion_engine.convert(ssd)  # type: ignore[arg-type]
                        primitive = (
                            res.primitive.name
                            if hasattr(res.primitive, "name")
                            else str(res.primitive)
                        )
                    except Exception:
                        primitive = None
                try:
                    if primitive == "B2B":
                        rep = self.conversion_engine.extract_ssd(boundary, rank)
                    elif primitive == "LW":
                        dim = 1 << len(boundary)
                        state = [0j] * dim
                        if dim:
                            state[0] = 1.0 + 0j
                        rep = self.conversion_engine.extract_local_window(state, boundary)
                    elif primitive == "ST":
                        left = self.conversion_engine.extract_ssd(boundary, rank)
                        right = self.conversion_engine.extract_ssd(boundary, rank)
                        rep = self.conversion_engine.build_bridge_tensor(left, right)
                    else:
                        raise ValueError("unknown primitive")
                    backend.ingest(rep)
                except Exception:
                    try:
                        if target == Backend.TABLEAU:
                            state = self.conversion_engine.convert_boundary_to_tableau(ssd)
                        elif target == Backend.DECISION_DIAGRAM:
                            state = self.conversion_engine.convert_boundary_to_dd(ssd)
                        else:
                            state = self.conversion_engine.convert_boundary_to_statevector(ssd)
                        backend.ingest(state)
                    except Exception:
                        backend.load(circuit.num_qubits)
            current_sim = backend
            current_backend = target

            segment = circuit.gates[step.start : step.end]
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

        if current_sim is not None:
            return current_sim.extract_ssd()
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
