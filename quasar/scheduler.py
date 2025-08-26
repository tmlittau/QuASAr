from __future__ import annotations

"""Execution scheduler for QuASAr."""

from dataclasses import dataclass
from typing import Callable, Dict, List

from .planner import Planner, PlanStep
from .cost import Backend
from .circuit import Circuit
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
    def run(self, circuit: Circuit, monitor: CostHook | None = None) -> None:
        """Execute ``circuit`` according to a planner-derived schedule.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        monitor:
            Optional callback receiving ``(step, cost)``.  If the callback
            returns ``True`` the scheduler re-plans the remaining gates starting
            from ``step.end``.
        """

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
                self.conversion_engine.convert(ssd)
                backend.load(circuit.num_qubits)
            current_sim = backend
            current_backend = target

            for gate in circuit.gates[step.start : step.end]:
                current_sim.apply_gate(gate.gate, gate.qubits, gate.params)

            if monitor:
                frag = circuit.gates[step.start : step.end]
                qubits = {q for g in frag for q in g.qubits}
                cost = self._estimate_cost(target, len(qubits), len(frag))
                if monitor(step, cost):
                    remaining = Circuit(circuit.gates[step.end :])
                    replanned = self.planner.plan(remaining)
                    offset = step.end
                    new_steps = [
                        PlanStep(s.start + offset, s.end + offset, s.backend)
                        for s in replanned.steps
                    ]
                    steps = steps[: i + 1] + new_steps
            i += 1

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
