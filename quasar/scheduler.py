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

        # Track qubit sets for each fragment to identify cross-fragment gates
        step_qubits: List[set[int]] = [
            {q for g in circuit.gates[s.start : s.end] for q in g.qubits}
            for s in steps
        ]

        current_backend = None
        current_sim = None
        prev_sim = None
        prev_qubits: set[int] = set()

        i = 0
        while i < len(steps):
            step = steps[i]
            target = step.backend
            backend = self.backends[target]
            qubits = step_qubits[i]

            gates = circuit.gates[step.start : step.end]

            # Determine whether this fragment contains gates that span the
            # previous fragment. This happens when a gate touches qubits from
            # ``prev_qubits`` and qubits outside that set simultaneously.
            spans_previous = any(
                any(q in prev_qubits for q in g.qubits)
                and any(q not in prev_qubits for q in g.qubits)
                for g in gates
            )

            # Prepare backend and perform conversions when switching
            if current_sim is None:
                backend.load(circuit.num_qubits)
                current_sim = backend
            elif backend is not current_sim:
                if spans_previous:
                    # Keep ``current_sim`` alive for bridge operations and load
                    # the new backend separately.
                    backend.load(circuit.num_qubits)
                    prev_sim = current_sim
                    current_sim = backend
                else:
                    ssd = current_sim.extract_ssd()
                    self.conversion_engine.convert(ssd)
                    backend.load(circuit.num_qubits)
                    current_sim = backend

            # Execute gates, building bridge tensors when required
            for gate in gates:
                crosses = prev_sim is not None and (
                    any(q in prev_qubits for q in gate.qubits)
                    and any(q not in prev_qubits for q in gate.qubits)
                )
                if crosses and prev_sim is not None:
                    boundary = sorted(prev_qubits & set(gate.qubits))
                    left = self.conversion_engine.extract_ssd(boundary, 0)
                    right = self.conversion_engine.extract_ssd(
                        sorted(set(gate.qubits) - prev_qubits), 0
                    )
                    tensor = self.conversion_engine.build_bridge_tensor(left, right)
                    prev_sim.apply_gate("BRIDGE", boundary)
                    current_sim.apply_gate("BRIDGE", boundary)
                    prev_sim.apply_gate(gate.gate, gate.qubits, gate.params)
                    current_sim.apply_gate(gate.gate, gate.qubits, gate.params)
                else:
                    current_sim.apply_gate(gate.gate, gate.qubits, gate.params)

            # After processing the gates we can discard the previous fragment if
            # a bridge was used and convert its state for bookkeeping.
            if prev_sim is not None and spans_previous:
                ssd = prev_sim.extract_ssd()
                self.conversion_engine.convert(ssd)
                prev_sim = None

            if monitor:
                frag = gates
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

            prev_qubits = qubits
            current_backend = target
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
