from __future__ import annotations

"""High level orchestration utilities for circuit simulation.

This module exposes :class:`SimulationEngine` which ties together the
:class:`~quasar.analyzer.CircuitAnalyzer`, :class:`~quasar.planner.Planner`,
:class:`~quasar.scheduler.Scheduler` and the
:class:`quasar_convert.ConversionEngine`.  It provides a compact API for
users that simply want to simulate a circuit and obtain both the final
:class:`~quasar.ssd.SSD` descriptor and a collection of execution metrics.
"""

from dataclasses import dataclass
from typing import Optional

from .circuit import Circuit
from .analyzer import CircuitAnalyzer, AnalysisResult
from .planner import Planner, PlanResult
from .scheduler import Scheduler
from .ssd import SSD
from .cost import CostEstimator
from quasar_convert import ConversionEngine


@dataclass
class SimulationResult:
    """Container bundling the outcome of :func:`SimulationEngine.simulate`.

    Attributes
    ----------
    ssd:
        Final subsystem descriptor obtained after executing the circuit.
    analysis:
        Static circuit analysis information produced before execution.
    plan:
        The execution plan derived by :class:`Planner`.
    """

    ssd: SSD
    analysis: AnalysisResult
    plan: PlanResult


class SimulationEngine:
    """Compose analyzer, planner and scheduler into a single entry point."""

    def __init__(
        self,
        *,
        planner: Planner | None = None,
        scheduler: Scheduler | None = None,
        conversion_engine: ConversionEngine | None = None,
        estimator: Optional[CostEstimator] = None,
    ) -> None:
        ce = conversion_engine or ConversionEngine()
        self.planner = planner or Planner(estimator=estimator)
        # Reuse the planner and conversion engine when creating the scheduler
        self.scheduler = scheduler or Scheduler(planner=self.planner, conversion_engine=ce)
        self.conversion_engine = ce

    # ------------------------------------------------------------------
    def simulate(self, circuit: Circuit) -> SimulationResult:
        """Simulate ``circuit`` and return the final :class:`SSD` and metrics."""

        analyzer = CircuitAnalyzer(circuit, estimator=self.planner.estimator)
        analysis = analyzer.analyze()
        plan = self.planner.plan(circuit)
        ssd = self.scheduler.run(circuit)
        return SimulationResult(ssd=ssd, analysis=analysis, plan=plan)


__all__ = ["SimulationEngine", "SimulationResult"]
