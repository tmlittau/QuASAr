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
import time

from .circuit import Circuit
from .analyzer import CircuitAnalyzer, AnalysisResult
from .planner import Planner, PlanResult
from .scheduler import Scheduler
from .ssd import SSD
from .cost import CostEstimator, Backend
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
    analysis_time, planning_time, execution_time:
        Wall-clock durations for each phase of :meth:`SimulationEngine.simulate`.
    """

    ssd: SSD
    analysis: AnalysisResult
    plan: PlanResult
    analysis_time: float = 0.0
    planning_time: float = 0.0
    execution_time: float = 0.0


class SimulationEngine:
    """Compose analyzer, planner and scheduler into a single entry point."""

    def __init__(
        self,
        *,
        planner: Planner | None = None,
        scheduler: Scheduler | None = None,
        conversion_engine: ConversionEngine | None = None,
        estimator: Optional[CostEstimator] = None,
        memory_threshold: float | None = None,
    ) -> None:
        ce = conversion_engine or ConversionEngine()
        self.planner = planner or Planner(estimator=estimator, max_memory=memory_threshold)
        # Reuse the planner and conversion engine when creating the scheduler
        self.scheduler = scheduler or Scheduler(planner=self.planner, conversion_engine=ce)
        self.conversion_engine = ce
        self.memory_threshold = memory_threshold

    # ------------------------------------------------------------------
    def simulate(
        self,
        circuit: Circuit,
        *,
        memory_threshold: float | None = None,
        backend: Backend | None = None,
    ) -> SimulationResult:
        """Simulate ``circuit`` and return the final :class:`SSD` and metrics."""

        start = time.perf_counter()
        analyzer = CircuitAnalyzer(circuit, estimator=self.planner.estimator)
        analysis = analyzer.analyze()
        analysis_time = time.perf_counter() - start

        start = time.perf_counter()
        threshold = (
            memory_threshold if memory_threshold is not None else self.memory_threshold
        )
        if (
            memory_threshold is None
            and self.scheduler.should_use_quick_path(circuit, backend=backend)
        ):
            plan = self.scheduler.prepare_run(circuit, backend=backend)
        else:
            plan = self.planner.plan(circuit, max_memory=threshold, backend=backend)
        planning_time = time.perf_counter() - start

        start = time.perf_counter()
        ssd = self.scheduler.run(circuit, plan, backend=backend)
        execution_time = time.perf_counter() - start

        return SimulationResult(
            ssd=ssd,
            analysis=analysis,
            plan=plan,
            analysis_time=analysis_time,
            planning_time=planning_time,
            execution_time=execution_time,
        )


__all__ = ["SimulationEngine", "SimulationResult"]
