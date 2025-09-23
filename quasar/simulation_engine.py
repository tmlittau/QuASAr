from __future__ import annotations

"""High level orchestration utilities for circuit simulation.

This module exposes :class:`SimulationEngine` which ties together the
:class:`~quasar.analyzer.CircuitAnalyzer`, :class:`~quasar.planner.Planner`,
:class:`~quasar.scheduler.Scheduler` and the
:class:`quasar_convert.ConversionEngine`.  It provides a compact API for
users that simply want to simulate a circuit and obtain both the final
:class:`~quasar.ssd.SSD` descriptor and a collection of execution metrics.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math
import os
import time

try:  # Optional dependency used for memory discovery when available
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore

from .circuit import Circuit
from .analyzer import CircuitAnalyzer, AnalysisResult
from .planner import Planner, PlanResult
from .method_selector import NoFeasibleBackendError
from .scheduler import Scheduler
from .ssd import SSD
from .cost import CostEstimator, Backend
from quasar_convert import ConversionEngine

STATEVECTOR_MEMORY_ENV = "QUASAR_STATEVECTOR_MAX_MEMORY_BYTES"
"""Environment variable overriding automatic statevector memory limits."""

DEFAULT_MEMORY_BYTES = 64 * 1024 ** 3
"""Fallback memory budget (64 GiB) used when detection fails."""


def _autodetect_memory_ceiling() -> float:
    """Return a best-effort memory ceiling for dense statevectors."""

    env = os.getenv(STATEVECTOR_MEMORY_ENV)
    if env:
        try:
            value = int(env)
            if value > 0:
                return float(value)
        except ValueError:
            pass

    if psutil is not None:
        try:
            available = psutil.virtual_memory().available
            if available > 0:
                return float(available)
        except Exception:
            pass

    return float(DEFAULT_MEMORY_BYTES)


def _coerce_explicit_threshold(value: float | None) -> tuple[bool, float | None]:
    """Normalize explicit thresholds and flag whether a user override exists."""

    if value is None:
        return False, None
    if math.isinf(value) or value <= 0:
        return True, None
    return True, float(value)


def _sanitize_threshold(value: float | None) -> float | None:
    """Return ``value`` if it denotes a positive memory budget."""

    if value is None:
        return None
    if math.isinf(value) or value <= 0:
        return None
    return float(value)


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
    backend_switches:
        Number of times execution changed between backends.
    conversion_durations:
        Wall-clock durations for each state conversion.
    plan_cache_hits:
        Number of times a plan was reused from the planner cache.
    fidelity:
        Optional fidelity of the final state against a supplied reference.
    """

    ssd: SSD
    analysis: AnalysisResult
    plan: PlanResult
    analysis_time: float = 0.0
    planning_time: float = 0.0
    execution_time: float = 0.0
    backend_switches: int = 0
    conversion_durations: List[float] = field(default_factory=list)
    plan_cache_hits: int = 0
    fidelity: float | None = None


class SimulationEngine:
    """Compose analyzer, planner and scheduler into a single entry point.

    The engine captures a memory ceiling during construction using the
    ``QUASAR_STATEVECTOR_MAX_MEMORY_BYTES`` environment variable, the system's
    available memory (when psutil is installed), or a 64 GiB fallback.  This
    threshold is forwarded to the planner so that dense statevector execution
    respects the configured resource limits by default.
    """

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
        override_set, override_threshold = _coerce_explicit_threshold(memory_threshold)

        if planner is None:
            resolved_threshold = (
                override_threshold if override_set else _autodetect_memory_ceiling()
            )
            self.planner = Planner(estimator=estimator, max_memory=resolved_threshold)
        else:
            self.planner = planner
            if override_set:
                self.planner.max_memory = override_threshold
                resolved_threshold = override_threshold
            else:
                resolved_threshold = _sanitize_threshold(getattr(self.planner, "max_memory", None))

        if scheduler is None:
            self.scheduler = Scheduler(planner=self.planner, conversion_engine=ce)
        else:
            self.scheduler = scheduler
            if getattr(self.scheduler, "planner", None) is not self.planner:
                self.scheduler.planner = self.planner
            if getattr(self.scheduler, "conversion_engine", None) is None:
                self.scheduler.conversion_engine = ce

        self.conversion_engine = ce
        self.memory_threshold = resolved_threshold

    # ------------------------------------------------------------------
    def simulate(
        self,
        circuit: Circuit,
        *,
        memory_threshold: float | None = None,
        backend: Backend | None = None,
        target_accuracy: float | None = None,
        max_time: float | None = None,
        optimization_level: int | None = None,
        reference_state: List[complex] | None = None,
    ) -> SimulationResult:
        """Simulate ``circuit`` and return the final :class:`SSD` and metrics.

        Parameters
        ----------
        circuit:
            Circuit to simulate.
        memory_threshold:
            Optional memory ceiling in bytes for planning. When ``None`` the
            engine reuses the limit determined during initialisation.  Passing
            a non-positive or infinite value disables the limit for this
            invocation.
        backend:
            Optional backend hint used during planning and scheduling.
        target_accuracy:
            Desired lower bound on simulation fidelity.  Forwarded to the
            planner which adjusts the cost model accordingly.
        max_time:
            Upper bound on the estimated execution time in seconds.  Plans
            exceeding this value raise a :class:`ValueError`.
        optimization_level:
            Heuristic tuning knob influencing planner and scheduler behaviour.
        reference_state:
            Optional statevector used to compute fidelity of the final state.
        """

        start = time.perf_counter()
        analyzer = CircuitAnalyzer(circuit, estimator=self.planner.estimator)
        analysis = analyzer.analyze()
        analysis_time = time.perf_counter() - start

        start = time.perf_counter()
        override_set, override_threshold = _coerce_explicit_threshold(memory_threshold)
        threshold = override_threshold if override_set else self.memory_threshold
        cache_hits_before = self.planner.cache_hits
        if (
            threshold is None
            and self.scheduler.should_use_quick_path(
                circuit,
                backend=backend,
                max_time=max_time,
                optimization_level=optimization_level,
            )
        ):
            plan = self.scheduler.prepare_run(
                circuit,
                analysis=analysis,
                backend=backend,
                target_accuracy=target_accuracy,
                max_time=max_time,
                optimization_level=optimization_level,
                max_memory=threshold,
            )
        else:
            try:
                plan = self.planner.plan(
                    circuit,
                    analysis=analysis,
                    max_memory=threshold,
                    backend=backend,
                    target_accuracy=target_accuracy,
                    max_time=max_time,
                    optimization_level=optimization_level,
                )
            except ValueError as exc:
                if str(exc) == "Requested backend exceeds memory threshold":
                    raise NoFeasibleBackendError(str(exc)) from exc
                raise
        planning_time = time.perf_counter() - start
        planning_cache_hits = self.planner.cache_hits - cache_hits_before

        start = time.perf_counter()
        ssd, metrics = self.scheduler.run(
            circuit,
            plan,
            analysis=analysis,
            backend=backend,
            target_accuracy=target_accuracy,
            max_time=max_time,
            optimization_level=optimization_level,
            instrument=True,
            reference_state=reference_state,
        )
        execution_time = time.perf_counter() - start
        total_cache_hits = planning_cache_hits + metrics.plan_cache_hits

        return SimulationResult(
            ssd=ssd,
            analysis=analysis,
            plan=plan,
            analysis_time=analysis_time,
            planning_time=planning_time,
            execution_time=execution_time,
            backend_switches=metrics.backend_switches,
            conversion_durations=metrics.conversion_durations,
            plan_cache_hits=total_cache_hits,
            fidelity=metrics.fidelity,
        )


__all__ = ["SimulationEngine", "SimulationResult"]
