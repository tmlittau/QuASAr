from __future__ import annotations

"""Utility helpers for executing benchmarks.

This module defines :class:`BenchmarkRunner` which provides a light-weight
API for executing circuits on different backends while collecting timing data.
Two entry points are exposed:

``run``
    Execute a circuit on an arbitrary backend or callable.

``run_quasar``
    Specialised variant that measures QuASAr execution via
    :class:`~quasar.scheduler.Scheduler`.  Planning and other analysis steps
    are performed ahead of time so that only the actual ``Scheduler.run`` call
    contributes to the recorded runtime.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import time

try:  # ``pandas`` is optional; benchmarks fall back to plain records.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore

# Type for objects that expose a ``run`` method
RunCallable = Callable[[Any], Any]


@dataclass
class BenchmarkRunner:
    """Execute circuits on various simulators and collect timing data."""

    results: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    def _invoke(self, backend: Any, circuit: Any, **kwargs: Any) -> Any:
        """Call ``backend`` with ``circuit``.

        ``backend`` may expose a ``run`` method or be directly callable.  Any
        return value is forwarded to the caller but also stored in the result
        record for completeness.  Additional keyword arguments are forwarded to
        the backend invocation.  This allows benchmarks to control whether the
        backend should return a state representation or merely perform the
        simulation.
        """

        if hasattr(backend, "run"):
            return backend.run(circuit, **kwargs)
        if callable(backend):
            return backend(circuit, **kwargs)
        raise TypeError("backend must be callable or provide a 'run' method")

    # ------------------------------------------------------------------
    def run(self, circuit: Any, backend: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute ``circuit`` on ``backend`` and record the runtime.

        If ``backend`` provides a :meth:`prepare` method, it is invoked prior
        to measurement.  The time spent in this step is recorded separately so
        that only the actual simulation call contributes to the ``run_time``
        measurement.  Any keyword arguments are forwarded to the backend's
        ``run`` method.
        """

        prepared = circuit
        prepare_time = 0.0
        if hasattr(backend, "prepare"):
            start_prepare = time.perf_counter()
            prepared = backend.prepare(circuit)
            prepare_time = time.perf_counter() - start_prepare

        start_run = time.perf_counter()
        result = self._invoke(backend, prepared, **kwargs)
        run_time = time.perf_counter() - start_run
        record = {
            "framework": getattr(backend, "name", backend.__class__.__name__),
            "prepare_time": prepare_time,
            "run_time": run_time,
            "result": result,
        }
        self.results.append(record)
        return record

    # ------------------------------------------------------------------
    def run_quasar(self, circuit: Any, engine: Any) -> Dict[str, Any]:
        """Execute ``circuit`` using a QuASAr scheduler ``engine``.

        Planning is performed prior to measurement so that only the actual
        :meth:`quasar.scheduler.Scheduler.run` invocation is timed.
        ``engine`` may either be a :class:`~quasar.scheduler.Scheduler` or an
        object providing ``scheduler`` and ``planner`` attributes (e.g.,
        :class:`~quasar.simulation_engine.SimulationEngine`).
        """

        scheduler = getattr(engine, "scheduler", engine)
        planner = getattr(engine, "planner", getattr(scheduler, "planner", None))
        prepare_time = 0.0
        if planner is not None:
            start_prepare = time.perf_counter()
            planner.plan(circuit)
            prepare_time = time.perf_counter() - start_prepare
        start_run = time.perf_counter()
        result = scheduler.run(circuit)
        run_time = time.perf_counter() - start_run
        record = {
            "framework": "quasar",
            "prepare_time": prepare_time,
            "run_time": run_time,
            "result": result,
        }
        self.results.append(record)
        return record

    # ------------------------------------------------------------------
    def dataframe(self) -> "pd.DataFrame | List[Dict[str, Any]]":
        """Return collected results as a :class:`pandas.DataFrame` if available.

        The returned data includes separate ``prepare_time`` and ``run_time``
        columns for downstream analysis.
        """

        if pd is None:
            return self.results
        return pd.DataFrame(self.results)


__all__ = ["BenchmarkRunner"]
